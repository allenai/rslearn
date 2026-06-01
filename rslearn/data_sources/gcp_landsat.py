"""Data source for Landsat imagery from gs://gee-public-data-landsat.

gs://gee-public-data-landsat contains Collection 2 Landsat data (L1 and L2)
for all missions (Landsat 1-9). The bucket is requester-pays.

Scene discovery uses the bucket's index.csv.gz, which is parsed once and
indexed into an on-disk rtree for fast spatial lookups.
"""

import csv
import gzip
import io
import json
import logging
import os
import tempfile
from collections.abc import Generator
from datetime import datetime
from enum import StrEnum
from typing import Any, BinaryIO

import dateutil.parser
import shapely
import tqdm
from google.cloud import storage
from typing_extensions import override
from upath import UPath

import rslearn.data_sources.utils
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.direct_materialize_data_source import (
    DirectMaterializeDataSource,
)
from rslearn.data_sources.utils import MatchedItemGroup
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.fsspec import join_upath, open_atomic
from rslearn.utils.geometry import STGeometry, flatten_shape, split_at_antimeridian
from rslearn.utils.rtree_index import RtreeIndex, get_cached_rtree

from .data_source import DataSourceContext, Item, QueryConfig

logger = logging.getLogger(__name__)

BUCKET_NAME = "gee-public-data-landsat"


class SpacecraftId(StrEnum):
    """Landsat spacecraft identifiers."""

    LANDSAT_1 = "LANDSAT_1"
    LANDSAT_2 = "LANDSAT_2"
    LANDSAT_3 = "LANDSAT_3"
    LANDSAT_4 = "LANDSAT_4"
    LANDSAT_5 = "LANDSAT_5"
    LANDSAT_7 = "LANDSAT_7"
    LANDSAT_8 = "LANDSAT_8"
    LANDSAT_9 = "LANDSAT_9"


class CollectionCategory(StrEnum):
    """Landsat collection tiers."""

    T1 = "T1"
    T2 = "T2"
    RT = "RT"


class DataType(StrEnum):
    """Landsat processing levels."""

    L1GS = "L1GS"
    L1GT = "L1GT"
    L1TP = "L1TP"
    L2SP = "L2SP"
    L2SR = "L2SR"


# Band definitions per sensor family.
BANDS_OLI_TIRS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]
BANDS_ETM = ["B1", "B2", "B3", "B4", "B5", "B6_VCID_1", "B6_VCID_2", "B7", "B8"]
BANDS_TM = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
BANDS_MSS = ["B4", "B5", "B6", "B7"]


class LandsatItem(Item):
    """An item in the Landsat data source."""

    def __init__(
        self,
        name: str,
        geometry: STGeometry,
        blob_path: str,
        cloud_cover: float,
        spacecraft_id: str,
        data_type: str,
    ) -> None:
        """Creates a new LandsatItem.

        Args:
            name: unique name of the item (PRODUCT_ID).
            geometry: the spatial and temporal extent of the item.
            blob_path: path within the GCS bucket to the scene folder, e.g.
                "LC09/L1/02/231/062/LC09_L1TP_231062_20260426_20260426_02_T1/".
            cloud_cover: the scene's cloud cover percentage (0-100).
            spacecraft_id: the spacecraft identifier, e.g. "LANDSAT_8".
            data_type: the processing level, e.g. "L1TP".
        """
        super().__init__(name, geometry)
        self.blob_path = blob_path
        self.cloud_cover = cloud_cover
        self.spacecraft_id = spacecraft_id
        self.data_type = data_type

    @override
    def serialize(self) -> dict[str, Any]:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["blob_path"] = self.blob_path
        d["cloud_cover"] = self.cloud_cover
        d["spacecraft_id"] = self.spacecraft_id
        d["data_type"] = self.data_type
        return d

    @staticmethod
    @override
    def deserialize(d: dict[str, Any]) -> "LandsatItem":
        """Deserializes an item from a JSON-decoded dictionary."""
        item = super(LandsatItem, LandsatItem).deserialize(d)
        return LandsatItem(
            name=item.name,
            geometry=item.geometry,
            blob_path=d["blob_path"],
            cloud_cover=d["cloud_cover"],
            spacecraft_id=d["spacecraft_id"],
            data_type=d["data_type"],
        )


class Landsat(DirectMaterializeDataSource[LandsatItem]):
    """Data source for Landsat imagery on GCP's public Landsat GCS bucket.

    Uses gs://gee-public-data-landsat which contains Collection 2 data for all
    Landsat missions (1-9). The bucket is requester-pays.

    Scene discovery uses the bucket's index.csv.gz (~23M rows), which is parsed
    once and stored in an on-disk rtree index for fast spatial lookups.

    Required environment variables:
        GOOGLE_APPLICATION_CREDENTIALS: path to a GCP service account JSON key.
        GS_USER_PROJECT: GCP project for requester-pays billing (must match
            the project the service account has access to).
    """

    def __init__(
        self,
        index_cache_dir: str,
        spacecraft_id: list[SpacecraftId] | None = None,
        bands: list[str] | None = None,
        sort_by: str | None = None,
        collection_category: list[CollectionCategory] | None = None,
        data_type: list[DataType] | None = None,
        gcp_project: str | None = None,
        rtree_time_range: tuple[datetime, datetime] | None = None,
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Initialize a new Landsat instance.

        Args:
            index_cache_dir: directory to cache the index and rtree files.
            spacecraft_id: filter by mission, e.g. ["LANDSAT_8", "LANDSAT_9"].
                None means all missions.
            bands: which bands to expose. Defaults to layer config bands or
                OLI-TIRS bands (B1-B11).
            sort_by: "cloud_cover" or None (arbitrary order).
            collection_category: filter by tier, e.g. ["T1"]. None means all.
            data_type: filter by processing level, e.g. ["L1TP"]. None means all.
            gcp_project: GCP project for requester-pays billing.
            rtree_time_range: only index scenes within this time range.
                Restricting to a shorter period significantly speeds up rtree
                creation.
            context: the data source context.
        """
        # Determine bands from context, explicit argument, or default.
        if context.layer_config is not None:
            needed_bands: list[str] = []
            for band_set in context.layer_config.band_sets:
                for band in band_set.bands:
                    if band not in needed_bands:
                        needed_bands.append(band)
            effective_bands = needed_bands
        elif bands is not None:
            effective_bands = bands
        else:
            effective_bands = list(BANDS_OLI_TIRS)

        asset_bands = {band: [band] for band in effective_bands}
        super().__init__(asset_bands=asset_bands)

        if context.ds_path is not None:
            self.index_cache_dir = join_upath(context.ds_path, index_cache_dir)
        else:
            self.index_cache_dir = UPath(index_cache_dir)

        self.spacecraft_id_filter = (
            set(spacecraft_id) if spacecraft_id is not None else None
        )
        self.collection_category_filter = (
            set(collection_category) if collection_category is not None else None
        )
        self.data_type_filter = set(data_type) if data_type is not None else None
        self.rtree_time_range = rtree_time_range
        self.sort_by = sort_by
        self.effective_bands = effective_bands
        self.gcp_project = gcp_project

        self.index_cache_dir.mkdir(parents=True, exist_ok=True)

        self._storage_client = storage.Client()
        self._bucket = self._storage_client.bucket(
            BUCKET_NAME, user_project=gcp_project
        )

        def build_fn(index: RtreeIndex) -> None:
            """Build the rtree from index.csv.gz."""
            for item in tqdm.tqdm(self._read_index(), desc="Building rtree index"):
                for shp in flatten_shape(item.geometry.shp):
                    index.insert(shp.bounds, json.dumps(item.serialize()))

        self.rtree_index = get_cached_rtree(self.index_cache_dir, build_fn)

    # -------------------------------------------------------------------------
    # Index loading
    # -------------------------------------------------------------------------

    def _ensure_index_downloaded(self) -> UPath:
        """Download index.csv.gz if not already present.

        Returns:
            Path to the local index.csv.gz file.
        """
        index_gz_path = self.index_cache_dir / "index.csv.gz"
        if not index_gz_path.exists():
            logger.info("Downloading index.csv.gz from gs://%s ...", BUCKET_NAME)
            blob = self._bucket.blob("index.csv.gz")
            with open_atomic(index_gz_path, "wb") as f:
                blob.download_to_file(f)
            logger.info("Download complete.")
        return index_gz_path

    def _read_index(self) -> Generator[LandsatItem, None, None]:
        """Stream LandsatItems from index.csv.gz.

        Downloads the file if needed, then yields one item per CSV row.
        """
        index_gz_path = self._ensure_index_downloaded()

        with gzip.open(str(index_gz_path), "rt") as f:
            reader = csv.DictReader(f)
            for row in reader:
                product_id = row["PRODUCT_ID"]
                spacecraft = row["SPACECRAFT_ID"]
                dt = row["DATA_TYPE"]

                if (
                    self.spacecraft_id_filter is not None
                    and spacecraft not in self.spacecraft_id_filter
                ):
                    continue
                if (
                    self.data_type_filter is not None
                    and dt not in self.data_type_filter
                ):
                    continue
                if self.collection_category_filter is not None:
                    tier = product_id.rsplit("_", 1)[-1]
                    if tier not in self.collection_category_filter:
                        continue

                date_acquired = row["DATE_ACQUIRED"]
                if self.rtree_time_range is not None:
                    acq_date = datetime.strptime(date_acquired, "%Y-%m-%d")
                    if (
                        acq_date < self.rtree_time_range[0]
                        or acq_date > self.rtree_time_range[1]
                    ):
                        continue

                sensing_time = row["SENSING_TIME"]
                cloud_cover = float(row["CLOUD_COVER"])
                north_lat = float(row["NORTH_LAT"])
                south_lat = float(row["SOUTH_LAT"])
                west_lon = float(row["WEST_LON"])
                east_lon = float(row["EAST_LON"])
                base_url = row["BASE_URL"]

                gs_prefix = f"gs://{BUCKET_NAME}/"
                if base_url.startswith(gs_prefix):
                    blob_path = base_url[len(gs_prefix) :]
                else:
                    blob_path = base_url
                if not blob_path.endswith("/"):
                    blob_path += "/"

                ts = dateutil.parser.isoparse(sensing_time)

                geometry = STGeometry(
                    WGS84_PROJECTION,
                    shapely.box(west_lon, south_lat, east_lon, north_lat),
                    (ts, ts),
                )
                geometry = split_at_antimeridian(geometry)

                yield LandsatItem(
                    name=product_id,
                    geometry=geometry,
                    blob_path=blob_path,
                    cloud_cover=cloud_cover,
                    spacecraft_id=spacecraft,
                    data_type=dt,
                )

    # -------------------------------------------------------------------------
    # DataSource interface
    # -------------------------------------------------------------------------

    @override
    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[MatchedItemGroup[LandsatItem]]]:
        """Get items intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries.
            query_config: the query configuration.

        Returns:
            List of groups of items for each geometry.
        """
        wgs84_geometries = [geometry.to_wgs84() for geometry in geometries]

        groups: list[list[MatchedItemGroup[LandsatItem]]] = []
        for geometry, wgs84_geometry in zip(geometries, wgs84_geometries):
            if wgs84_geometry.time_range is None:
                raise ValueError(
                    "gcp_landsat.Landsat requires geometry time ranges to be set"
                )

            encoded_items: set[str] = set()
            for shp in flatten_shape(wgs84_geometry.shp):
                encoded_items.update(self.rtree_index.query(shp.bounds))

            cur_items: list[LandsatItem] = []
            for encoded_item in encoded_items:
                item = LandsatItem.deserialize(json.loads(encoded_item))
                if not item.geometry.intersects_time_range(wgs84_geometry.time_range):
                    continue
                if not wgs84_geometry.shp.intersects(item.geometry.shp):
                    continue
                cur_items.append(item)

            if self.sort_by == "cloud_cover":
                cur_items.sort(
                    key=lambda item: (
                        item.cloud_cover if item.cloud_cover >= 0 else 100
                    )
                )
            elif self.sort_by is not None:
                raise ValueError(f"invalid sort_by setting ({self.sort_by})")

            cur_groups: list[MatchedItemGroup[LandsatItem]] = (
                rslearn.data_sources.utils.match_candidate_items_to_window(
                    geometry, cur_items, query_config
                )
            )
            groups.append(cur_groups)

        return groups

    @override
    def deserialize_item(self, serialized_item: dict) -> LandsatItem:
        """Deserializes an item from JSON-decoded data."""
        return LandsatItem.deserialize(serialized_item)

    # -------------------------------------------------------------------------
    # DirectMaterializeDataSource implementation
    # -------------------------------------------------------------------------

    @override
    def get_asset_url(self, item: LandsatItem, asset_key: str) -> str:
        """Get a gs:// URL for the band TIF.

        Args:
            item: the item.
            asset_key: the band name (e.g. "B4").

        Returns:
            a gs:// URL readable by rasterio.
        """
        blob_key = f"{item.blob_path}{item.name}_{asset_key}.TIF"
        return f"gs://{BUCKET_NAME}/{blob_key}"

    # -------------------------------------------------------------------------
    # Ingest
    # -------------------------------------------------------------------------

    @override
    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[LandsatItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into.
            items: the items to ingest.
            geometries: a list of geometries needed for each item.
        """
        for item in items:
            for band in self.effective_bands:
                band_names = [band]
                if tile_store.is_raster_ready(item, band_names):
                    continue

                blob_key = f"{item.blob_path}{item.name}_{band}.TIF"
                with tempfile.TemporaryDirectory() as tmp_dir:
                    fname = os.path.join(tmp_dir, f"{band}.tif")
                    blob = self._bucket.blob(blob_key)
                    logger.debug("Downloading %s", blob_key)
                    blob.download_to_filename(fname)
                    tile_store.write_raster_file(
                        item,
                        band_names,
                        UPath(fname),
                        time_range=item.geometry.time_range,
                    )

    # -------------------------------------------------------------------------
    # Retrieve
    # -------------------------------------------------------------------------

    def retrieve_item(
        self, item: LandsatItem
    ) -> Generator[tuple[str, BinaryIO], None, None]:
        """Retrieves the rasters corresponding to an item as file streams."""
        for band in self.effective_bands:
            blob_key = f"{item.blob_path}{item.name}_{band}.TIF"
            blob = self._bucket.blob(blob_key)
            buf = io.BytesIO()
            blob.download_to_file(buf)
            buf.seek(0)
            yield (f"{item.name}_{band}.TIF", buf)
