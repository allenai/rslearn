"""Data source for Landsat imagery from gs://gee-public-data-landsat.

gs://gee-public-data-landsat contains Collection 2 Landsat data (L1 and L2)
for all missions (Landsat 1-9). The bucket is requester-pays.

Scene discovery uses BigQuery table
earth-engine-public-data.geo_index.landsat_c2_index.
"""

import io
import json
import logging
import os
import tempfile
from collections.abc import Collection, Generator
from datetime import datetime
from enum import StrEnum
from typing import Any, BinaryIO

import shapely
import tqdm
from google.cloud import bigquery, storage
from typing_extensions import override
from upath import UPath

import rslearn.data_sources.utils
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.direct_materialize_data_source import (
    DirectMaterializeDataSource,
)
from rslearn.data_sources.utils import MatchedItemGroup
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.fsspec import join_upath
from rslearn.utils.geometry import STGeometry, flatten_shape, split_at_antimeridian
from rslearn.utils.grid_index import GridIndex
from rslearn.utils.rtree_index import RtreeIndex, get_cached_rtree

from .data_source import DataSourceContext, Item, QueryConfig
from .wrs2 import build_wrs2_grid_index, get_pathrows_for_geometry

logger = logging.getLogger(__name__)

BUCKET_NAME = "gee-public-data-landsat"
TABLE_NAME = "earth-engine-public-data.geo_index.landsat_c2_index"


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


class LandsatItem(Item):
    """An item in the Landsat data source."""

    def __init__(
        self,
        name: str,
        geometry: STGeometry,
        blob_path: str,
        cloud_cover: float,
        spacecraft_id: str | None,
        data_type: str,
        sensor_id: str | None = None,
    ) -> None:
        """Creates a new LandsatItem.

        Args:
            name: unique name of the item (PRODUCT_ID).
            geometry: the spatial and temporal extent of the item.
            blob_path: path within the GCS bucket to the scene folder, e.g.
                "LC09/L1/02/231/062/LC09_L1TP_231062_20260426_20260426_02_T1/".
            cloud_cover: the scene's cloud cover percentage (0-100).
            spacecraft_id: the spacecraft identifier, e.g. "LANDSAT_8".
            sensor_id: the sensor identifier, e.g. "OLI_TIRS".
            data_type: the processing level, e.g. "L1TP".
        """
        super().__init__(name, geometry)
        self.blob_path = blob_path
        self.cloud_cover = cloud_cover
        self.spacecraft_id = spacecraft_id
        self.sensor_id = sensor_id
        self.data_type = data_type

    @override
    def serialize(self) -> dict[str, Any]:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["blob_path"] = self.blob_path
        d["cloud_cover"] = self.cloud_cover
        d["spacecraft_id"] = self.spacecraft_id
        d["sensor_id"] = self.sensor_id
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
            sensor_id=d.get("sensor_id"),
        )


class Landsat(DirectMaterializeDataSource[LandsatItem]):
    """Data source for Landsat imagery on GCP's public Landsat GCS bucket.

    Uses gs://gee-public-data-landsat which contains Collection 2 data for all
    Landsat missions (1-9). The bucket is requester-pays.

    Scene discovery uses BigQuery table
    earth-engine-public-data.geo_index.landsat_c2_index.

    If use_rtree_index=True, one BigQuery scan populates an on-disk rtree for
    subsequent fast lookups. If use_rtree_index=False, each get_items call runs a
    single BigQuery query filtered by geometry time range, bounding box, and WRS
    path/row.

    Required environment variables:
        GOOGLE_APPLICATION_CREDENTIALS: path to a GCP service account JSON key.
        GS_USER_PROJECT: GCP project for requester-pays billing (must match
            the project the service account has access to).
    """

    def __init__(
        self,
        index_cache_dir: str,
        spacecraft_id: list[SpacecraftId] | None = None,
        sensor_id: list[str] | None = None,
        bands: list[str] | None = None,
        sort_by: str | None = None,
        collection_category: list[CollectionCategory] | None = None,
        data_type: list[DataType] | None = None,
        use_rtree_index: bool = True,
        gcp_project: str | None = None,
        rtree_time_range: tuple[datetime, datetime] | None = None,
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Initialize a new Landsat instance.

        Args:
            index_cache_dir: directory to cache the index and rtree files.
            spacecraft_id: filter by mission, e.g. ["LANDSAT_8", "LANDSAT_9"].
                None means all missions.
            sensor_id: filter by sensor, e.g. ["OLI_TIRS"]. None means all.
            bands: which bands to expose. Required if the layer config does not
                specify band sets.
            sort_by: "cloud_cover" or None (arbitrary order).
            collection_category: filter by tier, e.g. ["T1"]. None means all.
            data_type: filter by processing level, e.g. ["L1TP"]. None means all.
            use_rtree_index: whether to build and query local rtree index.
            gcp_project: GCP project for requester-pays billing.
            rtree_time_range: only index scenes within this time range.
                Restricting to a shorter period significantly speeds up rtree
                creation.
            context: the data source context.
        """
        # Determine bands from context or explicit argument.
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
            raise ValueError(
                "bands must be specified when no layer config band sets are provided"
            )

        asset_bands = {band: [band] for band in effective_bands}
        super().__init__(asset_bands=asset_bands)

        if context.ds_path is not None:
            self.index_cache_dir = join_upath(context.ds_path, index_cache_dir)
        else:
            self.index_cache_dir = UPath(index_cache_dir)

        self.spacecraft_id_filter = (
            set(spacecraft_id) if spacecraft_id is not None else None
        )
        self.sensor_id_filter = set(sensor_id) if sensor_id is not None else None
        self.collection_category_filter = (
            set(collection_category) if collection_category is not None else None
        )
        self.data_type_filter = set(data_type) if data_type is not None else None
        self.use_rtree_index = use_rtree_index
        self.rtree_time_range = rtree_time_range
        self.sort_by = sort_by
        self.effective_bands = effective_bands
        self.gcp_project = gcp_project

        self.index_cache_dir.mkdir(parents=True, exist_ok=True)

        self._storage_client = storage.Client()
        self._bucket = self._storage_client.bucket(
            BUCKET_NAME, user_project=gcp_project
        )
        self._bigquery_client: bigquery.Client | None = None
        self._wrs2_index: GridIndex | None = None

        self.rtree_index: RtreeIndex | None
        if self.use_rtree_index:

            def build_fn(index: RtreeIndex) -> None:
                """Build the rtree from BigQuery rows."""
                for item in self._read_bigquery(
                    desc="Building rtree index",
                    time_range=rtree_time_range,
                ):
                    for shp in flatten_shape(item.geometry.shp):
                        index.insert(shp.bounds, json.dumps(item.serialize()))

            self.rtree_index = get_cached_rtree(self.index_cache_dir, build_fn)
        else:
            self.rtree_index = None

    def _get_bigquery_client(self) -> bigquery.Client:
        """Lazily initialize BigQuery client."""
        if self._bigquery_client is None:
            self._bigquery_client = bigquery.Client()
        return self._bigquery_client

    def _get_wrs2_index(self) -> GridIndex:
        """Lazily initialize WRS2 index for direct BigQuery mode."""
        if self._wrs2_index is None:
            self._wrs2_index = build_wrs2_grid_index(self.index_cache_dir)
        return self._wrs2_index

    def _read_bigquery(
        self,
        desc: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        wgs84_bbox: tuple[float, float, float, float] | None = None,
        pathrows: set[tuple[str, str]] | None = None,
    ) -> Generator[LandsatItem, None, None]:
        """Read Landsat scenes from BigQuery table."""
        query_str = f"""
            SELECT  product_id, spacecraft_id, sensor_id, sensing_time,
                    data_type, collection_category, wrs_path, wrs_row,
                    cloud_cover, north_lat, south_lat, west_lon, east_lon,
                    base_url
            FROM    `{TABLE_NAME}`
            WHERE   product_id IS NOT NULL
                    AND sensing_time IS NOT NULL
                    AND cloud_cover IS NOT NULL
                    AND west_lon IS NOT NULL
                    AND south_lat IS NOT NULL
                    AND east_lon IS NOT NULL
                    AND north_lat IS NOT NULL
                    AND base_url IS NOT NULL
        """
        if time_range is not None:
            query_str += (
                f' AND sensing_time >= "{time_range[0]}"'
                f' AND sensing_time <= "{time_range[1]}"'
            )
        if wgs84_bbox is not None:
            query_str += (
                f" AND west_lon < {wgs84_bbox[2]}"
                f" AND east_lon > {wgs84_bbox[0]}"
                f" AND south_lat < {wgs84_bbox[3]}"
                f" AND north_lat > {wgs84_bbox[1]}"
            )
        if pathrows:
            pairs = [
                f"(wrs_path = {int(path)} AND wrs_row = {int(row)})"
                for path, row in sorted(pathrows)
            ]
            query_str += f" AND ({' OR '.join(pairs)})"

        def _add_in_filter(column_name: str, values: Collection[str] | None) -> None:
            nonlocal query_str
            if values is None:
                return
            formatted_values = ", ".join(f'"{v}"' for v in sorted(values))
            query_str += f" AND {column_name} IN ({formatted_values})"

        _add_in_filter("spacecraft_id", self.spacecraft_id_filter)
        _add_in_filter("sensor_id", self.sensor_id_filter)
        _add_in_filter("data_type", self.data_type_filter)
        _add_in_filter("collection_category", self.collection_category_filter)

        result = self._get_bigquery_client().query(query_str)
        if desc is not None:
            result = tqdm.tqdm(result, desc=desc)

        for row in result:
            product_id = row["product_id"]
            base_url = row["base_url"]

            # base_url is always a gs://BUCKET_NAME/... path without a trailing
            # slash, so convert it to a bucket-relative blob path ending in "/".
            gs_prefix = f"gs://{BUCKET_NAME}/"
            if not base_url.startswith(gs_prefix):
                raise ValueError(f"unexpected base_url {base_url}")
            blob_path = base_url[len(gs_prefix) :] + "/"

            ts = row["sensing_time"]

            geometry = STGeometry(
                WGS84_PROJECTION,
                shapely.box(
                    float(row["west_lon"]),
                    float(row["south_lat"]),
                    float(row["east_lon"]),
                    float(row["north_lat"]),
                ),
                (ts, ts),
            )
            geometry = split_at_antimeridian(geometry)

            yield LandsatItem(
                name=product_id,
                geometry=geometry,
                blob_path=blob_path,
                cloud_cover=float(row["cloud_cover"]),
                spacecraft_id=row["spacecraft_id"],
                data_type=row["data_type"],
                sensor_id=row["sensor_id"],
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

        for wgs84_geometry in wgs84_geometries:
            if wgs84_geometry.time_range is None:
                raise ValueError(
                    "gcp_landsat.Landsat requires geometry time ranges to be set"
                )

        if self.rtree_index is not None:
            candidate_items = self._get_candidate_items_index(wgs84_geometries)
        else:
            candidate_items = self._get_candidate_items_bigquery(wgs84_geometries)

        groups: list[list[MatchedItemGroup[LandsatItem]]] = []
        for geometry, wgs84_geometry, cur_items in zip(
            geometries, wgs84_geometries, candidate_items
        ):
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

    def _get_candidate_items_index(
        self, wgs84_geometries: list[STGeometry]
    ) -> list[list[LandsatItem]]:
        """List relevant items using rtree index."""
        candidates: list[list[LandsatItem]] = [[] for _ in wgs84_geometries]
        for idx, wgs84_geometry in enumerate(wgs84_geometries):
            encoded_items: set[str] = set()
            if self.rtree_index is None:
                raise ValueError("rtree_index is required")
            for shp in flatten_shape(wgs84_geometry.shp):
                encoded_items.update(self.rtree_index.query(shp.bounds))

            for encoded_item in encoded_items:
                item = LandsatItem.deserialize(json.loads(encoded_item))
                if not item.geometry.intersects_time_range(wgs84_geometry.time_range):
                    continue
                if not wgs84_geometry.shp.intersects(item.geometry.shp):
                    continue
                candidates[idx].append(item)

        return candidates

    def _get_candidate_items_bigquery(
        self, wgs84_geometries: list[STGeometry]
    ) -> list[list[LandsatItem]]:
        """List relevant items using one BigQuery query for the get_items call."""
        wrs2_index = self._get_wrs2_index()

        needed_pathrows = set()
        min_west = None
        min_south = None
        max_east = None
        max_north = None
        min_time = None
        max_time = None

        for wgs84_geometry in wgs84_geometries:
            needed_pathrows.update(
                get_pathrows_for_geometry(wrs2_index, wgs84_geometry)
            )
            west, south, east, north = wgs84_geometry.shp.bounds
            min_west = west if min_west is None else min(min_west, west)
            min_south = south if min_south is None else min(min_south, south)
            max_east = east if max_east is None else max(max_east, east)
            max_north = north if max_north is None else max(max_north, north)

            if wgs84_geometry.time_range is not None:
                start, end = wgs84_geometry.time_range
                min_time = start if min_time is None else min(min_time, start)
                max_time = end if max_time is None else max(max_time, end)

        if (
            min_west is None
            or min_south is None
            or max_east is None
            or max_north is None
            or min_time is None
            or max_time is None
            or len(needed_pathrows) == 0
        ):
            return [[] for _ in wgs84_geometries]

        all_items = []
        seen_names: set[str] = set()
        for item in self._read_bigquery(
            desc="Querying BigQuery",
            time_range=(min_time, max_time),
            wgs84_bbox=(min_west, min_south, max_east, max_north),
            pathrows=needed_pathrows,
        ):
            if item.name in seen_names:
                continue
            seen_names.add(item.name)
            all_items.append(item)

        candidates: list[list[LandsatItem]] = [[] for _ in wgs84_geometries]
        for idx, wgs84_geometry in enumerate(wgs84_geometries):
            for item in all_items:
                if not item.geometry.intersects_time_range(wgs84_geometry.time_range):
                    continue
                if not wgs84_geometry.shp.intersects(item.geometry.shp):
                    continue
                candidates[idx].append(item)

        return candidates

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
