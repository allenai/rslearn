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

from .data_source import (
    DataSourceContext,
    Item,
    QueryConfig,
    RetrieveItemDataSource,
)
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


class ProcessingLevel(StrEnum):
    """Landsat processing levels."""

    L1GS = "L1GS"
    L1GT = "L1GT"
    L1TP = "L1TP"
    L2SP = "L2SP"
    L2SR = "L2SR"


class SensorId(StrEnum):
    """Landsat sensor identifiers."""

    MSS = "MSS"
    TM = "TM"
    ETM = "ETM"
    OLI = "OLI"
    TIRS = "TIRS"
    OLI_TIRS = "OLI_TIRS"


# Level-2 product folders use SR_ surface reflectance assets and, for L2SP
# products, ST_ surface temperature assets. L2SR products omit ST_* files. This
# map identifies the sensor-specific thermal band used for L2SP ST_* filenames.
_LEVEL2_THERMAL_BAND: dict[SensorId, str] = {
    SensorId.OLI_TIRS: "B10",
    SensorId.TM: "B6",
    SensorId.ETM: "B6",
}

_LEVEL1_PROCESSING_LEVELS = {
    ProcessingLevel.L1GS,
    ProcessingLevel.L1GT,
    ProcessingLevel.L1TP,
}

_L1_BANDS: dict[SensorId, list[str]] = {
    SensorId.OLI_TIRS: [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B9",
        "B10",
        "B11",
    ],
    SensorId.OLI: ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"],
    SensorId.TIRS: ["B10", "B11"],
    SensorId.ETM: [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6_VCID_1",
        "B6_VCID_2",
        "B7",
        "B8",
    ],
    SensorId.TM: ["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
    SensorId.MSS: ["B4", "B5", "B6", "B7"],
}

AVAILABLE_BANDS: dict[ProcessingLevel, dict[SensorId, list[str]]] = {
    ProcessingLevel.L1GS: _L1_BANDS,
    ProcessingLevel.L1GT: _L1_BANDS,
    ProcessingLevel.L1TP: _L1_BANDS,
    ProcessingLevel.L2SP: {
        SensorId.OLI_TIRS: ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B10"],
        SensorId.ETM: ["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
        SensorId.TM: ["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
    },
    ProcessingLevel.L2SR: {
        SensorId.OLI_TIRS: ["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
        SensorId.ETM: ["B1", "B2", "B3", "B4", "B5", "B7"],
        SensorId.TM: ["B1", "B2", "B3", "B4", "B5", "B7"],
    },
}


class LandsatItem(Item):
    """An item in the Landsat data source."""

    def __init__(
        self,
        name: str,
        geometry: STGeometry,
        blob_path: str,
        cloud_cover: float,
        spacecraft_id: SpacecraftId,
        processing_level: ProcessingLevel,
        sensor_id: SensorId,
    ) -> None:
        """Creates a new LandsatItem.

        Args:
            name: unique name of the item (PRODUCT_ID).
            geometry: the spatial and temporal extent of the item.
            blob_path: path within the GCS bucket to the scene folder, e.g.
                "LC09/L1/02/231/062/LC09_L1TP_231062_20260426_20260426_02_T1/".
            cloud_cover: the scene's cloud cover percentage (0-100).
            spacecraft_id: the spacecraft identifier, e.g. SpacecraftId.LANDSAT_8.
            sensor_id: the sensor identifier, e.g. SensorId.OLI_TIRS.
            processing_level: the processing level, e.g. ProcessingLevel.L1TP.
        """
        super().__init__(name, geometry)
        self.blob_path = blob_path
        self.cloud_cover = cloud_cover
        self.spacecraft_id = spacecraft_id
        self.sensor_id = sensor_id
        self.processing_level = processing_level

    @override
    def serialize(self) -> dict[str, Any]:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["blob_path"] = self.blob_path
        d["cloud_cover"] = self.cloud_cover
        d["spacecraft_id"] = self.spacecraft_id.value
        d["sensor_id"] = self.sensor_id.value
        d["processing_level"] = self.processing_level.value
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
            spacecraft_id=SpacecraftId(d["spacecraft_id"]),
            processing_level=ProcessingLevel(d["processing_level"]),
            sensor_id=SensorId(d["sensor_id"]),
        )


class Landsat(
    DirectMaterializeDataSource[LandsatItem],
    RetrieveItemDataSource[LandsatItem],
):
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
        sensor_ids: list[SensorId],
        processing_levels: list[ProcessingLevel],
        spacecraft_ids: list[SpacecraftId] | None = None,
        bands: list[str] | None = None,
        sort_by: str | None = None,
        collection_category: list[CollectionCategory] | None = None,
        use_rtree_index: bool = True,
        rtree_time_range: tuple[datetime, datetime] | None = None,
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Initialize a new Landsat instance.

        Args:
            index_cache_dir: directory to cache the index and rtree files.
            sensor_ids: filter by sensor, e.g. [SensorId.OLI_TIRS]. Multiple sensors
                are only supported for Level-1 processing levels.
            processing_levels: filter by processing level, e.g.
                [ProcessingLevel.L1TP]. Multiple processing levels are only
                supported when all configured processing levels are Level-1.
            spacecraft_ids: optional filter by mission, e.g.
                [SpacecraftId.LANDSAT_8, SpacecraftId.LANDSAT_9]. None means
                all missions with the configured sensors.
            bands: which bands to expose. Defaults to all bands available for the
                configured sensor(s) and processing level(s) if the layer config
                does not specify band sets. Requested/default bands must be
                available for every configured sensor and processing level.
            sort_by: "cloud_cover" or None (arbitrary order).
            collection_category: filter by tier, e.g. ["T1"]. None means all.
            use_rtree_index: whether to build and query local rtree index.
            rtree_time_range: only index scenes within this time range.
                Restricting to a shorter period significantly speeds up rtree
                creation.
            context: the data source context.
        """
        if len(sensor_ids) == 0:
            raise ValueError("sensor_ids must contain at least one sensor")
        if len(processing_levels) == 0:
            raise ValueError("processing_levels must contain at least one level")
        self.sensor_ids = sensor_ids
        self.processing_levels = processing_levels
        self.spacecraft_ids = spacecraft_ids
        available_bands = self._get_available_bands(
            self.sensor_ids, self.processing_levels
        )

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
            effective_bands = list(available_bands)

        missing_bands = set(effective_bands) - set(available_bands)
        if missing_bands:
            raise ValueError(
                f"bands {sorted(missing_bands)} are not available for "
                f"sensor_ids={self.sensor_ids} and "
                f"processing_levels={self.processing_levels}; "
                f"available bands are {available_bands}"
            )

        asset_bands = {band: [band] for band in effective_bands}
        super().__init__(asset_bands=asset_bands)

        if context.ds_path is not None:
            self.index_cache_dir = join_upath(context.ds_path, index_cache_dir)
        else:
            self.index_cache_dir = UPath(index_cache_dir)

        self.collection_category_filter = (
            set(collection_category) if collection_category is not None else None
        )
        self.use_rtree_index = use_rtree_index
        self.rtree_time_range = rtree_time_range
        self.sort_by = sort_by
        self.effective_bands = effective_bands

        # The bucket is requester-pays, so we need a project to bill. GDAL uses the
        # GS_USER_PROJECT environment variable for the same purpose when reading the
        # rasters directly via gs:// URLs, so we reuse it here for consistency and to
        # avoid requiring separate configuration.
        user_project = os.environ.get("GS_USER_PROJECT")
        if not user_project:
            raise ValueError(
                "the GS_USER_PROJECT environment variable must be set to a GCP "
                "project for requester-pays billing on the Landsat bucket"
            )
        self.user_project = user_project

        self.index_cache_dir.mkdir(parents=True, exist_ok=True)

        self._bucket: storage.Bucket | None = None
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

    def _get_available_bands(
        self, sensor_ids: list[SensorId], processing_levels: list[ProcessingLevel]
    ) -> list[str]:
        """Get bands available for every configured sensor and processing level."""
        all_level1 = all(
            processing_level in _LEVEL1_PROCESSING_LEVELS
            for processing_level in processing_levels
        )
        if len(processing_levels) > 1 and not all_level1:
            raise ValueError(
                "multiple processing_levels values are only supported when all "
                f"configured processing levels are Level-1; got {processing_levels}"
            )
        if len(sensor_ids) > 1 and not all_level1:
            raise ValueError(
                "multiple sensor_ids values are only supported for Level-1 "
                f"processing levels; got processing_levels={processing_levels}"
            )

        common_bands: set[str] | None = None
        for processing_level in processing_levels:
            sensor_bands = AVAILABLE_BANDS[processing_level]
            valid_sensors = sorted(sensor.value for sensor in sensor_bands)
            for sensor_id in sensor_ids:
                if sensor_id not in sensor_bands:
                    raise ValueError(
                        f"sensor_id={sensor_id} is not available for "
                        f"processing_level={processing_level}; valid sensors are "
                        f"{valid_sensors}"
                    )
                if common_bands is None:
                    common_bands = set(sensor_bands[sensor_id])
                else:
                    common_bands.intersection_update(sensor_bands[sensor_id])

        assert common_bands is not None
        return list(common_bands)

    def _get_bigquery_client(self) -> bigquery.Client:
        """Lazily initialize BigQuery client."""
        if self._bigquery_client is None:
            self._bigquery_client = bigquery.Client()
        return self._bigquery_client

    def _get_bucket(self) -> storage.Bucket:
        """Lazily initialize the requester-pays GCS bucket."""
        if self._bucket is None:
            self._bucket = storage.Client().bucket(
                BUCKET_NAME, user_project=self.user_project
            )
        return self._bucket

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
                    collection_category, wrs_path, wrs_row,
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
        query_params: list[
            bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter
        ] = []
        if time_range is not None:
            query_str += (
                " AND sensing_time >= @time_start AND sensing_time <= @time_end"
            )
            query_params.append(
                bigquery.ScalarQueryParameter("time_start", "TIMESTAMP", time_range[0])
            )
            query_params.append(
                bigquery.ScalarQueryParameter("time_end", "TIMESTAMP", time_range[1])
            )
        if wgs84_bbox is not None:
            query_str += (
                " AND west_lon < @bbox_east"
                " AND east_lon > @bbox_west"
                " AND south_lat < @bbox_north"
                " AND north_lat > @bbox_south"
            )
            query_params.append(
                bigquery.ScalarQueryParameter("bbox_west", "FLOAT64", wgs84_bbox[0])
            )
            query_params.append(
                bigquery.ScalarQueryParameter("bbox_south", "FLOAT64", wgs84_bbox[1])
            )
            query_params.append(
                bigquery.ScalarQueryParameter("bbox_east", "FLOAT64", wgs84_bbox[2])
            )
            query_params.append(
                bigquery.ScalarQueryParameter("bbox_north", "FLOAT64", wgs84_bbox[3])
            )
        if pathrows:
            # Match against "path,row" strings to filter on both columns at once.
            query_str += (
                " AND CONCAT(CAST(wrs_path AS STRING), ',',"
                " CAST(wrs_row AS STRING)) IN UNNEST(@pathrows)"
            )
            pathrow_strs = [f"{int(path)},{int(row)}" for path, row in sorted(pathrows)]
            query_params.append(
                bigquery.ArrayQueryParameter("pathrows", "STRING", pathrow_strs)
            )

        def _add_in_filter(
            param_name: str, column_name: str, values: Collection[str] | None
        ) -> None:
            nonlocal query_str
            if values is None:
                return
            query_str += f" AND {column_name} IN UNNEST(@{param_name})"
            query_params.append(
                bigquery.ArrayQueryParameter(param_name, "STRING", sorted(values))
            )

        _add_in_filter(
            "sensor_ids",
            "sensor_id",
            [sensor_id.value for sensor_id in self.sensor_ids],
        )
        _add_in_filter(
            "spacecraft_ids",
            "spacecraft_id",
            (
                [spacecraft_id.value for spacecraft_id in self.spacecraft_ids]
                if self.spacecraft_ids is not None
                else None
            ),
        )
        _add_in_filter(
            "collection_categories",
            "collection_category",
            self.collection_category_filter,
        )

        result = self._get_bigquery_client().query(
            query_str,
            job_config=bigquery.QueryJobConfig(query_parameters=query_params),
        )
        if desc is not None:
            result = tqdm.tqdm(result, desc=desc)

        configured_processing_levels = {
            processing_level.value for processing_level in self.processing_levels
        }
        for row in result:
            base_url = row["base_url"]

            # base_url is always a gs://BUCKET_NAME/... path without a trailing
            # slash, so convert it to a bucket-relative blob path ending in "/".
            gs_prefix = f"gs://{BUCKET_NAME}/"
            if not base_url.startswith(gs_prefix):
                raise ValueError(f"unexpected base_url {base_url}")
            blob_path = base_url[len(gs_prefix) :] + "/"

            # Use the product folder name as the canonical product ID. The
            # product_id column can disagree with base_url for a small number
            # of rows (e.g. a Level-1 product_id paired with a Level-2
            # base_url). The band files on GCS live in the base_url folder and
            # are named after it, so the folder name is authoritative for both
            # the item name and the derived processing level.
            product_id = blob_path.rstrip("/").split("/")[-1]

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

            # Derive processing level from the product_id e.g. "L1TP" in
            # "LC09_L1TP_231062_20260426_20260426_02_T1").
            product_id_parts = product_id.split("_")
            processing_level = product_id_parts[1] if len(product_id_parts) > 1 else ""

            if processing_level not in configured_processing_levels:
                continue

            yield LandsatItem(
                name=product_id,
                geometry=geometry,
                blob_path=blob_path,
                cloud_cover=float(row["cloud_cover"]),
                spacecraft_id=SpacecraftId(row["spacecraft_id"]),
                processing_level=ProcessingLevel(processing_level),
                sensor_id=SensorId(row["sensor_id"]),
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
        if self.rtree_index is None:
            raise ValueError("rtree_index is required")

        candidates: list[list[LandsatItem]] = [[] for _ in wgs84_geometries]
        for idx, wgs84_geometry in enumerate(wgs84_geometries):
            encoded_items: set[str] = set()
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

            # get_items already verifies that every geometry has a time range.
            assert wgs84_geometry.time_range is not None
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

    def _band_to_file_token(self, item: LandsatItem, band: str) -> str:
        """Get the band token used in the GCS blob filename for the given band.

        Level-1 products name band files simply by the band (e.g. "B4"), while
        Level-2 products prefix them with the asset type: surface temperature
        ("ST_B10" for OLI-TIRS, "ST_B6" for TM/ETM+) for the sensor's thermal
        band and surface reflectance ("SR_B4") for all other bands.

        Args:
            item: the item.
            band: the band name (e.g. "B4").

        Returns:
            the band token used in the blob filename (e.g. "B4" or "SR_B4").
        """
        if item.processing_level in _LEVEL1_PROCESSING_LEVELS:
            return band
        thermal_band = _LEVEL2_THERMAL_BAND.get(item.sensor_id)
        if band == thermal_band:
            return f"ST_{band}"
        return f"SR_{band}"

    def _band_blob_key(self, item: LandsatItem, band: str) -> str:
        """Get the blob key (path within the bucket) for an item's band file.

        Args:
            item: the item.
            band: the band name (e.g. "B4").

        Returns:
            the blob key, e.g. "<blob_path><name>_SR_B4.TIF".
        """
        return f"{item.blob_path}{item.name}_{self._band_to_file_token(item, band)}.TIF"

    @override
    def get_asset_url(self, item: LandsatItem, asset_key: str) -> str:
        """Get a gs:// URL for the band TIF.

        Args:
            item: the item.
            asset_key: the band name (e.g. "B4").

        Returns:
            a gs:// URL readable by rasterio.
        """
        blob_key = self._band_blob_key(item, asset_key)
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

                blob_key = self._band_blob_key(item, band)
                with tempfile.TemporaryDirectory() as tmp_dir:
                    fname = os.path.join(tmp_dir, f"{band}.tif")
                    blob = self._get_bucket().blob(blob_key)
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

    @override
    def retrieve_item(
        self, item: LandsatItem
    ) -> Generator[tuple[str, BinaryIO], None, None]:
        """Retrieves the rasters corresponding to an item as file streams."""
        for band in self.effective_bands:
            blob_key = self._band_blob_key(item, band)
            blob = self._get_bucket().blob(blob_key)
            buf = io.BytesIO()
            blob.download_to_file(buf)
            buf.seek(0)
            yield (f"{item.name}_{band}.TIF", buf)
