"""Data source for FIRMS active fire detections."""

import csv
import io
import os
from datetime import UTC, date, datetime, timedelta
from typing import Any

import requests
import shapely

from rslearn.config import QueryConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource, DataSourceContext, Item
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import STGeometry

logger = get_logger(__name__)


class FIRMSItem(Item):
    """An item in the FIRMS data source."""

    def __init__(
        self,
        name: str,
        geometry: STGeometry,
        source: str,
        bbox: tuple[float, float, float, float],
    ):
        """Create a new FIRMSItem.

        Args:
            name: unique item name
            geometry: item geometry
            source: FIRMS source ID, e.g. VIIRS_SNPP_NRT
            bbox: area queried from FIRMS as (west, south, east, north) in WGS84
        """
        super().__init__(name, geometry)
        self.source = source
        self.bbox = bbox

    def serialize(self) -> dict:
        """Serialize the item to JSON-compatible dict."""
        d = super().serialize()
        d["source"] = self.source
        d["bbox"] = list(self.bbox)
        return d

    @staticmethod
    def deserialize(d: dict) -> "FIRMSItem":
        """Deserialize an item from JSON-compatible dict."""
        item = super(FIRMSItem, FIRMSItem).deserialize(d)
        bbox = d["bbox"]
        return FIRMSItem(
            name=item.name,
            geometry=item.geometry,
            source=d["source"],
            bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
        )


class FIRMS(DataSource[FIRMSItem]):
    """Data source for FIRMS active fire detections.

    This source queries FIRMS area CSV endpoint and writes point features (one feature
    per fire detection) to the tile store as vector data.

    Notes:
    - FIRMS API key (MAP_KEY) is required.
    - Each item corresponds to one window's spatial bbox and one window time range.
    - Requests are chunked by day range to satisfy FIRMS API limits.

    MAP_KEY limit is 5000 transactions / 10-minute interval.
    """

    BASE_URL = "https://firms.modaps.eosdis.nasa.gov"
    AREA_CSV_ENDPOINT = "/api/area/csv"
    # FIRMS area CSV endpoint currently enforces day_range in [1..5].
    API_MAX_DAYS_PER_REQUEST = 5

    def __init__(
        self,
        map_key: str | None = None,
        source: str = "VIIRS_SNPP_NRT",
        max_days_per_request: int = API_MAX_DAYS_PER_REQUEST,
        timeout: timedelta = timedelta(seconds=30),
        context: DataSourceContext = DataSourceContext(),
    ):
        """Initialize a new FIRMS instance.

        Args:
            map_key: FIRMS MAP_KEY. If not set, reads FIRMS_MAP_KEY env variable.
            source: FIRMS source ID, e.g. VIIRS_SNPP_NRT, VIIRS_NOAA20_NRT,
                MODIS_NRT.
            max_days_per_request: maximum day range used in each FIRMS request. Values
                above FIRMS API limits are clamped to 5.
            timeout: timeout for requests.
            context: the data source context.
        """
        del context  # unused but kept for standard data source signature

        if map_key is None:
            map_key = os.environ.get("FIRMS_MAP_KEY")
        if not map_key:
            raise ValueError(
                "FIRMS map_key must be provided (or set FIRMS_MAP_KEY env variable)"
            )
        self.map_key = map_key
        self.source = source
        self.timeout = timeout

        if max_days_per_request < 1:
            raise ValueError("max_days_per_request must be >= 1")
        if max_days_per_request > self.API_MAX_DAYS_PER_REQUEST:
            logger.warning(
                "max_days_per_request=%d exceeds FIRMS API limit (%d); clamping",
                max_days_per_request,
                self.API_MAX_DAYS_PER_REQUEST,
            )
            max_days_per_request = self.API_MAX_DAYS_PER_REQUEST
        self.max_days_per_request = max_days_per_request

    @staticmethod
    def _normalize_col_name(col_name: str) -> str:
        return col_name.strip().lower().replace(" ", "_")

    @staticmethod
    def _clip_bbox(
        bbox: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Clip bounds to valid WGS84 range."""
        west, south, east, north = bbox
        west = max(-180.0, min(180.0, west))
        east = max(-180.0, min(180.0, east))
        south = max(-90.0, min(90.0, south))
        north = max(-90.0, min(90.0, north))

        if east <= west or north <= south:
            raise ValueError(
                f"invalid bbox after clipping: {(west, south, east, north)}"
            )
        return (west, south, east, north)

    def _build_item(self, geometry: STGeometry) -> FIRMSItem:
        if geometry.time_range is None:
            raise ValueError("FIRMS requires geometry to have time_range")
        start_time, end_time = geometry.time_range
        if end_time <= start_time:
            raise ValueError(
                f"FIRMS geometry time range must satisfy end > start, got {geometry.time_range}"
            )

        wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
        bbox = self._clip_bbox(wgs84_geometry.shp.bounds)
        item_geometry = STGeometry(
            WGS84_PROJECTION, shapely.box(*bbox), (start_time, end_time)
        )
        item_name = (
            f"firms_{self.source}_{start_time.date().isoformat()}_{end_time.date().isoformat()}_"
            f"{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}"
        )
        return FIRMSItem(
            name=item_name,
            geometry=item_geometry,
            source=self.source,
            bbox=bbox,
        )

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[FIRMSItem]]]:
        """Get a FIRMS item for each input geometry.

        Each geometry maps to one FIRMS item whose spatial extent is the geometry bbox
        in WGS84 and whose temporal extent is the geometry time range.
        """
        del query_config  # query behavior is fixed: one item per geometry
        return [[[self._build_item(geometry)]] for geometry in geometries]

    def deserialize_item(self, serialized_item: Any) -> FIRMSItem:
        """Deserialize an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return FIRMSItem.deserialize(serialized_item)

    def _iter_date_chunks(
        self, start_time: datetime, end_time: datetime
    ) -> list[tuple[date, int]]:
        """Split [start_time, end_time) into day chunks for FIRMS requests."""
        start_date = start_time.astimezone(UTC).date()
        end_time_utc = end_time.astimezone(UTC)
        end_date_exclusive = end_time_utc.date()
        if end_time_utc.time() != datetime.min.time():
            end_date_exclusive += timedelta(days=1)
        if start_time >= end_time:
            raise ValueError(f"expected start < end, got {(start_time, end_time)}")

        chunks: list[tuple[date, int]] = []
        cur_date = start_date
        while cur_date < end_date_exclusive:
            next_date = min(
                end_date_exclusive,
                cur_date + timedelta(days=self.max_days_per_request),
            )
            chunks.append((cur_date, (next_date - cur_date).days))
            cur_date = next_date
        return chunks

    def _build_area_csv_url(
        self, bbox: tuple[float, float, float, float], start_date: date, day_range: int
    ) -> str:
        bbox_str = f"{bbox[0]:.6f},{bbox[1]:.6f},{bbox[2]:.6f},{bbox[3]:.6f}"
        return (
            f"{self.BASE_URL}{self.AREA_CSV_ENDPOINT}/{self.map_key}/{self.source}/"
            f"{bbox_str}/{day_range}/{start_date.isoformat()}"
        )

    @classmethod
    def _get_row_value(cls, row: dict[str, str], names: list[str]) -> str | None:
        wanted = {cls._normalize_col_name(name) for name in names}
        for key, value in row.items():
            if key is None:
                continue
            if cls._normalize_col_name(key) in wanted:
                return value
        return None

    @classmethod
    def _get_row_float(cls, row: dict[str, str], names: list[str]) -> float | None:
        value = cls._get_row_value(row, names)
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _parse_csv_features(
        self, csv_text: str, bbox: tuple[float, float, float, float]
    ) -> list[Feature]:
        reader = csv.DictReader(io.StringIO(csv_text))
        if not reader.fieldnames:
            return []

        has_lat_col = any(
            self._normalize_col_name(col) in {"latitude", "lat"}
            for col in reader.fieldnames
            if col is not None
        )
        has_lon_col = any(
            self._normalize_col_name(col) in {"longitude", "lon"}
            for col in reader.fieldnames
            if col is not None
        )

        # FIRMS returns a plain-text error payload for some failures (e.g., invalid key).
        if not has_lat_col or not has_lon_col:
            snippet = csv_text.strip().splitlines()[0] if csv_text.strip() else ""
            raise ValueError(
                "FIRMS response did not contain latitude/longitude columns. "
                f"Response starts with: {snippet!r}"
            )

        bbox_shp = shapely.box(*bbox)
        features: list[Feature] = []
        for row in reader:
            lat = self._get_row_float(row, ["latitude", "lat"])
            lon = self._get_row_float(row, ["longitude", "lon"])
            if lat is None or lon is None:
                continue

            point = shapely.Point(lon, lat)
            if not bbox_shp.intersects(point):
                continue

            properties = {k: v for k, v in row.items() if k is not None}
            properties["source"] = self.source
            features.append(Feature(STGeometry(WGS84_PROJECTION, point, None), properties))
        return features

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[FIRMSItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest FIRMS item data into the tile store as vector features.

        Args:
            tile_store: tile store for this layer
            items: FIRMS items to ingest
            geometries: geometries where each item is needed (unused)
        """
        del geometries  # item geometry already encodes request bounds and time range

        for item in items:
            if tile_store.is_vector_ready(item.name):
                continue
            if item.geometry.time_range is None:
                raise ValueError(f"item {item.name} is missing time_range")

            start_time, end_time = item.geometry.time_range
            features: list[Feature] = []
            for start_date, day_range in self._iter_date_chunks(start_time, end_time):
                url = self._build_area_csv_url(item.bbox, start_date, day_range)
                logger.debug("Requesting FIRMS CSV: %s", url)
                response = requests.get(url, timeout=self.timeout.total_seconds())
                response.raise_for_status()
                features.extend(self._parse_csv_features(response.text, item.bbox))

            logger.debug("Writing %d FIRMS features for item %s", len(features), item.name)
            tile_store.write_vector(item.name, features)
