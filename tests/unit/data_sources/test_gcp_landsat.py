"""Unit tests for the gcp_landsat data source."""

import csv
import gzip
import pathlib
from datetime import datetime
from io import StringIO
from typing import Any
from unittest.mock import MagicMock, patch

import shapely

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.gcp_landsat import (
    BUCKET_NAME,
    CollectionCategory,
    DataType,
    Landsat,
    LandsatItem,
    SpacecraftId,
)
from rslearn.utils.geometry import STGeometry

INDEX_COLUMNS = [
    "SCENE_ID",
    "PRODUCT_ID",
    "SPACECRAFT_ID",
    "SENSOR_ID",
    "DATE_ACQUIRED",
    "COLLECTION_NUMBER",
    "COLLECTION_CATEGORY",
    "SENSING_TIME",
    "DATA_TYPE",
    "WRS_PATH",
    "WRS_ROW",
    "CLOUD_COVER",
    "NORTH_LAT",
    "SOUTH_LAT",
    "WEST_LON",
    "EAST_LON",
    "TOTAL_SIZE",
    "BASE_URL",
]


def _make_index_row(
    product_id: str,
    west_lon: float,
    south_lat: float,
    east_lon: float,
    north_lat: float,
    spacecraft_id: str = "LANDSAT_8",
    date_acquired: str = "2025-01-31",
    sensing_time: str = "2025-01-31T18:22:13Z",
    data_type: str = "L1TP",
) -> dict:
    """Build a single row dict for a mock index CSV."""
    return {
        "SCENE_ID": "SCENE",
        "PRODUCT_ID": product_id,
        "SPACECRAFT_ID": spacecraft_id,
        "SENSOR_ID": "OLI_TIRS",
        "DATE_ACQUIRED": date_acquired,
        "COLLECTION_NUMBER": "02",
        "COLLECTION_CATEGORY": "T1",
        "SENSING_TIME": sensing_time,
        "DATA_TYPE": data_type,
        "WRS_PATH": "40",
        "WRS_ROW": "36",
        "CLOUD_COVER": "10.0",
        "NORTH_LAT": str(north_lat),
        "SOUTH_LAT": str(south_lat),
        "WEST_LON": str(west_lon),
        "EAST_LON": str(east_lon),
        "TOTAL_SIZE": "1000000",
        "BASE_URL": f"gs://{BUCKET_NAME}/LC08/L1/02/040/036/{product_id}",
    }


def _write_mock_index(tmp_path: pathlib.Path, rows: list[dict]) -> None:
    """Write a gzipped CSV index file from row dicts."""
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=INDEX_COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    gz_path = tmp_path / "index.csv.gz"
    with gzip.open(str(gz_path), "wt") as f:
        f.write(buf.getvalue())


def _make_item(
    name: str,
    spacecraft_id: str = "LANDSAT_8",
    data_type: str = "L1TP",
    cloud_cover: float = 10.0,
    lon: float = -122.0,
    lat: float = 47.0,
) -> LandsatItem:
    """Create a LandsatItem with reasonable defaults."""
    ts = datetime(2025, 1, 31)
    geometry = STGeometry(
        WGS84_PROJECTION,
        shapely.Point(lon, lat).buffer(1.0),
        (ts, ts),
    )
    blob_path = f"LC08/L1/02/040/036/{name}/"
    return LandsatItem(
        name=name,
        geometry=geometry,
        blob_path=blob_path,
        cloud_cover=cloud_cover,
        spacecraft_id=spacecraft_id,
        data_type=data_type,
    )


def _make_data_source(tmp_path: pathlib.Path, **kwargs: Any) -> Landsat:
    """Create a Landsat data source with mocked GCS client and rtree."""
    with (
        patch("rslearn.data_sources.gcp_landsat.storage") as mock_storage,
        patch("rslearn.data_sources.gcp_landsat.get_cached_rtree") as mock_rtree,
    ):
        mock_storage.Client.return_value = MagicMock()
        mock_storage.Client.return_value.bucket.return_value = MagicMock()
        mock_rtree.return_value = MagicMock()
        return Landsat(index_cache_dir=str(tmp_path), **kwargs)


class TestLandsatItemSerialize:
    """Tests for LandsatItem serialization roundtrip."""

    def test_roundtrip(self) -> None:
        item = _make_item("LC08_L1TP_040036_20250131_20250208_02_T1")
        d = item.serialize()
        restored = LandsatItem.deserialize(d)
        assert restored.name == item.name
        assert restored.blob_path == item.blob_path
        assert restored.cloud_cover == item.cloud_cover
        assert restored.spacecraft_id == item.spacecraft_id
        assert restored.data_type == item.data_type


class TestReadIndex:
    """Tests for _read_index filtering."""

    def test_yields_matching_items(self, tmp_path: pathlib.Path) -> None:
        ds = _make_data_source(tmp_path)
        _write_mock_index(
            tmp_path,
            [
                _make_index_row(
                    "LC08_L1TP_040036_20250131_20250208_02_T1",
                    west_lon=-118.0,
                    south_lat=33.0,
                    east_lon=-115.0,
                    north_lat=35.0,
                ),
            ],
        )

        items = list(ds._read_index())
        assert len(items) == 1
        assert items[0].name == "LC08_L1TP_040036_20250131_20250208_02_T1"

    def test_filters_by_spacecraft_id(self, tmp_path: pathlib.Path) -> None:
        ds = _make_data_source(tmp_path, spacecraft_id=[SpacecraftId.LANDSAT_9])
        _write_mock_index(
            tmp_path,
            [
                _make_index_row(
                    "LC08_L1TP_040036_20250131_20250208_02_T1",
                    west_lon=-118.0,
                    south_lat=33.0,
                    east_lon=-115.0,
                    north_lat=35.0,
                    spacecraft_id="LANDSAT_8",
                ),
                _make_index_row(
                    "LC09_L1TP_040036_20250215_20250215_02_T1",
                    west_lon=-118.0,
                    south_lat=33.0,
                    east_lon=-115.0,
                    north_lat=35.0,
                    spacecraft_id="LANDSAT_9",
                    date_acquired="2025-02-15",
                    sensing_time="2025-02-15T18:00:00Z",
                ),
            ],
        )

        items = list(ds._read_index())
        assert len(items) == 1
        assert items[0].spacecraft_id == "LANDSAT_9"

    def test_filters_by_data_type(self, tmp_path: pathlib.Path) -> None:
        ds = _make_data_source(tmp_path, data_type=[DataType.L1TP])
        _write_mock_index(
            tmp_path,
            [
                _make_index_row(
                    "LC08_L1GT_040036_20250131_20250208_02_T2",
                    west_lon=-118.0,
                    south_lat=33.0,
                    east_lon=-115.0,
                    north_lat=35.0,
                    data_type="L1GT",
                ),
                _make_index_row(
                    "LC08_L1TP_040036_20250215_20250215_02_T1",
                    west_lon=-118.0,
                    south_lat=33.0,
                    east_lon=-115.0,
                    north_lat=35.0,
                    date_acquired="2025-02-15",
                    sensing_time="2025-02-15T18:00:00Z",
                ),
            ],
        )

        items = list(ds._read_index())
        assert len(items) == 1
        assert items[0].data_type == "L1TP"

    def test_filters_by_collection_category(self, tmp_path: pathlib.Path) -> None:
        ds = _make_data_source(tmp_path, collection_category=[CollectionCategory.T1])
        _write_mock_index(
            tmp_path,
            [
                _make_index_row(
                    "LC08_L1TP_040036_20250131_20250208_02_T2",
                    west_lon=-118.0,
                    south_lat=33.0,
                    east_lon=-115.0,
                    north_lat=35.0,
                ),
                _make_index_row(
                    "LC08_L1TP_040036_20250215_20250215_02_T1",
                    west_lon=-118.0,
                    south_lat=33.0,
                    east_lon=-115.0,
                    north_lat=35.0,
                    date_acquired="2025-02-15",
                    sensing_time="2025-02-15T18:00:00Z",
                ),
            ],
        )

        items = list(ds._read_index())
        assert len(items) == 1
        assert items[0].name.endswith("_T1")

    def test_filters_by_rtree_time_range(self, tmp_path: pathlib.Path) -> None:
        ds = _make_data_source(
            tmp_path,
            rtree_time_range=(datetime(2025, 2, 1), datetime(2025, 12, 31)),
        )
        _write_mock_index(
            tmp_path,
            [
                _make_index_row(
                    "LC08_L1TP_040036_20250131_20250208_02_T1",
                    west_lon=-118.0,
                    south_lat=33.0,
                    east_lon=-115.0,
                    north_lat=35.0,
                    date_acquired="2025-01-31",
                    sensing_time="2025-01-31T18:00:00Z",
                ),
                _make_index_row(
                    "LC08_L1TP_040036_20250215_20250215_02_T1",
                    west_lon=-118.0,
                    south_lat=33.0,
                    east_lon=-115.0,
                    north_lat=35.0,
                    date_acquired="2025-02-15",
                    sensing_time="2025-02-15T18:00:00Z",
                ),
            ],
        )

        items = list(ds._read_index())
        assert len(items) == 1
        assert items[0].name == "LC08_L1TP_040036_20250215_20250215_02_T1"

    def test_antimeridian_crossing(self, tmp_path: pathlib.Path) -> None:
        """A scene crossing the antimeridian produces a valid multi-polygon."""
        ds = _make_data_source(tmp_path)
        _write_mock_index(
            tmp_path,
            [
                _make_index_row(
                    "LE07_L1TP_091014_20120313_20200909_02_T2",
                    west_lon=176.5,
                    south_lat=50.0,
                    east_lon=-177.7,
                    north_lat=52.0,
                    date_acquired="2012-03-13",
                    sensing_time="2012-03-13T00:00:00Z",
                ),
            ],
        )

        items = list(ds._read_index())
        assert len(items) == 1
        shp = items[0].geometry.shp
        assert shp.geom_type == "MultiPolygon"


class TestGetAssetUrl:
    """Tests for get_asset_url."""

    def test_returns_gs_url(self, tmp_path: pathlib.Path) -> None:
        """get_asset_url returns a gs:// URL."""
        ds = _make_data_source(tmp_path)
        item = _make_item("LC08_L1TP_040036_20250131_20250208_02_T1")

        url = ds.get_asset_url(item, "B4")

        expected = (
            f"gs://{BUCKET_NAME}/"
            "LC08/L1/02/040/036/LC08_L1TP_040036_20250131_20250208_02_T1/"
            "LC08_L1TP_040036_20250131_20250208_02_T1_B4.TIF"
        )
        assert url == expected
