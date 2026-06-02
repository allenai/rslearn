"""Unit tests for the gcp_landsat data source."""

import pathlib
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import shapely

from rslearn.config import QueryConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.gcp_landsat import (
    BUCKET_NAME,
    Landsat,
    LandsatItem,
    SpacecraftId,
)
from rslearn.data_sources.wrs2 import WRS2_GRID_SIZE
from rslearn.utils.geometry import STGeometry
from rslearn.utils.grid_index import GridIndex


@pytest.fixture(autouse=True)
def _gs_user_project(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set GS_USER_PROJECT, which Landsat requires for requester-pays billing."""
    monkeypatch.setenv("GS_USER_PROJECT", "test-project")


def _make_bigquery_row(
    product_id: str,
    west_lon: float,
    south_lat: float,
    east_lon: float,
    north_lat: float,
    spacecraft_id: str = "LANDSAT_8",
    sensor_id: str | None = "OLI_TIRS",
    sensing_time: str = "2025-01-31T18:22:13Z",
    processing_level: str = "L1TP",
    collection_category: str = "T1",
    wrs_path: int = 40,
    wrs_row: int = 36,
) -> dict[str, Any]:
    """Build a single row dict for a mock BigQuery result."""
    return {
        "product_id": product_id,
        "spacecraft_id": spacecraft_id,
        "sensor_id": sensor_id,
        "sensing_time": datetime.fromisoformat(sensing_time),
        "processing_level": processing_level,
        "collection_category": collection_category,
        "wrs_path": wrs_path,
        "wrs_row": wrs_row,
        "cloud_cover": 10.0,
        "north_lat": north_lat,
        "south_lat": south_lat,
        "west_lon": west_lon,
        "east_lon": east_lon,
        "base_url": f"gs://{BUCKET_NAME}/LC08/L1/02/040/036/{product_id}",
    }


def _make_item(
    name: str,
    spacecraft_id: str = "LANDSAT_8",
    sensor_id: str | None = "OLI_TIRS",
    processing_level: str = "L1TP",
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
        sensor_id=sensor_id,
        processing_level=processing_level,
    )


def _make_data_source(tmp_path: pathlib.Path, **kwargs: Any) -> Landsat:
    """Create a Landsat data source with the rtree build mocked out."""
    kwargs.setdefault("bands", ["B4"])
    with patch("rslearn.data_sources.gcp_landsat.get_cached_rtree") as mock_rtree:
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
        assert restored.sensor_id == item.sensor_id
        assert restored.processing_level == item.processing_level


class TestReadBigQuery:
    """Tests for _read_bigquery filtering."""

    def test_yields_matching_items(self, tmp_path: pathlib.Path) -> None:
        ds = _make_data_source(tmp_path)
        rows = [
            _make_bigquery_row(
                "LC08_L1TP_040036_20250131_20250208_02_T1",
                west_lon=-118.0,
                south_lat=33.0,
                east_lon=-115.0,
                north_lat=35.0,
            )
        ]

        with patch(
            "rslearn.data_sources.gcp_landsat.bigquery.Client"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client.query.return_value = rows
            mock_client_cls.return_value = mock_client

            items = list(ds._read_bigquery())

        assert len(items) == 1
        assert items[0].name == "LC08_L1TP_040036_20250131_20250208_02_T1"

    def test_filters_by_spacecraft_id_in_sql(self, tmp_path: pathlib.Path) -> None:
        ds = _make_data_source(tmp_path, spacecraft_id=[SpacecraftId.LANDSAT_9])

        with patch(
            "rslearn.data_sources.gcp_landsat.bigquery.Client"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client.query.return_value = []
            mock_client_cls.return_value = mock_client

            list(ds._read_bigquery())
            query_str = mock_client.query.call_args[0][0]
            job_config = mock_client.query.call_args[1]["job_config"]

        assert "spacecraft_id IN UNNEST(@spacecraft_ids)" in query_str
        spacecraft_params = [
            p for p in job_config.query_parameters if p.name == "spacecraft_ids"
        ]
        assert len(spacecraft_params) == 1
        assert spacecraft_params[0].values == ["LANDSAT_9"]

    def test_antimeridian_crossing(self, tmp_path: pathlib.Path) -> None:
        """A scene crossing the antimeridian produces a valid multi-polygon."""
        ds = _make_data_source(tmp_path)
        rows = [
            _make_bigquery_row(
                "LE07_L1TP_091014_20120313_20200909_02_T2",
                west_lon=176.5,
                south_lat=50.0,
                east_lon=-177.7,
                north_lat=52.0,
                sensing_time="2012-03-13T00:00:00Z",
            )
        ]

        with patch(
            "rslearn.data_sources.gcp_landsat.bigquery.Client"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client.query.return_value = rows
            mock_client_cls.return_value = mock_client

            items = list(ds._read_bigquery())

        assert len(items) == 1
        shp = items[0].geometry.shp
        assert shp.geom_type == "MultiPolygon"
        # The scene spans 176.5E to 177.7W, so splitting at the antimeridian must
        # yield one part hugging +180 and one hugging -180 (not a globe-spanning box).
        part_bounds = sorted(part.bounds for part in shp.geoms)
        west_part, east_part = part_bounds
        assert west_part[0] == pytest.approx(-180)
        assert west_part[2] == pytest.approx(-177.7)
        assert east_part[0] == pytest.approx(176.5)
        assert east_part[2] == pytest.approx(180)


class TestGetItemsBigQueryMode:
    """Tests for get_items in direct BigQuery mode.

    These tests mock the BigQuery client and the WRS2 grid build (which would
    otherwise download a shapefile), but test the path/row lookup, scene parsing,
    spatial/temporal intersection, and window matching.
    """

    def test_matches_item_with_real_pathrow_lookup(
        self, tmp_path: pathlib.Path
    ) -> None:
        ts = datetime(2025, 1, 31, tzinfo=UTC)
        geometry = STGeometry(
            WGS84_PROJECTION,
            shapely.box(-118.0, 33.0, -115.0, 35.0),
            (ts, ts + timedelta(days=1)),
        )

        # Build a GridIndex covering the geometry, tagged with WRS path/row 40/36.
        wrs2_index = GridIndex(WRS2_GRID_SIZE)
        wrs2_polygon = shapely.box(-119.0, 32.0, -114.0, 36.0)
        wrs2_index.insert(wrs2_polygon.bounds, (wrs2_polygon, "40", "36"))

        rows = [
            _make_bigquery_row(
                "LC08_L1TP_040036_20250131_20250208_02_T1",
                west_lon=-118.0,
                south_lat=33.0,
                east_lon=-115.0,
                north_lat=35.0,
                wrs_path=40,
                wrs_row=36,
            )
        ]

        with (
            patch(
                "rslearn.data_sources.gcp_landsat.bigquery.Client"
            ) as mock_client_cls,
            patch(
                "rslearn.data_sources.gcp_landsat.build_wrs2_grid_index",
                return_value=wrs2_index,
            ),
        ):
            mock_client = MagicMock()
            mock_client.query.return_value = rows
            mock_client_cls.return_value = mock_client

            ds = Landsat(
                index_cache_dir=str(tmp_path),
                bands=["B4"],
                use_rtree_index=False,
            )
            result = ds.get_items([geometry], QueryConfig())

        # One query for the whole get_items call.
        assert mock_client.query.call_count == 1

        # The real path/row lookup must have produced the 40/36 filter, passed as a
        # "path,row" entry in the @pathrows query parameter.
        query_str = mock_client.query.call_args[0][0]
        job_config = mock_client.query.call_args[1]["job_config"]
        assert "IN UNNEST(@pathrows)" in query_str
        pathrow_params = [
            p for p in job_config.query_parameters if p.name == "pathrows"
        ]
        assert len(pathrow_params) == 1
        assert "40,36" in pathrow_params[0].values

        # The candidate item should be matched into a single group for the geometry.
        assert len(result) == 1
        group = result[0]
        assert len(group) == 1
        matched_names = [item.name for item in group[0].items]
        assert matched_names == ["LC08_L1TP_040036_20250131_20250208_02_T1"]


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


class TestBandToFileToken:
    """Tests for Level-1 vs Level-2 band file token naming.

    Level-1 band files are named simply by the band (e.g. "B4"). Level-2 files
    are prefixed with the asset type: surface reflectance ("SR_") for all bands
    except the sensor's thermal band, which is surface temperature ("ST_").
    OLI-TIRS thermal is B10; TM and ETM+ thermal is B6.
    """

    def test_level1_returns_band_as_is(self, tmp_path: pathlib.Path) -> None:
        ds = _make_data_source(tmp_path)
        item = _make_item(
            "LC08_L1TP_040036_20250131_20250208_02_T1",
            sensor_id="OLI_TIRS",
            processing_level="L1TP",
        )
        assert ds._band_to_file_token(item, "B4") == "B4"
        assert ds._band_to_file_token(item, "B10") == "B10"

    @pytest.mark.parametrize(
        ("sensor_id", "thermal_band"),
        [("OLI_TIRS", "B10"), ("TM", "B6"), ("ETM", "B6")],
    )
    def test_level2_thermal_and_reflectance(
        self, tmp_path: pathlib.Path, sensor_id: str, thermal_band: str
    ) -> None:
        ds = _make_data_source(tmp_path)
        item = _make_item(
            "X_L2SP_040036_20250131_20250208_02_T1",
            sensor_id=sensor_id,
            processing_level="L2SP",
        )
        assert ds._band_to_file_token(item, thermal_band) == f"ST_{thermal_band}"
        # A non-thermal band is surface reflectance.
        assert ds._band_to_file_token(item, "B4") == "SR_B4"
