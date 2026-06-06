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
    ProcessingLevel,
    SensorId,
    SpacecraftId,
)
from rslearn.data_sources.wrs2 import WRS2_GRID_SIZE
from rslearn.utils.geometry import STGeometry
from rslearn.utils.grid_index import GridIndex


@pytest.fixture(autouse=True)
def _gs_user_project(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set GS_USER_PROJECT, which Landsat requires for requester-pays billing."""
    monkeypatch.setenv("GS_USER_PROJECT", "test-project")


LC09_L1_PRODUCT_ID = "LC09_L1TP_129040_20260603_20260603_02_T1"
LC09_L1_BASE_URL = f"gs://{BUCKET_NAME}/LC09/L1/02/129/040/{LC09_L1_PRODUCT_ID}"
LC09_L2SP_PRODUCT_ID = "LC09_L2SP_001093_20240324_20240325_02_T2"
LC09_L2SP_BASE_URL = f"gs://{BUCKET_NAME}/LC09/L2/02/001/093/{LC09_L2SP_PRODUCT_ID}"
LC09_L2SR_PRODUCT_ID = "LC09_L2SR_145065_20260603_20260605_02_T1"
LC09_L2SR_BASE_URL = f"gs://{BUCKET_NAME}/LC09/L2/02/145/065/{LC09_L2SR_PRODUCT_ID}"
LE07_L1_PRODUCT_ID = "LE07_L1TP_001091_20160724_20200902_02_T1"
LE07_L1_BASE_URL = f"gs://{BUCKET_NAME}/LE07/L1/02/001/091/{LE07_L1_PRODUCT_ID}"
LC08_L1GT_PRODUCT_ID = "LC08_L1GT_155005_20260601_20260603_02_RT"
LC08_L1GT_BASE_URL = f"gs://{BUCKET_NAME}/LC08/L1/02/155/005/{LC08_L1GT_PRODUCT_ID}"


def _make_bigquery_row(
    west_lon: float,
    south_lat: float,
    east_lon: float,
    north_lat: float,
    product_id: str = LC09_L1_PRODUCT_ID,
    spacecraft_id: str = "LANDSAT_9",
    sensor_id: str | None = "OLI_TIRS",
    sensing_time: str = "2026-06-03T02:45:00Z",
    collection_category: str = "T1",
    wrs_path: int = 129,
    wrs_row: int = 40,
    base_url: str = LC09_L1_BASE_URL,
) -> dict[str, Any]:
    """Build a single row dict for a mock BigQuery result."""
    return {
        "product_id": product_id,
        "spacecraft_id": spacecraft_id,
        "sensor_id": sensor_id,
        "sensing_time": datetime.fromisoformat(sensing_time),
        "collection_category": collection_category,
        "wrs_path": wrs_path,
        "wrs_row": wrs_row,
        "cloud_cover": 10.0,
        "north_lat": north_lat,
        "south_lat": south_lat,
        "west_lon": west_lon,
        "east_lon": east_lon,
        "base_url": base_url,
    }


def _make_item(
    name: str = LC09_L1_PRODUCT_ID,
    spacecraft_id: SpacecraftId = SpacecraftId.LANDSAT_9,
    sensor_id: SensorId = SensorId.OLI_TIRS,
    processing_level: ProcessingLevel = ProcessingLevel.L1TP,
    cloud_cover: float = 10.0,
    lon: float = -122.0,
    lat: float = 47.0,
) -> LandsatItem:
    """Create a LandsatItem with reasonable defaults."""
    ts = datetime(2026, 6, 3)
    geometry = STGeometry(
        WGS84_PROJECTION,
        shapely.Point(lon, lat).buffer(1.0),
        (ts, ts),
    )
    blob_path = f"LC09/L1/02/129/040/{name}/"
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
    kwargs.setdefault("sensor_ids", [SensorId.OLI_TIRS])
    kwargs.setdefault("processing_levels", [ProcessingLevel.L1TP])
    kwargs.setdefault("bands", ["B4"])
    with patch("rslearn.data_sources.gcp_landsat.get_cached_rtree") as mock_rtree:
        mock_rtree.return_value = MagicMock()
        return Landsat(index_cache_dir=str(tmp_path), **kwargs)


class TestLandsatItemSerialize:
    """Tests for LandsatItem serialization roundtrip."""

    def test_roundtrip(self) -> None:
        item = _make_item()
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
        assert items[0].name == LC09_L1_PRODUCT_ID

    def test_filters_by_spacecraft_ids_in_sql(self, tmp_path: pathlib.Path) -> None:
        ds = _make_data_source(tmp_path, spacecraft_ids=[SpacecraftId.LANDSAT_9])

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
        assert "sensor_id IN UNNEST(@sensor_ids)" in query_str
        assert "processing_level" not in query_str
        spacecraft_params = [
            p for p in job_config.query_parameters if p.name == "spacecraft_ids"
        ]
        assert len(spacecraft_params) == 1
        assert spacecraft_params[0].values == ["LANDSAT_9"]
        sensor_params = [
            p for p in job_config.query_parameters if p.name == "sensor_ids"
        ]
        assert len(sensor_params) == 1
        assert sensor_params[0].values == ["OLI_TIRS"]

    def test_uses_base_url_product_id_as_authoritative(
        self, tmp_path: pathlib.Path
    ) -> None:
        ds = _make_data_source(
            tmp_path,
            processing_levels=[ProcessingLevel.L2SP],
            bands=["B4", "B10"],
        )
        rows = [
            _make_bigquery_row(
                # Real public-index mismatch shape: product_id can be L1-style while
                # base_url points at an L2 product folder.
                product_id="LC08_L1TP_181075_20201028_20201106_02_T1",
                base_url=LC09_L2SP_BASE_URL,
                west_lon=-118.0,
                south_lat=33.0,
                east_lon=-115.0,
                north_lat=35.0,
                sensing_time="2024-03-24T02:17:00Z",
                collection_category="T2",
                wrs_path=1,
                wrs_row=93,
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
        assert items[0].name == LC09_L2SP_PRODUCT_ID
        assert items[0].processing_level == ProcessingLevel.L2SP

    def test_multiple_l1_processing_levels_parse(self, tmp_path: pathlib.Path) -> None:
        ds = _make_data_source(
            tmp_path,
            processing_levels=[ProcessingLevel.L1TP, ProcessingLevel.L1GT],
            bands=["B4"],
        )
        rows = [
            _make_bigquery_row(
                product_id=LC08_L1GT_PRODUCT_ID,
                base_url=LC08_L1GT_BASE_URL,
                west_lon=70.0,
                south_lat=78.0,
                east_lon=74.0,
                north_lat=81.0,
                spacecraft_id="LANDSAT_8",
                sensing_time="2026-06-01T05:59:37Z",
                collection_category="RT",
                wrs_path=155,
                wrs_row=5,
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
        assert items[0].name == LC08_L1GT_PRODUCT_ID
        assert items[0].processing_level == ProcessingLevel.L1GT

    def test_antimeridian_crossing(self, tmp_path: pathlib.Path) -> None:
        """A scene crossing the antimeridian produces a valid multi-polygon."""
        ds = _make_data_source(
            tmp_path,
            sensor_ids=[SensorId.ETM],
            spacecraft_ids=[SpacecraftId.LANDSAT_7],
            bands=["B4"],
        )
        rows = [
            _make_bigquery_row(
                product_id=LE07_L1_PRODUCT_ID,
                base_url=LE07_L1_BASE_URL,
                west_lon=176.5,
                south_lat=50.0,
                east_lon=-177.7,
                north_lat=52.0,
                spacecraft_id="LANDSAT_7",
                sensor_id="ETM",
                sensing_time="2016-07-24T00:00:00Z",
                wrs_path=1,
                wrs_row=91,
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
        ts = datetime(2026, 6, 3, tzinfo=UTC)
        geometry = STGeometry(
            WGS84_PROJECTION,
            shapely.box(-118.0, 33.0, -115.0, 35.0),
            (ts, ts + timedelta(days=1)),
        )

        # Build a GridIndex covering the geometry, tagged with the real scene's WRS
        # path/row.
        wrs2_index = GridIndex(WRS2_GRID_SIZE)
        wrs2_polygon = shapely.box(-119.0, 32.0, -114.0, 36.0)
        wrs2_index.insert(wrs2_polygon.bounds, (wrs2_polygon, "129", "40"))

        rows = [
            _make_bigquery_row(
                west_lon=-118.0,
                south_lat=33.0,
                east_lon=-115.0,
                north_lat=35.0,
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
                sensor_ids=[SensorId.OLI_TIRS],
                processing_levels=[ProcessingLevel.L1TP],
                bands=["B4"],
                use_rtree_index=False,
            )
            result = ds.get_items([geometry], QueryConfig())

        # One query for the whole get_items call.
        assert mock_client.query.call_count == 1

        # The real path/row lookup must have produced the 129/40 filter, passed as a
        # "path,row" entry in the @pathrows query parameter.
        query_str = mock_client.query.call_args[0][0]
        job_config = mock_client.query.call_args[1]["job_config"]
        assert "IN UNNEST(@pathrows)" in query_str
        pathrow_params = [
            p for p in job_config.query_parameters if p.name == "pathrows"
        ]
        assert len(pathrow_params) == 1
        assert "129,40" in pathrow_params[0].values

        # The candidate item should be matched into a single group for the geometry.
        assert len(result) == 1
        group = result[0]
        assert len(group) == 1
        matched_names = [item.name for item in group[0].items]
        assert matched_names == [LC09_L1_PRODUCT_ID]


class TestGetAssetUrl:
    """Tests for get_asset_url."""

    def test_returns_gs_url(self, tmp_path: pathlib.Path) -> None:
        """get_asset_url returns a gs:// URL."""
        ds = _make_data_source(tmp_path)
        item = _make_item()

        url = ds.get_asset_url(item, "B4")

        expected = (
            f"gs://{BUCKET_NAME}/"
            "LC09/L1/02/129/040/LC09_L1TP_129040_20260603_20260603_02_T1/"
            "LC09_L1TP_129040_20260603_20260603_02_T1_B4.TIF"
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
        item = _make_item()
        assert ds._band_to_file_token(item, "B4") == "B4"
        assert ds._band_to_file_token(item, "B10") == "B10"

    @pytest.mark.parametrize(
        ("sensor_id", "thermal_band"),
        [(SensorId.OLI_TIRS, "B10"), (SensorId.TM, "B6"), (SensorId.ETM, "B6")],
    )
    def test_level2_thermal_and_reflectance(
        self, tmp_path: pathlib.Path, sensor_id: SensorId, thermal_band: str
    ) -> None:
        ds = _make_data_source(
            tmp_path,
            sensor_ids=[sensor_id],
            processing_levels=[ProcessingLevel.L2SP],
            bands=["B4", thermal_band],
        )
        item = _make_item(
            LC09_L2SP_PRODUCT_ID,
            sensor_id=sensor_id,
            processing_level=ProcessingLevel.L2SP,
        )
        assert ds._band_to_file_token(item, thermal_band) == f"ST_{thermal_band}"
        # A non-thermal band is surface reflectance.
        assert ds._band_to_file_token(item, "B4") == "SR_B4"

    def test_l2sr_reflectance_band(self, tmp_path: pathlib.Path) -> None:
        ds = _make_data_source(
            tmp_path,
            processing_levels=[ProcessingLevel.L2SR],
            bands=["B4"],
        )
        item = _make_item(
            LC09_L2SR_PRODUCT_ID,
            sensor_id=SensorId.OLI_TIRS,
            processing_level=ProcessingLevel.L2SR,
        )

        assert ds._band_to_file_token(item, "B4") == "SR_B4"
