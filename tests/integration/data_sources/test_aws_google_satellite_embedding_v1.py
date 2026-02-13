"""Mocked integration tests for the Google Satellite Embedding V1 data source."""

import io
import pathlib
import shutil
from typing import Any
from unittest.mock import MagicMock

import boto3
import numpy as np
import pytest
import rasterio
import shapely
from upath import UPath

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.aws_google_satellite_embedding_v1 import (
    BANDS,
    GoogleSatelliteEmbeddingV1,
)
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat

# Seattle area in WGS84
SEATTLE_WGS84_BOUNDS = (-122.34, 47.60, -122.32, 47.62)
DEGREES_PER_PIXEL = 0.001

# Raw pixel value written into the test GeoTIFF (int8, valid range -127..127).
RAW_PIXEL_VALUE = 50

# Expected value after dequantization:
# ((50 / 127.5) ** 2) * sign(50) = (0.3922...) ** 2 * 1 ≈ 0.1538
DEQUANTIZED_VALUE = ((RAW_PIXEL_VALUE / 127.5) ** 2) * np.sign(RAW_PIXEL_VALUE)


@pytest.fixture
def test_geotiff(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a small 64-band int8 GeoTIFF mimicking AEF data."""
    projection = Projection(WGS84_PROJECTION.crs, DEGREES_PER_PIXEL, -DEGREES_PER_PIXEL)
    west, south, east, north = SEATTLE_WGS84_BOUNDS
    bounds = (
        round(west / DEGREES_PER_PIXEL),
        round(north / -DEGREES_PER_PIXEL),
        round(east / DEGREES_PER_PIXEL),
        round(south / -DEGREES_PER_PIXEL),
    )
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    # Use int8 values like the real data (range roughly -127 to 127, -128 is nodata).
    data = np.full((64, height, width), RAW_PIXEL_VALUE, dtype=np.int8)
    raster_dir = UPath(tmp_path / "raster")
    fmt = GeotiffRasterFormat()
    fmt.encode_raster(raster_dir, projection, bounds, RasterArray(chw_array=data))
    return raster_dir / fmt.fname


def _make_csv_content() -> bytes:
    """Create a mock CSV index matching the format expected by the data source.

    The real CSV has 13 columns (with no header):
      0: WKT geometry
      1: CRS (e.g. EPSG:32601)
      2: S3 path to the TIFF file
      3: year
      4: UTM zone (e.g. 1N)
      5-8: CRS bounds (col_min, row_min, col_max, row_max)
      9-12: WGS84 bounds (west, south, east, north)

    The data source reads only columns 0 (WKT), 2 (path), 3 (year).
    """
    wkt = shapely.box(*SEATTLE_WGS84_BOUNDS).wkt
    west, south, east, north = SEATTLE_WGS84_BOUNDS
    line = (
        f'"{wkt}",EPSG:32610,'
        f"s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2020/test_tile.tiff,"
        f"2020,10N,0,0,8192,8192,{west},{south},{east},{north}\n"
    )
    return line.encode()


def _make_mock_s3(
    test_geotiff: pathlib.Path,
) -> MagicMock:
    """Create a mock boto3 S3 client that serves the CSV index and test GeoTIFF."""
    # get_object serves the CSV index, and download_file copies the test GeoTIFF
    # to whatever local path the data source requests (we ignore bucket/key).
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": io.BytesIO(_make_csv_content())}

    def mock_download(bucket: str, key: str, local_path: str) -> None:
        shutil.copy(str(test_geotiff), local_path)

    mock_s3.download_file.side_effect = mock_download
    return mock_s3


@pytest.mark.parametrize(
    "apply_dequantization,expected_value",
    [
        (False, RAW_PIXEL_VALUE),
        (True, pytest.approx(DEQUANTIZED_VALUE, abs=1e-4)),
    ],
    ids=["raw", "dequantized"],
)
def test_ingest(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    test_geotiff: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    apply_dequantization: bool,
    expected_value: float,
) -> None:
    """Test get_items and ingest with mocked S3, with and without dequantization."""
    monkeypatch.setattr(boto3, "client", lambda *a, **kw: _make_mock_s3(test_geotiff))

    data_source = GoogleSatelliteEmbeddingV1(
        metadata_cache_dir=str(tmp_path / "cache"),
        apply_dequantization=apply_dequantization,
    )

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    assert len(item_groups) > 0 and len(item_groups[0]) > 0
    item = item_groups[0][0]

    tile_store_dir = UPath(tmp_path / "tiles")
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)
    layer_name = "layer"

    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name),
        item_groups[0],
        [[seattle2020]],
    )
    assert tile_store.is_raster_ready(layer_name, item.name, BANDS)

    # Read back and verify pixel values.
    bounds = (
        int(seattle2020.shp.bounds[0]),
        int(seattle2020.shp.bounds[1]),
        int(seattle2020.shp.bounds[2]),
        int(seattle2020.shp.bounds[3]),
    )
    array = tile_store.read_raster(
        layer_name, item.name, BANDS, seattle2020.projection, bounds
    )
    assert array.get_chw_array().max() == expected_value


@pytest.mark.parametrize(
    "apply_dequantization,expected_value",
    [
        (False, RAW_PIXEL_VALUE),
        (True, pytest.approx(DEQUANTIZED_VALUE, abs=1e-4)),
    ],
    ids=["raw", "dequantized"],
)
def test_materialize(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    test_geotiff: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    apply_dequantization: bool,
    expected_value: float,
) -> None:
    """Test read_raster (direct materialize path) with and without dequantization."""
    monkeypatch.setattr(boto3, "client", lambda *a, **kw: _make_mock_s3(test_geotiff))

    data_source = GoogleSatelliteEmbeddingV1(
        metadata_cache_dir=str(tmp_path / "cache"),
        apply_dequantization=apply_dequantization,
    )

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    item = item_groups[0][0]

    # read_raster uses rasterio to open a /vsicurl/ URL.
    # Redirect it to the local test file instead.
    original_rasterio_open = rasterio.open

    def mock_rasterio_open(url: Any, *args: Any, **kwargs: Any) -> Any:
        if "/vsicurl/" in str(url):
            return original_rasterio_open(str(test_geotiff), *args, **kwargs)
        return original_rasterio_open(url, *args, **kwargs)

    monkeypatch.setattr(rasterio, "open", mock_rasterio_open)

    bounds = (
        int(seattle2020.shp.bounds[0]),
        int(seattle2020.shp.bounds[1]),
        int(seattle2020.shp.bounds[2]),
        int(seattle2020.shp.bounds[3]),
    )

    array = data_source.read_raster(
        layer_name="fake",
        item_name=item.name,
        bands=BANDS,
        projection=seattle2020.projection,
        bounds=bounds,
    )
    assert array.get_chw_array().max() == expected_value
