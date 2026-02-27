"""Integration tests for the WorldCover data source (S3 COG-backed)."""

import io
import json
import pathlib
from typing import Any
from unittest.mock import MagicMock

import boto3
import numpy as np
import pytest
import rasterio
import requests
import shapely
from upath import UPath

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import Item
from rslearn.data_sources.worldcover import WorldCover
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import STGeometry
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat

SEATTLE_POINT = shapely.Point(-122.33, 47.61)
PIXEL_VALUE = 40  # "Cropland" class

# The 3x3 degree WorldCover tile covering Seattle.
TILE_NAME = "N45W123"
TILE_BOUNDS = (-123, 45, -120, 48)


@pytest.fixture
def test_geotiff(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a 3x3 pixel uint8 GeoTIFF at 1 degree/pixel matching TILE_BOUNDS."""
    bounds = TILE_BOUNDS
    array = np.full((1, 3, 3), PIXEL_VALUE, dtype=np.uint8)
    raster_dir = UPath(tmp_path / "raster")
    fmt = GeotiffRasterFormat()
    fmt.encode_raster(
        raster_dir, WGS84_PROJECTION, bounds, RasterArray(chw_array=array)
    )
    return raster_dir / fmt.fname


def _make_grid_geojson() -> bytes:
    """Create a minimal GeoJSON FeatureCollection with one tile over Seattle."""
    tile_box = shapely.box(*TILE_BOUNDS)
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"ll_tile": TILE_NAME},
                "geometry": shapely.geometry.mapping(tile_box),
            }
        ],
    }
    return json.dumps(fc).encode()


def _make_mock_s3() -> MagicMock:
    """Create a mock boto3 S3 client that serves the grid GeoJSON."""
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": io.BytesIO(_make_grid_geojson())}
    return mock_s3


def test_get_items(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that get_items returns the correct tile for a Seattle query."""
    monkeypatch.setattr(boto3, "client", lambda *a, **kw: _make_mock_s3())

    data_source = WorldCover(metadata_cache_dir=str(tmp_path / "cache"))
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    assert len(item_groups) == 1
    assert len(item_groups[0]) == 1
    item = item_groups[0][0]
    assert isinstance(item, Item)
    assert item.name == TILE_NAME


def test_ingest(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    test_geotiff: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test ingesting a WorldCover tile into a tile store."""
    monkeypatch.setattr(boto3, "client", lambda *a, **kw: _make_mock_s3())

    # Mock requests.get to serve our test GeoTIFF instead of hitting S3.
    geotiff_bytes = test_geotiff.read_bytes()

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

        def iter_content(self, chunk_size: int = 8192) -> Any:
            yield geotiff_bytes

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    monkeypatch.setattr(requests, "get", lambda *a, **kw: FakeResponse())

    data_source = WorldCover(metadata_cache_dir=str(tmp_path / "cache"))
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
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
    assert tile_store.is_raster_ready(layer_name, item.name, ["B1"])

    bounds = (
        int(seattle2020.shp.bounds[0]),
        int(seattle2020.shp.bounds[1]),
        int(seattle2020.shp.bounds[2]),
        int(seattle2020.shp.bounds[3]),
    )
    raster_data = tile_store.read_raster(
        layer_name, item.name, ["B1"], seattle2020.projection, bounds
    )
    assert (raster_data.get_chw_array() == PIXEL_VALUE).all()


def test_direct_materialize(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    test_geotiff: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test read_raster (direct materialization) by redirecting COG reads to local file."""
    monkeypatch.setattr(boto3, "client", lambda *a, **kw: _make_mock_s3())

    data_source = WorldCover(metadata_cache_dir=str(tmp_path / "cache"))
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    item = item_groups[0][0]

    # Change rasterio.open to open our local test GeoTIFF.
    original_open = rasterio.open

    def mock_rasterio_open(url: Any, *args: Any, **kwargs: Any) -> Any:
        if "esa-worldcover" in str(url):
            return original_open(str(test_geotiff), *args, **kwargs)
        return original_open(url, *args, **kwargs)

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
        bands=["B1"],
        projection=seattle2020.projection,
        bounds=bounds,
    )
    assert (array.get_chw_array() == PIXEL_VALUE).all()
