"""Integration tests for the Meta CHMv2 data source (S3 COG-backed, mocked)."""

import io
import json
import pathlib
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
from rslearn.data_sources.data_source import Item
from rslearn.data_sources.meta_canopy_height import MetaCanopyHeightV2
from rslearn.utils.geometry import STGeometry
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat

PIXEL_VALUE = 25  # canopy height in meters

# A quadkey tile covering the seattle2020 fixture.
TILE_NAME = "0230102123"
TILE_BOUNDS = (-123, 47, -122, 48)


@pytest.fixture
def test_geotiff(tmp_path: pathlib.Path) -> UPath:
    """Create a 3x3 pixel uint8 GeoTIFF at 1 degree/pixel matching TILE_BOUNDS."""
    array = np.full((1, 3, 3), PIXEL_VALUE, dtype=np.uint8)
    raster_dir = UPath(tmp_path / "raster")
    fmt = GeotiffRasterFormat()
    fmt.encode_raster(
        raster_dir, WGS84_PROJECTION, TILE_BOUNDS, RasterArray(chw_array=array)
    )
    return raster_dir / fmt.fname


def _make_tiles_geojson() -> bytes:
    """Create a minimal tiles.geojson FeatureCollection with one tile over Seattle."""
    tile_box = shapely.box(*TILE_BOUNDS)
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"tile": TILE_NAME},
                "geometry": shapely.geometry.mapping(tile_box),
            }
        ],
    }
    return json.dumps(fc).encode()


def _make_mock_s3() -> MagicMock:
    """Create a mock boto3 S3 client that serves tiles.geojson."""
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": io.BytesIO(_make_tiles_geojson())}
    return mock_s3


def test_get_items(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that get_items returns the correct tile for a Seattle query."""
    monkeypatch.setattr(boto3, "client", lambda *a, **kw: _make_mock_s3())

    data_source = MetaCanopyHeightV2(metadata_cache_dir=str(tmp_path / "cache"))
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    assert len(item_groups) == 1
    assert len(item_groups[0].items) == 1
    item = item_groups[0].items[0]
    assert isinstance(item, Item)
    assert item.name == TILE_NAME


def test_get_item_by_name(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test looking up a tile by its quadkey name."""
    monkeypatch.setattr(boto3, "client", lambda *a, **kw: _make_mock_s3())

    data_source = MetaCanopyHeightV2(metadata_cache_dir=str(tmp_path / "cache"))
    item = data_source.get_item_by_name(TILE_NAME)
    assert item.name == TILE_NAME
    with pytest.raises(ValueError):
        data_source.get_item_by_name("nonexistent")


def test_get_asset_url(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the asset URL points at the expected COG path."""
    monkeypatch.setattr(boto3, "client", lambda *a, **kw: _make_mock_s3())

    data_source = MetaCanopyHeightV2(metadata_cache_dir=str(tmp_path / "cache"))
    item = data_source.get_item_by_name(TILE_NAME)
    url = data_source.get_asset_url(item, "chm")
    assert url.endswith(
        f"forests/v2/global/dinov3_global_chm_v2_ml3/chm/{TILE_NAME}.tif"
    )


def test_direct_materialize(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    test_geotiff: UPath,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test read_raster (direct materialization) by redirecting COG reads to local file."""
    monkeypatch.setattr(boto3, "client", lambda *a, **kw: _make_mock_s3())

    data_source = MetaCanopyHeightV2(metadata_cache_dir=str(tmp_path / "cache"))
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    item = item_groups[0].items[0]

    original_open = rasterio.open

    def mock_rasterio_open(url: Any, *args: Any, **kwargs: Any) -> Any:
        if "dataforgood-fb-data" in str(url):
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
        item=item,
        bands=["canopy_height"],
        projection=seattle2020.projection,
        bounds=bounds,
    )
    assert (array.get_chw_array() == PIXEL_VALUE).all()
