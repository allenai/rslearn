import pathlib

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from upath import UPath

from rslearn.tile_stores.default import DefaultTileStore
from rslearn.utils.geometry import Projection

LAYER_NAME = "layer"
ITEM_NAME = "item"
BANDS = ["B1"]
PROJECTION = Projection(CRS.from_epsg(3857), 1, -1)


@pytest.fixture
def tile_store_with_ones(tmp_path: pathlib.Path) -> DefaultTileStore:
    ds_path = UPath(tmp_path)
    tile_store = DefaultTileStore()
    tile_store.set_dataset_path(ds_path)
    # Write square.
    raster_size = 4
    tile_store.write_raster(
        LAYER_NAME,
        ITEM_NAME,
        BANDS,
        PROJECTION,
        (0, 0, raster_size, raster_size),
        np.ones((len(BANDS), raster_size, raster_size), dtype=np.uint8),
    )
    return tile_store


def test_rectangle_read(tile_store_with_ones: DefaultTileStore) -> None:
    # Make sure that when we read a rectangle with different width/height it returns
    # the right shape.
    width = 2
    height = 3
    result = tile_store_with_ones.read_raster(
        LAYER_NAME, ITEM_NAME, BANDS, PROJECTION, (0, 0, width, height)
    )
    assert result.shape == (len(BANDS), height, width)


def test_partial_read(tile_store_with_ones: DefaultTileStore) -> None:
    # Make sure that if we read an array that partially overlaps the raster, the
    # portion overlapping the raster has right value while the rest is zero.
    result = tile_store_with_ones.read_raster(
        LAYER_NAME, ITEM_NAME, BANDS, PROJECTION, (2, 2, 6, 6)
    )
    # This portion matches the raster which is all ones.
    assert np.all(result[:, 0:2, 0:2] == 1)
    # These portions do not.
    assert np.all(result[:, :, 2:4] == 0)
    assert np.all(result[:, 2:4, :] == 0)


def test_zstd_compression(tmp_path: pathlib.Path) -> None:
    # Make sure we can correctly write a GeoTIFF with ZSTD compression.
    ds_path = UPath(tmp_path)
    tile_store = DefaultTileStore(
        geotiff_options=dict(
            compress="zstd",
        )
    )
    tile_store.set_dataset_path(ds_path)
    raster_size = 4
    tile_store.write_raster(
        LAYER_NAME,
        ITEM_NAME,
        BANDS,
        PROJECTION,
        (0, 0, raster_size, raster_size),
        np.zeros((len(BANDS), raster_size, raster_size), dtype=np.uint8),
    )

    assert tile_store.path is not None
    fname = tile_store.path / LAYER_NAME / ITEM_NAME / "_".join(BANDS) / "geotiff.tif"
    with rasterio.open(fname) as raster:
        assert raster.profile["compress"] == "zstd"
