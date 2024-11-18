import pathlib

import numpy as np
from rasterio.crs import CRS
from upath import UPath

from rslearn.tile_stores.default import DefaultTileStore
from rslearn.utils.geometry import Projection

LAYER_NAME = "layer"
ITEM_NAME = "item"
BANDS = ["B1"]


def test_rectangle_read(tmp_path: pathlib.Path) -> None:
    # Make sure that when we read a rectangle with different width/height it returns
    # the right shape.
    ds_path = UPath(tmp_path)
    tile_store = DefaultTileStore()
    tile_store.set_dataset_path(ds_path)
    projection = Projection(CRS.from_epsg(3857), 1, -1)
    # Write square.
    raster_size = 4
    tile_store.write_raster(
        LAYER_NAME,
        ITEM_NAME,
        BANDS,
        projection,
        (0, 0, raster_size, raster_size),
        np.zeros((len(BANDS), raster_size, raster_size), dtype=np.uint8),
    )
    # Read rectangle.
    width = 2
    height = 3
    result = tile_store.read_raster(
        LAYER_NAME, ITEM_NAME, BANDS, projection, (0, 0, width, height)
    )
    assert result.shape == (len(BANDS), height, width)


def test_partial_read(tmp_path: pathlib.Path) -> None:
    # Make sure that if we read an array that partially overlaps the raster, the
    # portion overlapping the raster has right value while the rest is zero.
    ds_path = UPath(tmp_path)
    tile_store = DefaultTileStore()
    tile_store.set_dataset_path(ds_path)
    projection = Projection(CRS.from_epsg(3857), 1, -1)
    # Write ones.
    raster_size = 4
    tile_store.write_raster(
        LAYER_NAME,
        ITEM_NAME,
        BANDS,
        projection,
        (0, 0, raster_size, raster_size),
        np.ones((len(BANDS), raster_size, raster_size), dtype=np.uint8),
    )
    # Now read an offset square.
    result = tile_store.read_raster(
        LAYER_NAME, ITEM_NAME, BANDS, projection, (2, 2, 6, 6)
    )
    assert np.all(result[:, 0:2, 0:2] == 1)
    assert np.all(result[:, :, 2:4] == 0)
    assert np.all(result[:, 2:4, :] == 0)
