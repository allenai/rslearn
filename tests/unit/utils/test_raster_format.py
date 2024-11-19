import pathlib

import numpy as np
import rasterio
from rasterio.crs import CRS
from upath import UPath

from rslearn.const import Projection
from rslearn.utils.raster_format import GeotiffRasterFormat


def test_geotiff_tiling(tmp_path: pathlib.Path) -> None:
    path = UPath(tmp_path)
    block_size = 128
    projection = Projection(CRS.from_epsg(3857), 1, -1)

    # If always_enable_tiling=False, it should create tiled GeoTIFF only if one
    # of the dimensions exceeds the block size.
    # For some reason the GeoTIFF still ends up being tiled if the dimensions are the
    # same in some cases so here we set them different.
    array = np.zeros((1, 60, 64), dtype=np.uint8)
    GeotiffRasterFormat(
        block_size=block_size, always_enable_tiling=False
    ).encode_raster(path, projection, (0, 0, 64, 60), array)
    with (path / "geotiff.tif").open("rb") as f:
        with rasterio.open(f) as raster:
            assert not raster.profile["tiled"]

    array = np.zeros((1, 252, 256), dtype=np.uint8)
    GeotiffRasterFormat(
        block_size=block_size, always_enable_tiling=False
    ).encode_raster(path, projection, (0, 0, 256, 252), array)
    with (path / "geotiff.tif").open("rb") as f:
        with rasterio.open(f) as raster:
            assert raster.profile["tiled"]

    # If always_enable_tiling=True it should create tiled GeoTIFF either way.
    array = np.zeros((1, 60, 64), dtype=np.uint8)
    GeotiffRasterFormat(block_size=block_size, always_enable_tiling=True).encode_raster(
        path, projection, (0, 0, 64, 60), array
    )
    with (path / "geotiff.tif").open("rb") as f:
        with rasterio.open(f) as raster:
            assert raster.profile["tiled"]


def test_geotiff_out_of_bounds(tmp_path: pathlib.Path) -> None:
    # GeotiffRasterFormat should log warning but return zero array if we request a
    # tile that is fully out of bounds.
    # If it is partially out of bounds, it shouldn't warn and should just return the
    # partial content.
    path = UPath(tmp_path)
    projection = Projection(CRS.from_epsg(3857), 1, -1)
    format = GeotiffRasterFormat()

    array = np.ones((1, 8, 8), dtype=np.uint8)
    format.encode_raster(path, projection, (0, 0, 8, 8), array)

    array = format.decode_raster(path, (2, 2, 6, 6))
    assert array.shape == (1, 4, 4)
    assert np.all(array == 1)

    array = format.decode_raster(path, (4, 4, 12, 12))
    assert array.shape == (1, 8, 8)
    assert np.all(array[:, 0:4, 0:4] == 1)
    assert np.all(array[:, 0:8, 4:8] == 0)

    array = format.decode_raster(path, (8, 8, 12, 12))
    assert array.shape == (1, 4, 4)
    assert np.all(array == 0)
