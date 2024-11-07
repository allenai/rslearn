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
