import pathlib

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from upath import UPath

from rslearn.utils.geometry import Projection
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


class TestGeotiffInOrOutOfBounds:
    PROJECTION = Projection(CRS.from_epsg(3857), 1, -1)

    @pytest.fixture
    def encoded_raster_path(self, tmp_path: pathlib.Path) -> UPath:
        path = UPath(tmp_path)
        array = np.ones((1, 8, 8), dtype=np.uint8)
        GeotiffRasterFormat().encode_raster(path, self.PROJECTION, (0, 0, 8, 8), array)
        return path

    def test_geotiff_in_bounds(self, encoded_raster_path: UPath) -> None:
        array = GeotiffRasterFormat().decode_raster(
            encoded_raster_path, self.PROJECTION, (2, 2, 6, 6)
        )
        assert array.shape == (1, 4, 4)
        assert np.all(array == 1)

    def test_geotiff_partial_overlap(self, encoded_raster_path: UPath) -> None:
        array = GeotiffRasterFormat().decode_raster(
            encoded_raster_path, self.PROJECTION, (4, 4, 12, 12)
        )
        assert array.shape == (1, 8, 8)
        assert np.all(array[:, 0:4, 0:4] == 1)
        assert np.all(array[:, 0:8, 4:8] == 0)

    def test_geotiff_out_of_bounds(self, encoded_raster_path: UPath) -> None:
        array = GeotiffRasterFormat().decode_raster(
            encoded_raster_path, self.PROJECTION, (8, 8, 12, 12)
        )
        assert array.shape == (1, 4, 4)
        assert np.all(array == 0)


def test_geotiff_compress_zstd(tmp_path: pathlib.Path) -> None:
    # Make sure we can use ZSTD compression successfully.
    path = UPath(tmp_path)
    projection = Projection(CRS.from_epsg(3857), 1, -1)
    array = np.zeros((1, 4, 4))
    raster_format = GeotiffRasterFormat(
        geotiff_options=dict(
            compress="zstd",
        )
    )
    raster_format.encode_raster(path, projection, (0, 0, 4, 4), array)
    with rasterio.open(path / "geotiff.tif") as raster:
        assert raster.profile["compress"] == "zstd"


def test_geotiff_write_nodata_val(tmp_path: pathlib.Path) -> None:
    """Test that nodata_val is correctly set when writing a GeoTIFF."""
    path = UPath(tmp_path)
    projection = Projection(CRS.from_epsg(3857), 1, -1)
    array = np.zeros((1, 4, 4), dtype=np.float32)
    nodata_val = -9999.0

    GeotiffRasterFormat().encode_raster(
        path, projection, (0, 0, 4, 4), array, nodata_val=nodata_val
    )

    with rasterio.open(path / "geotiff.tif") as raster:
        assert raster.nodata == nodata_val


def test_geotiff_read_nodata_val_out_of_bounds(tmp_path: pathlib.Path) -> None:
    """Test that nodata_val is used to fill pixels outside source bounds when reading."""
    path = UPath(tmp_path)
    projection = Projection(CRS.from_epsg(3857), 1, -1)
    nodata_val = -9999.0

    # Create a raster with value 1 and nodata=0.
    array = np.ones((1, 4, 4), dtype=np.float32)
    GeotiffRasterFormat().encode_raster(
        path, projection, (0, 0, 4, 4), array, nodata_val=0
    )

    # Read a region that partially overlaps the source raster.
    # We override the nodata_val to -9999.
    # In the out-of-bounds portions it should be filled in as -9999.
    result = GeotiffRasterFormat().decode_raster(
        path, projection, (2, 2, 8, 8), nodata_val=nodata_val
    )

    assert result.shape == (1, 6, 6)
    # Top-left 2x2 region overlaps source and should have value 1.
    assert np.all(result[:, 0:2, 0:2] == 1)
    # Pixels outside source bounds should be filled with nodata_val.
    assert np.all(result[:, 2:6, :] == nodata_val)
    assert np.all(result[:, :, 2:6] == nodata_val)


def test_geotiff_read_nodata_val_orig_nodata(tmp_path: pathlib.Path) -> None:
    """Test reading from GeoTIFF that has nodata values, but we set different nodata_val.

    Since we override the nodata value in the read operation, the source pixels that
    were a different nodata value should still have the original value.
    """
    path = UPath(tmp_path)
    projection = Projection(CRS.from_epsg(3857), 1, -1)
    original_nodata = -9999.0
    new_nodata = -1.0

    # Create a raster where some pixels have the nodata value -9999.
    # Other pixels are valid (1).
    array = np.ones((1, 4, 4), dtype=np.float32)
    array[:, 2:4, 2:4] = original_nodata
    GeotiffRasterFormat().encode_raster(
        path, projection, (0, 0, 4, 4), array, nodata_val=original_nodata
    )

    # Decode with a different nodata_val.
    # The pixels that were originally nodata should be unchanged.
    result = GeotiffRasterFormat().decode_raster(
        path, projection, (0, 0, 4, 4), nodata_val=new_nodata
    )

    assert result.shape == (1, 4, 4)
    # Valid data pixels should still have value 1.
    assert np.all(result[:, 0:2, :] == 1)
    assert np.all(result[:, :, 0:2] == 1)
    # Original nodata pixels should still be as they were.
    assert np.all(result[:, 2:4, 2:4] == original_nodata)
