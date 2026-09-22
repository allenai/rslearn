"""Regression tests for fixed-size NumPy layers used alongside spatial crops."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from rasterio.crs import CRS
from upath import UPath

from rslearn.config.dataset import BandSetConfig, DType
from rslearn.utils.geometry import Projection
from rslearn.utils.raster_array import RasterArray, RasterMetadata


@pytest.mark.parametrize("spatial_size", [(1, 1), (2, 3)])
@pytest.mark.parametrize("crop_offset", [0, 31])
def test_numpy_spatial_size_preserves_array_and_timestamps(
    tmp_path: Path, spatial_size: tuple[int, int], crop_offset: int
) -> None:
    band_set = BandSetConfig(
        dtype=DType.FLOAT32,
        bands=["temperature", "rainfall"],
        spatial_size=spatial_size,
        format={"class_path": "rslearn.utils.raster_format.NumpyRasterFormat"},
    )
    projection = Projection(CRS.from_epsg(32613), 10, -10)
    # Realistic non-aligned origins, including a negative coordinate. A 1/63
    # resolution factor would not reproduce spatial_size's rounded origin.
    bounds = (44783, -433277, 44846, -433214)
    native_projection, native_bounds = band_set.get_final_projection_and_bounds(
        projection, bounds
    )
    h, w = spatial_size
    array = np.arange(2 * 448 * h * w, dtype=np.float32).reshape(2, 448, h, w)
    start = datetime(2020, 1, 1, tzinfo=UTC)
    timestamps = [
        (start + timedelta(days=i), start + timedelta(days=i + 1)) for i in range(448)
    ]
    raster = RasterArray(
        array=array,
        timestamps=timestamps,
        metadata=RasterMetadata(nodata_value=-9999),
    )
    fmt = band_set.instantiate_raster_format()
    path = UPath(tmp_path)
    fmt.encode_raster(path, native_projection, native_bounds, raster)
    crop = (
        bounds[0] + crop_offset,
        bounds[1] + crop_offset,
        bounds[0] + crop_offset + 32,
        bounds[1] + crop_offset + 32,
    )
    result = fmt.decode_raster(path, projection, crop)
    np.testing.assert_array_equal(result.array, array)
    assert result.timestamps == timestamps
    assert result.metadata.nodata_value == -9999
    # A cached format instance must keep the same native-array semantics.
    again = band_set.instantiate_raster_format().decode_raster(path, projection, crop)
    np.testing.assert_array_equal(again.array, array)


def test_numpy_without_spatial_size_keeps_checks_and_cropping(tmp_path: Path) -> None:
    band_set = BandSetConfig(
        dtype=DType.FLOAT32,
        bands=["value"],
        format={"class_path": "rslearn.utils.raster_format.NumpyRasterFormat"},
    )
    fmt = band_set.instantiate_raster_format()
    projection = Projection(CRS.from_epsg(32613), 10, -10)
    array = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    path = UPath(tmp_path)
    fmt.encode_raster(path, projection, (0, 0, 4, 4), RasterArray(array=array))
    result = fmt.decode_raster(path, projection, (1, 1, 3, 3))
    np.testing.assert_array_equal(result.array, array[:, :, 1:3, 1:3])
    with pytest.raises(NotImplementedError, match="does not support reprojection"):
        fmt.decode_raster(path, Projection(projection.crs, 20, -20), (0, 0, 2, 2))


def test_geotiff_spatial_size_still_resamples_requested_crop(tmp_path: Path) -> None:
    band_set = BandSetConfig(
        dtype=DType.FLOAT32,
        bands=["value"],
        spatial_size=(1, 1),
        format={"class_path": "rslearn.utils.raster_format.GeotiffRasterFormat"},
    )
    projection = Projection(CRS.from_epsg(32613), 10, -10)
    native_projection, native_bounds = band_set.get_final_projection_and_bounds(
        projection, (0, 0, 63, 63)
    )
    fmt = band_set.instantiate_raster_format()
    path = UPath(tmp_path)
    fmt.encode_raster(
        path,
        native_projection,
        native_bounds,
        RasterArray(array=np.full((1, 1, 1, 1), 7, dtype=np.float32)),
    )
    result = fmt.decode_raster(path, projection, (0, 0, 32, 32))
    np.testing.assert_array_equal(result.array, np.full((1, 1, 32, 32), 7))
