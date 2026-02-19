"""Utilities for interpolating irregular lon/lat samples onto a grid."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from rasterio.crs import CRS
from scipy.interpolate import griddata

from rslearn.log_utils import get_logger
from rslearn.utils.geometry import PixelBounds, Projection, WGS84_EPSG

logger = get_logger(__name__)

NODATA_VALUE = 0.0


def interpolate_to_grid(
    data: npt.NDArray,
    lon: npt.NDArray,
    lat: npt.NDArray,
    grid_resolution: float,
) -> tuple[npt.NDArray, Projection, PixelBounds]:
    """Interpolate points onto a fixed-resolution grid.

    Args:
        data: the band data to convert (CxN).
        lon: longitude of each pixel (N). Pixels with NaN longitude are ignored.
        lat: latitude of each pixel (N). Pixels with NaN latitude are ignored.
        grid_resolution: the resolution of the grid.

    Returns:
        a tuple (array, projection, bounds) containing the gridded array along with
            the projection and bounds of that array.
    """
    lon_flat = np.asarray(lon).reshape(-1)
    lat_flat = np.asarray(lat).reshape(-1)
    flat_data = np.asarray(data).reshape(data.shape[0], -1)
    if flat_data.shape[1] != lon_flat.size:
        raise ValueError(
            "expected lon/lat to match data points; "
            f"got {flat_data.shape[1]} data points and {lon_flat.size} lon/lat points"
        )

    valid = np.isfinite(lon_flat) & np.isfinite(lat_flat)
    if not np.any(valid):
        raise ValueError("no valid lon/lat points to interpolate")

    lon_valid = lon_flat[valid]
    lat_valid = lat_flat[valid]
    data_valid = flat_data[:, valid]

    logger.debug(
        "Identified lon/lat valid bounds to interpolate: %s %s %s %s",
        lon_valid.min(),
        lat_valid.min(),
        lon_valid.max(),
        lat_valid.max(),
    )
    bounds = (
        math.floor(lon_valid.min() / grid_resolution),
        math.floor(lat_valid.min() / grid_resolution),
        math.floor(lon_valid.max() / grid_resolution) + 1,
        math.floor(lat_valid.max() / grid_resolution) + 1,
    )

    logger.debug(
        "Computing initial gridded array with bounds=%s grid_resolution=%s",
        bounds,
        grid_resolution,
    )

    num_bands = data.shape[0]
    height = bounds[3] - bounds[1]
    width = bounds[2] - bounds[0]
    # Construct lon/lat coordinates for each grid cell in the output.
    xs = (np.arange(bounds[0], bounds[2]) * grid_resolution).astype(np.float64)
    ys = (np.arange(bounds[1], bounds[3]) * grid_resolution).astype(np.float64)
    grid_lon, grid_lat = np.meshgrid(xs, ys)

    gridded_array = NODATA_VALUE * np.ones(
        (num_bands, height, width), dtype=np.float32
    )
    points = np.column_stack([lon_valid, lat_valid])

    for band in range(num_bands):
        values = data_valid[band]
        band_valid = np.isfinite(values)
        if not np.any(band_valid):
            continue
        grid = griddata(
            points[band_valid],
            values[band_valid],
            (grid_lon, grid_lat),
            method="linear",
        )
        grid = np.where(np.isfinite(grid), grid, NODATA_VALUE).astype(np.float32)
        gridded_array[band] = grid

    projection = Projection(CRS.from_epsg(WGS84_EPSG), grid_resolution, grid_resolution)
    return (gridded_array, projection, bounds)
