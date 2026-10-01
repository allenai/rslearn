import numpy as np

from rslearn.utils.interpolation import interpolate_to_grid


def test_interpolate_to_grid_linear_griddata() -> None:
    # Four corner samples spanning [0, 2] in both axes, with value = lon + 10 * lat.
    # Linear interpolation of a plane is exact, so each cell should equal the plane
    # evaluated at the cell center.
    data = np.array([[0.0, 2.0, 20.0, 22.0]], dtype=np.float32)
    lon = np.array([0.0, 2.0, 0.0, 2.0], dtype=np.float64)
    lat = np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float64)

    grid, projection, bounds = interpolate_to_grid(
        data=data,
        lon=lon,
        lat=lat,
        grid_resolution=1.0,
    )

    assert bounds == (0, 0, 3, 3)
    assert grid.shape == (1, 3, 3)
    # Row index increases with latitude (positive y_resolution).
    np.testing.assert_allclose(grid[0, 0, 0], 0.5 + 10 * 0.5, rtol=1e-6)
    np.testing.assert_allclose(grid[0, 0, 1], 1.5 + 10 * 0.5, rtol=1e-6)
    np.testing.assert_allclose(grid[0, 1, 0], 0.5 + 10 * 1.5, rtol=1e-6)
    np.testing.assert_allclose(grid[0, 1, 1], 1.5 + 10 * 1.5, rtol=1e-6)
    # Cells whose center lies outside the swath hull are nodata.
    assert grid[0, 2, 2] == 0.0
    assert projection.x_resolution == 1.0
    assert projection.y_resolution == 1.0


def test_interpolate_to_grid_samples_at_pixel_centers() -> None:
    # Regression test for a half-pixel geolocation shift: a feature at a known
    # lon/lat must land in the pixel whose georeferenced footprint contains it.
    res = 0.01
    lon_1d = np.arange(10.0, 10.2 + 1e-9, 0.002)
    lat_1d = np.arange(45.0, 45.2 + 1e-9, 0.002)
    lon, lat = np.meshgrid(lon_1d, lat_1d)
    # Plane with distinct gradients in x and y.
    values = 1000 * (lon - 10.0) + 100 * (lat - 45.0)
    data = values[None, :, :].astype(np.float32)

    grid, projection, bounds = interpolate_to_grid(
        data=data, lon=lon, lat=lat, grid_resolution=res
    )

    col, row = 7, 5
    center_lon = (bounds[0] + col + 0.5) * projection.x_resolution
    center_lat = (bounds[1] + row + 0.5) * projection.y_resolution
    expected = 1000 * (center_lon - 10.0) + 100 * (center_lat - 45.0)
    np.testing.assert_allclose(grid[0, row, col], expected, atol=1e-3)


def test_interpolate_to_grid_custom_nodata() -> None:
    data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    lon = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    lat = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    grid, _, _ = interpolate_to_grid(
        data=data,
        lon=lon,
        lat=lat,
        grid_resolution=1.0,
        nodata_value=-9999.0,
    )

    assert grid[0, 1, 1] == -9999.0
