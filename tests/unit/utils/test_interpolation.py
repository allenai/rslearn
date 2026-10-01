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
