import numpy as np

from rslearn.utils.interpolation import interpolate_to_grid


def test_interpolate_to_grid_linear_griddata() -> None:
    # Four corner samples to define a 2x2 grid.
    data = np.array([[10.0, 0.0, 0.0, 4.0]], dtype=np.float32)
    lon = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    lat = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)

    grid, projection, bounds = interpolate_to_grid(
        data=data,
        lon=lon,
        lat=lat,
        grid_resolution=1.0,
        dilation_steps=0,
    )

    assert bounds == (0, 0, 2, 2)
    assert grid.shape == (1, 2, 2)
    np.testing.assert_allclose(grid[0, 0, 0], 10.0, rtol=1e-6)
    np.testing.assert_allclose(grid[0, 0, 1], 0.0, rtol=1e-6)
    np.testing.assert_allclose(grid[0, 1, 0], 0.0, rtol=1e-6)
    np.testing.assert_allclose(grid[0, 1, 1], 4.0, rtol=1e-6)
    assert projection.x_resolution == 1.0
    assert projection.y_resolution == 1.0
