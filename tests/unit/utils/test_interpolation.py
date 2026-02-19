import numpy as np

from rslearn.utils.interpolation import interpolate_to_grid


def test_interpolate_to_grid_bilinear_weights() -> None:
    # Two samples to expand bounds to 2x1 grid.
    data = np.array([[8.0, 0.0]], dtype=np.float32)
    lon = np.array([0.25, 1.25], dtype=np.float64)
    lat = np.array([0.25, 0.25], dtype=np.float64)

    grid, projection, bounds = interpolate_to_grid(
        data=data,
        lon=lon,
        lat=lat,
        grid_resolution=1.0,
        dilation_steps=0,
    )

    assert bounds == (0, 0, 2, 1)
    assert grid.shape == (1, 1, 2)
    # Expected: cell0 gets only point1 => 8.0, cell1 mixes point1 (0.1875) + point2 (0.5625)
    # value = 1.5 / 0.75 = 2.0
    np.testing.assert_allclose(grid[0, 0, 0], 8.0, rtol=1e-6)
    np.testing.assert_allclose(grid[0, 0, 1], 2.0, rtol=1e-6)
    assert projection.x_resolution == 1.0
    assert projection.y_resolution == 1.0
