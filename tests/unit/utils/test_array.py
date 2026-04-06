"""Tests for rslearn.utils.array."""

import numpy as np

from rslearn.utils.array import nodata_eq


class TestNodataEq:
    """Unit tests for the NaN-aware nodata_eq helper."""

    def test_integer_nodata(self) -> None:
        arr = np.array([0, 1, 2, 0, 3])
        nodata = np.array([0])
        result = nodata_eq(arr, nodata)
        np.testing.assert_array_equal(result, [True, False, False, True, False])

    def test_float_nodata(self) -> None:
        arr = np.array([0.0, 1.5, -9999.0, 2.0], dtype=np.float32)
        nodata = np.array([-9999.0], dtype=np.float32)
        result = nodata_eq(arr, nodata)
        np.testing.assert_array_equal(result, [False, False, True, False])

    def test_nan_nodata(self) -> None:
        arr = np.array([1.0, np.nan, 3.0, np.nan], dtype=np.float32)
        nodata = np.array([np.nan], dtype=np.float32)
        result = nodata_eq(arr, nodata)
        np.testing.assert_array_equal(result, [False, True, False, True])

    def test_nodata_multiple_bands(self) -> None:
        """Per-band nodata where one band uses NaN and another uses 0."""
        arr = np.array(
            [
                [np.nan, 1.0, np.nan],
                [0.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        )
        nodata = np.array([np.nan, 0.0], dtype=np.float32).reshape(2, 1)
        result = nodata_eq(arr, nodata)
        expected = np.array(
            [
                [True, False, True],
                [True, False, True],
            ]
        )
        np.testing.assert_array_equal(result, expected)
