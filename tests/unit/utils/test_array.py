"""Tests for rslearn.utils.array."""

import numpy as np
import pytest

from rslearn.utils.array import nodata_eq, unique_nodata_value


class TestNodataEq:
    """Unit tests for the NaN-aware nodata_eq helper."""

    def test_integer_nodata(self) -> None:
        arr = np.array([0, 1, 2, 0, 3])
        result = nodata_eq(arr, 0)
        np.testing.assert_array_equal(result, [True, False, False, True, False])

    def test_float_nodata(self) -> None:
        arr = np.array([0.0, 1.5, -9999.0, 2.0], dtype=np.float32)
        result = nodata_eq(arr, -9999.0)
        np.testing.assert_array_equal(result, [False, False, True, False])

    def test_nan_nodata(self) -> None:
        arr = np.array([1.0, np.nan, 3.0, np.nan], dtype=np.float32)
        result = nodata_eq(arr, float("nan"))
        np.testing.assert_array_equal(result, [False, True, False, True])

    def test_multidimensional(self) -> None:
        """Scalar nodata works across all elements of a multi-dim array."""
        arr = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        )
        result = nodata_eq(arr, 0.0)
        expected = np.array(
            [
                [True, False, True],
                [True, False, True],
            ]
        )
        np.testing.assert_array_equal(result, expected)


class TestUniqueNodataValue:
    """Unit tests for unique_nodata_value."""

    def test_single_value(self) -> None:
        assert unique_nodata_value([0]) == 0

    def test_duplicate_values(self) -> None:
        assert unique_nodata_value([-9999.0, -9999.0, -9999.0]) == -9999.0

    def test_nan_values(self) -> None:
        result = unique_nodata_value([float("nan"), float("nan")])
        assert isinstance(result, float) and np.isnan(result)

    def test_multiple_distinct_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected exactly one unique"):
            unique_nodata_value([0, 1])

    def test_empty_returns_none(self) -> None:
        assert unique_nodata_value([]) is None
