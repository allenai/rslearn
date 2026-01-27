"""Tests for the raster data storage abstraction."""

import tempfile

import numpy as np
import pytest
from upath import UPath

from rslearn.dataset.raster_storage import (
    PerItemGroupStorage,
    PerLayerStorage,
)
from rslearn.utils import Projection
from rslearn.utils.raster_format import GeotiffRasterFormat


@pytest.fixture
def projection() -> Projection:
    """Create a test projection."""
    return Projection("EPSG:4326", 1, -1)


@pytest.fixture
def bounds() -> tuple[int, int, int, int]:
    """Create test bounds."""
    return (0, 0, 32, 32)


@pytest.fixture
def raster_format() -> GeotiffRasterFormat:
    """Create a test raster format."""
    return GeotiffRasterFormat()


@pytest.fixture
def sample_array() -> np.ndarray:
    """Create a sample CHW array."""
    return np.random.randint(0, 255, size=(3, 32, 32), dtype=np.uint8)


class TestPerItemGroupStorage:
    """Tests for PerItemGroupStorage."""

    def test_write_and_read_single_raster(
        self,
        projection: Projection,
        bounds: tuple[int, int, int, int],
        raster_format: GeotiffRasterFormat,
        sample_array: np.ndarray,
    ) -> None:
        """Test writing and reading a single raster."""
        with tempfile.TemporaryDirectory() as tmpdir:
            window_root = UPath(tmpdir)
            storage = PerItemGroupStorage()

            # Write raster
            storage.write_raster(
                window_root,
                "test_layer",
                ["B1", "B2", "B3"],
                group_idx=0,
                raster_format=raster_format,
                projection=projection,
                bounds=bounds,
                array=sample_array,
            )

            # Read raster
            result = storage.read_raster(
                window_root,
                "test_layer",
                ["B1", "B2", "B3"],
                group_idx=0,
                raster_format=raster_format,
                projection=projection,
                bounds=bounds,
            )

            np.testing.assert_array_equal(result, sample_array)

    def test_write_and_read_multiple_groups(
        self,
        projection: Projection,
        bounds: tuple[int, int, int, int],
        raster_format: GeotiffRasterFormat,
    ) -> None:
        """Test writing and reading multiple item groups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            window_root = UPath(tmpdir)
            storage = PerItemGroupStorage()

            # Create TCHW array for all groups
            rasters = np.random.randint(0, 255, size=(3, 3, 32, 32), dtype=np.uint8)

            # Write all rasters
            storage.write_all_rasters(
                window_root,
                "test_layer",
                ["B1", "B2", "B3"],
                raster_format,
                projection,
                bounds,
                rasters,
            )

            # Read all rasters
            result = storage.read_all_rasters(
                window_root,
                "test_layer",
                ["B1", "B2", "B3"],
                num_groups=3,
                raster_format=raster_format,
                projection=projection,
                bounds=bounds,
            )

            assert result.shape == (3, 3, 32, 32)  # TCHW
            np.testing.assert_array_equal(result[0], rasters[0])
            np.testing.assert_array_equal(result[1], rasters[1])
            np.testing.assert_array_equal(result[2], rasters[2])

    def test_directory_structure(
        self,
        projection: Projection,
        bounds: tuple[int, int, int, int],
        raster_format: GeotiffRasterFormat,
        sample_array: np.ndarray,
    ) -> None:
        """Test that the correct directory structure is created.

        This verifies that it is compatible with the directory structure that was
        previously more explicitly defined.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            window_root = UPath(tmpdir)
            storage = PerItemGroupStorage()

            # Write raster for group 0
            storage.write_raster(
                window_root,
                "test_layer",
                ["B1", "B2", "B3"],
                group_idx=0,
                raster_format=raster_format,
                projection=projection,
                bounds=bounds,
                array=sample_array,
            )

            # Write raster for group 1
            storage.write_raster(
                window_root,
                "test_layer",
                ["B1", "B2", "B3"],
                group_idx=1,
                raster_format=raster_format,
                projection=projection,
                bounds=bounds,
                array=sample_array,
            )

            # Check directory structure
            assert (window_root / "layers" / "test_layer" / "B1_B2_B3").exists()
            assert (window_root / "layers" / "test_layer.1" / "B1_B2_B3").exists()


class TestPerLayerStorage:
    """Tests for PerLayerStorage."""

    def test_write_all_and_read_all(
        self,
        projection: Projection,
        bounds: tuple[int, int, int, int],
        raster_format: GeotiffRasterFormat,
    ) -> None:
        """Test writing and reading all rasters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            window_root = UPath(tmpdir)
            storage = PerLayerStorage()

            # Create TCHW array for all groups
            rasters = np.random.randint(0, 255, size=(3, 3, 32, 32), dtype=np.uint8)

            # Write all rasters
            storage.write_all_rasters(
                window_root,
                "test_layer",
                ["B1", "B2", "B3"],
                raster_format,
                projection,
                bounds,
                rasters,
            )

            # Read all rasters efficiently
            result = storage.read_all_rasters(
                window_root,
                "test_layer",
                ["B1", "B2", "B3"],
                num_groups=3,
                raster_format=raster_format,
                projection=projection,
                bounds=bounds,
            )

            assert result.shape == (3, 3, 32, 32)  # TCHW
            np.testing.assert_array_equal(result[0], rasters[0])
            np.testing.assert_array_equal(result[1], rasters[1])
            np.testing.assert_array_equal(result[2], rasters[2])

    def test_read_single_raster(
        self,
        projection: Projection,
        bounds: tuple[int, int, int, int],
        raster_format: GeotiffRasterFormat,
    ) -> None:
        """Test reading a single raster (slow path)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            window_root = UPath(tmpdir)
            storage = PerLayerStorage()

            # Create TCHW array for all groups
            rasters = np.random.randint(0, 255, size=(3, 3, 32, 32), dtype=np.uint8)

            # Write all rasters
            storage.write_all_rasters(
                window_root,
                "test_layer",
                ["B1", "B2", "B3"],
                raster_format,
                projection,
                bounds,
                rasters,
            )

            # Read single raster (should work but is slow)
            result = storage.read_raster(
                window_root,
                "test_layer",
                ["B1", "B2", "B3"],
                group_idx=1,
                raster_format=raster_format,
                projection=projection,
                bounds=bounds,
            )

            assert result.shape == (3, 32, 32)  # CHW
            np.testing.assert_array_equal(result, rasters[1])
