"""Unit tests for rslearn.train.dataset."""

import warnings
from collections.abc import Callable
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest
import shapely
import torch
import torch.utils.data
from upath import UPath

from rslearn.config import (
    BandSetConfig,
    DataSourceConfig,
    DType,
    LayerConfig,
    LayerType,
    QueryConfig,
    SpaceMode,
)
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import Item
from rslearn.dataset import Dataset, Window
from rslearn.dataset.window import WindowLayerData
from rslearn.train.dataset import (
    DataInput,
    IndexMode,
    ModelDataset,
    RetryDataset,
    SplitConfig,
    compute_expected_timestamps,
    read_layer_time_range,
)
from rslearn.train.dataset_index import INDEX_DIR_NAME
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.train.transforms.concatenate import Concatenate
from rslearn.utils.geometry import STGeometry
from rslearn.utils.raster_format import GeotiffRasterFormat


class TestException(Exception):
    pass


class DummyTestDataset(torch.utils.data.Dataset):
    def __init__(self, failures: int = 0) -> None:
        # Raise Exception in __getitem__ for the given number of failures before
        # ultimately succeeding.
        self.failures = failures
        self.counter = 0

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> int:
        if idx != 0:
            raise IndexError

        self.counter += 1
        if self.counter <= self.failures:
            raise TestException(f"counter={self.counter} <= failures={self.failures}")
        return 1


def test_retry_dataset() -> None:
    # First try with 3 failures, this should succeed.
    dataset = DummyTestDataset(failures=3)
    dataset = RetryDataset(dataset, retries=3, delay=0.01)
    for _ in dataset:
        pass

    # Now try with 4 failures, it should fail.
    dataset = DummyTestDataset(failures=4)
    dataset = RetryDataset(dataset, retries=3, delay=0.01)
    with pytest.raises(TestException):
        for _ in dataset:
            pass


def test_basic_time_series(
    basic_classification_dataset: Dataset,
    add_window_to_basic_classification_dataset: Callable,
) -> None:
    # Create a window with two images in the first layer to make sure we will be able
    # to load it when explicitly adding a DataInput for it.
    image = np.zeros((1, 4, 4), dtype=np.uint8)
    add_window_to_basic_classification_dataset(
        basic_classification_dataset,
        images={
            ("image_layer1", 0): image,
            ("image_layer1", 1): image,
        },
    )
    dataset = ModelDataset(
        basic_classification_dataset,
        split_config=SplitConfig(
            transforms=[
                Concatenate(
                    {
                        "image0": [],
                        "image1": [],
                    },
                    "image",
                    # concatenate on the time dimension
                    concatenate_dim=1,
                )
            ],
        ),
        task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
        workers=1,
        inputs={
            "image0": DataInput(
                "raster", ["image_layer1"], bands=["band"], passthrough=True
            ),
            "image1": DataInput(
                "raster", ["image_layer1.1"], bands=["band"], passthrough=True
            ),
            "targets": DataInput("vector", ["vector_layer"]),
        },
    )

    assert len(dataset) == 1
    inputs, _, _ = dataset[0]
    assert inputs["image"].image.shape == (1, 2, 4, 4)


def test_load_all_layers(
    basic_classification_dataset: Dataset,
    add_window_to_basic_classification_dataset: Callable,
) -> None:
    """Make sure we can load a time series by using load_all_layers option."""
    # Create a window with two images in the first layer to make sure we will be able
    # to load it when explicitly adding a DataInput for it.
    image = np.zeros((1, 4, 4), dtype=np.uint8)
    add_window_to_basic_classification_dataset(
        basic_classification_dataset,
        images={
            ("image_layer1", 0): image,
            ("image_layer1", 1): image,
        },
    )
    dataset = ModelDataset(
        basic_classification_dataset,
        split_config=SplitConfig(),
        task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
        workers=1,
        inputs={
            "image": DataInput(
                "raster",
                ["image_layer1"],
                bands=["band"],
                passthrough=True,
                load_all_layers=True,
                load_all_item_groups=True,
            ),
            "targets": DataInput("vector", ["vector_layer"]),
        },
    )

    assert len(dataset) == 1
    inputs, _, _ = dataset[0]
    # two layers - timesteps - have been loaded
    assert inputs["image"].image.shape == (1, 2, 4, 4)


def test_load_two_layers(
    basic_classification_dataset: Dataset,
    add_window_to_basic_classification_dataset: Callable,
) -> None:
    """Make sure when load_all_layers is passed we load all of the layer options."""
    # We create a window with two images in the first layer and one image in the second
    # layer. Then in the DataInput we only refer to the second image in the first layer
    # and the only image in the second layer. With load_all_layers but not
    # load_all_item_groups, just these two images should be read.
    add_window_to_basic_classification_dataset(
        basic_classification_dataset,
        images={
            ("image_layer1", 0): 0 * np.ones((1, 4, 4), dtype=np.uint8),
            ("image_layer1", 1): 1 * np.ones((1, 4, 4), dtype=np.uint8),
            ("image_layer2", 0): 2 * np.ones((1, 4, 4), dtype=np.uint8),
        },
    )
    dataset = ModelDataset(
        basic_classification_dataset,
        split_config=SplitConfig(),
        task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
        workers=1,
        inputs={
            "image": DataInput(
                "raster",
                ["image_layer1.1", "image_layer2"],
                bands=["band"],
                passthrough=True,
                load_all_layers=True,
            ),
            "targets": DataInput("vector", ["vector_layer"]),
        },
    )

    assert len(dataset) == 1
    inputs, _, _ = dataset[0]
    assert inputs["image"].image.shape == (1, 2, 4, 4)
    assert torch.all(inputs["image"].image[:, 0] == 1)
    assert torch.all(inputs["image"].image[:, 1] == 2)


def test_read_layer_time_range(tmp_path: UPath) -> None:
    """Test that time_range is correctly read from layer_data items.

    This test verifies that when items in layer_data have time_range set,
    the read_layer_time_range function correctly returns the min/max time
    range from all items.
    """
    ds_path = UPath(tmp_path)

    # Create dataset config with a raster layer
    dataset_config = {
        "layers": {
            "image": {
                "type": "raster",
                "band_sets": [
                    {
                        "dtype": "uint8",
                        "bands": ["band"],
                    }
                ],
            },
        },
    }
    ds_path.mkdir(parents=True, exist_ok=True)
    import json

    with (ds_path / "config.json").open("w") as f:
        json.dump(dataset_config, f)

    dataset = Dataset(ds_path)

    # Create a window
    window = Window(
        storage=dataset.storage,
        name="test_window",
        group="default",
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 4, 4),
        time_range=(datetime(2024, 1, 1), datetime(2024, 1, 31)),
    )
    window.save()

    # Write raster data
    image = np.ones((1, 4, 4), dtype=np.uint8)
    raster_dir = window.get_raster_dir("image", ["band"])
    GeotiffRasterFormat().encode_raster(
        raster_dir, window.projection, window.bounds, image
    )
    window.mark_layer_completed("image")

    # Create layer data with items that have time_range set
    item1_time_range = (datetime(2024, 1, 5), datetime(2024, 1, 10))
    item2_time_range = (datetime(2024, 1, 15), datetime(2024, 1, 20))

    item1 = Item(
        "item1",
        STGeometry(
            WGS84_PROJECTION,
            shapely.box(*window.bounds),
            item1_time_range,
        ),
    )
    item2 = Item(
        "item2",
        STGeometry(
            WGS84_PROJECTION,
            shapely.box(*window.bounds),
            item2_time_range,
        ),
    )

    layer_data = WindowLayerData(
        "image",
        serialized_item_groups=[[item1.serialize(), item2.serialize()]],
    )

    # Call the function that reads time ranges from layer data
    time_range = read_layer_time_range(layer_data, group_idx=0)

    # Verify the time_range is correct (min of starts, max of ends)
    assert time_range is not None
    assert time_range[0] == datetime(2024, 1, 5)  # min of item1 and item2 start
    assert time_range[1] == datetime(2024, 1, 20)  # max of item1 and item2 end


def test_model_dataset_index_uses_cache(
    basic_classification_dataset: Dataset,
    add_window_to_basic_classification_dataset: Callable,
) -> None:
    """Test that index_mode=USE actually uses cached results.

    Creates an index, then adds a new window. With USE mode, the cached
    index should be returned (not including the new window).
    """
    image = np.zeros((1, 4, 4), dtype=np.uint8)
    add_window_to_basic_classification_dataset(
        basic_classification_dataset,
        name="window1",
        images={("image_layer1", 0): image},
    )
    add_window_to_basic_classification_dataset(
        basic_classification_dataset,
        name="window2",
        images={("image_layer1", 0): image},
    )

    inputs = {
        "image": DataInput("raster", ["image_layer1"], bands=["band"]),
        "targets": DataInput("vector", ["vector_layer"]),
    }
    task = ClassificationTask("label", ["cls0", "cls1"], read_class_id=True)
    split_config = SplitConfig()

    # First run: create index
    dataset1 = ModelDataset(
        basic_classification_dataset,
        split_config=split_config,
        task=task,
        workers=0,
        inputs=inputs,
        index_mode=IndexMode.USE,
    )
    assert len(dataset1) == 2

    # Add a new window AFTER the index was created
    add_window_to_basic_classification_dataset(
        basic_classification_dataset,
        name="window3",
        images={("image_layer1", 0): image},
    )

    # Second run: should still return 2 windows (proving cache is used)
    dataset2 = ModelDataset(
        basic_classification_dataset,
        split_config=split_config,
        task=task,
        workers=0,
        inputs=inputs,
        index_mode=IndexMode.USE,
    )
    assert len(dataset2) == 2  # Still 2, not 3


def test_model_dataset_index_refresh_rebuilds(
    basic_classification_dataset: Dataset,
    add_window_to_basic_classification_dataset: Callable,
) -> None:
    """Test that index_mode=REFRESH rebuilds the index.

    Creates an index, adds a new window, then uses REFRESH mode.
    The refreshed index should include the new window.
    """
    image = np.zeros((1, 4, 4), dtype=np.uint8)
    add_window_to_basic_classification_dataset(
        basic_classification_dataset,
        name="window1",
        images={("image_layer1", 0): image},
    )
    add_window_to_basic_classification_dataset(
        basic_classification_dataset,
        name="window2",
        images={("image_layer1", 0): image},
    )

    inputs = {
        "image": DataInput("raster", ["image_layer1"], bands=["band"]),
        "targets": DataInput("vector", ["vector_layer"]),
    }
    task = ClassificationTask("label", ["cls0", "cls1"], read_class_id=True)
    split_config = SplitConfig()

    # First run: create index
    dataset1 = ModelDataset(
        basic_classification_dataset,
        split_config=split_config,
        task=task,
        workers=0,
        inputs=inputs,
        index_mode=IndexMode.USE,
    )
    assert len(dataset1) == 2

    # Add a new window AFTER the index was created
    add_window_to_basic_classification_dataset(
        basic_classification_dataset,
        name="window3",
        images={("image_layer1", 0): image},
    )

    # Refresh: should now include window3
    dataset2 = ModelDataset(
        basic_classification_dataset,
        split_config=split_config,
        task=task,
        workers=0,
        inputs=inputs,
        index_mode=IndexMode.REFRESH,
    )
    assert len(dataset2) == 3  # Now 3, because we refreshed the index


def test_model_dataset_without_index(
    basic_classification_dataset: Dataset,
    add_window_to_basic_classification_dataset: Callable,
) -> None:
    """Test that ModelDataset works correctly with index_mode=OFF (default)."""
    image = np.zeros((1, 4, 4), dtype=np.uint8)
    add_window_to_basic_classification_dataset(
        basic_classification_dataset,
        images={("image_layer1", 0): image},
    )

    # With index_mode=OFF (default), no index should be created
    dataset = ModelDataset(
        basic_classification_dataset,
        split_config=SplitConfig(),
        task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
        workers=0,
        inputs={
            "image": DataInput("raster", ["image_layer1"], bands=["band"]),
            "targets": DataInput("vector", ["vector_layer"]),
        },
        index_mode=IndexMode.OFF,
    )
    assert len(dataset) == 1

    # Verify no index directory was created
    index_dir = basic_classification_dataset.path / INDEX_DIR_NAME
    assert not index_dir.exists()


def test_skip_if_output_layer_exists(
    basic_classification_dataset: Dataset,
    add_window_to_basic_classification_dataset: Callable,
) -> None:
    """Test that windows with existing output layers are skipped when configured."""
    # Create two windows with images
    image = np.zeros((1, 4, 4), dtype=np.uint8)

    # First window - will have the output layer already completed
    window1 = add_window_to_basic_classification_dataset(
        basic_classification_dataset,
        images={
            ("image_layer1", 0): image,
        },
        window_name="window_with_output",
    )

    # Second window - will NOT have the output layer
    add_window_to_basic_classification_dataset(
        basic_classification_dataset,
        images={
            ("image_layer1", 0): image,
        },
        window_name="window_without_output",
    )

    # Mark the first window as having the output layer completed
    # Ensure the output layer directory exists before marking completed.
    layer_dir = window1.get_layer_dir("predictions")
    layer_dir.mkdir(parents=True, exist_ok=True)
    window1.mark_layer_completed("predictions")

    dataset = ModelDataset(
        basic_classification_dataset,
        split_config=SplitConfig(
            output_layer_name_skip_inference_if_exists="predictions",
        ),
        task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
        workers=1,
        inputs={
            "image": DataInput("raster", ["image_layer1"], bands=["band"]),
            "targets": DataInput("vector", ["vector_layer"]),
        },
    )

    assert len(dataset) == 1
    windows = dataset.get_dataset_examples()
    assert windows[0].name == "window_without_output"

    # Test 3: Without setting output_layer_name_skip_inference_if_exists, should get both windows
    dataset = ModelDataset(
        basic_classification_dataset,
        split_config=SplitConfig(),
        task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
        workers=1,
        inputs={
            "image": DataInput("raster", ["image_layer1"], bands=["band"]),
            "targets": DataInput("vector", ["vector_layer"]),
        },
    )

    assert len(dataset) == 2


def test_non_required_layer_missing(
    basic_classification_dataset: Dataset,
    add_window_to_basic_classification_dataset: Callable,
) -> None:
    """Test that windows with missing non-required layers are still loaded.

    When a DataInput has required=False, windows where that layer is missing
    should still be included in the dataset, and reading from those windows
    should skip the missing input without raising an error.
    """
    image = np.zeros((1, 4, 4), dtype=np.uint8)

    # Window 1: has both image_layer1 and image_layer2
    add_window_to_basic_classification_dataset(
        basic_classification_dataset,
        name="window_with_both",
        images={
            ("image_layer1", 0): image,
            ("image_layer2", 0): image,
        },
    )

    # Window 2: has only image_layer1 (image_layer2 is missing)
    add_window_to_basic_classification_dataset(
        basic_classification_dataset,
        name="window_with_only_layer1",
        images={
            ("image_layer1", 0): image,
            # image_layer2 is intentionally missing
        },
    )

    # Create dataset with image_layer2 as non-required
    dataset = ModelDataset(
        basic_classification_dataset,
        split_config=SplitConfig(),
        task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
        workers=1,
        inputs={
            "image1": DataInput(
                "raster",
                ["image_layer1"],
                bands=["band"],
                passthrough=True,
                required=True,
            ),
            "image2": DataInput(
                "raster",
                ["image_layer2"],
                bands=["band"],
                passthrough=True,
                required=False,  # This layer is optional
            ),
            "targets": DataInput("vector", ["vector_layer"]),
        },
    )

    # Both windows should be included (non-required layer doesn't filter)
    assert len(dataset) == 2

    # Reading from both windows should work
    for idx in range(2):
        inputs, _, metadata = dataset[idx]
        # image1 should always be present
        assert "image1" in inputs

        # image2 may or may not be present depending on the window
        if metadata.window_name == "window_with_both":
            assert "image2" in inputs
        else:
            # For window_with_only_layer1, image2 should be skipped
            assert "image2" not in inputs


class TestSplitConfig:
    """Tests for SplitConfig."""

    def test_overlap_ratio_with_patch_size_in_separate_configs(self) -> None:
        """Test that overlap_ratio works when patch_size is set in a different config.

        This test simulates the user setting patch_size in the default config, and
        overlap_ratio in the predict config (which is merged via merge_and_validate).
        """
        default_config = SplitConfig(patch_size=128, load_all_crops=True)
        predict_config = SplitConfig(overlap_ratio=0.5)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            merged = SplitConfig.merge_and_validate([default_config, predict_config])

        # get_overlap_pixels should compute correctly: 128 * 0.5 = 64
        assert merged.get_overlap_pixels() == 64

    def test_overlap_ratio_without_crop_size_raises_on_get(self) -> None:
        """Test that overlap_ratio without crop_size raises error in get_overlap_pixels."""
        config = SplitConfig(overlap_ratio=0.5)

        # Should raise when trying to get overlap_pixels
        with pytest.raises(ValueError, match="overlap_ratio requires crop_size"):
            config.get_overlap_pixels()

    def test_crop_size_and_patch_size_in_separate_configs_raises(self) -> None:
        """Test that setting crop_size and patch_size in different configs raises error."""
        config1 = SplitConfig(crop_size=128)
        config2 = SplitConfig(patch_size=256)

        with pytest.raises(
            ValueError, match="Cannot specify both crop_size and patch_size"
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                SplitConfig.merge_and_validate([config1, config2])

    def test_negative_overlap_pixels_raises(self) -> None:
        """Test that negative overlap_pixels raises error."""
        config = SplitConfig(crop_size=128, load_all_crops=True, overlap_pixels=-1)

        with pytest.raises(ValueError, match="overlap_pixels must be non-negative"):
            SplitConfig.merge_and_validate([config])


def test_compute_expected_timestamps_per_period_mosaic() -> None:
    """Test compute_expected_timestamps for PER_PERIOD_MOSAIC mode.

    Should compute expected timestamps based on window time_range, period_duration,
    and max_matches from the query config.
    """
    # Create a mock window with a 4-month time range
    mock_storage = MagicMock()
    window = Window(
        storage=mock_storage,
        group="test",
        name="test_window",
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 100, 100),
        time_range=(
            datetime(2025, 1, 1),
            datetime(2025, 5, 1),
        ),  # Jan 1 - May 1 (4 months)
    )

    # Create layer config with PER_PERIOD_MOSAIC mode, 30-day periods, max 4 matches
    layer_config = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.UINT8, bands=["band"])],
        data_source=DataSourceConfig(
            class_path="rslearn.data_sources.sentinel2.Sentinel2",
            query_config=QueryConfig(
                space_mode=SpaceMode.PER_PERIOD_MOSAIC,
                period_duration=timedelta(days=30),
                max_matches=4,
            ),
        ),
    )

    expected_ts = compute_expected_timestamps(window, layer_config)

    assert expected_ts is not None
    assert len(expected_ts) == 4  # 4 periods

    # Timestamps should be in chronological order (oldest first)
    for i in range(len(expected_ts) - 1):
        assert expected_ts[i][0] < expected_ts[i + 1][0]
    assert expected_ts[-1][1] == datetime(2025, 5, 1)

    # Each period should be 30 days
    for start, end in expected_ts:
        assert (end - start).days == 30


def test_compute_expected_timestamps_with_time_offset() -> None:
    """Test compute_expected_timestamps applies time_offset correctly."""
    mock_storage = MagicMock()
    window = Window(
        storage=mock_storage,
        group="test",
        name="test_window",
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 100, 100),
        time_range=(datetime(2025, 1, 1), datetime(2025, 4, 1)),
    )

    # Layer config with a 30-day time offset
    layer_config = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.UINT8, bands=["band"])],
        data_source=DataSourceConfig(
            class_path="rslearn.data_sources.sentinel2.Sentinel2",
            query_config=QueryConfig(
                space_mode=SpaceMode.PER_PERIOD_MOSAIC,
                period_duration=timedelta(days=30),
                max_matches=3,
            ),
            time_offset=timedelta(days=30),  # Shift 30 days into future
        ),
    )

    expected_ts = compute_expected_timestamps(window, layer_config)

    assert expected_ts is not None
    # First timestamp should be after Jan 1 + 30 days offset
    assert expected_ts[0][0] >= datetime(2025, 1, 31)
    assert expected_ts[0][0] <= datetime(2025, 2, 1)


def test_compute_expected_timestamps_with_duration_override() -> None:
    """Test compute_expected_timestamps applies duration override correctly."""
    mock_storage = MagicMock()
    window = Window(
        storage=mock_storage,
        group="test",
        name="test_window",
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 100, 100),
        time_range=(datetime(2025, 1, 1), datetime(2025, 12, 31)),  # Full year
    )

    # Override duration to only 60 days from start
    layer_config = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.UINT8, bands=["band"])],
        data_source=DataSourceConfig(
            class_path="rslearn.data_sources.sentinel2.Sentinel2",
            query_config=QueryConfig(
                space_mode=SpaceMode.PER_PERIOD_MOSAIC,
                period_duration=timedelta(days=30),
                max_matches=12,
            ),
            duration=timedelta(days=60),  # Only use 60 days
        ),
    )

    expected_ts = compute_expected_timestamps(window, layer_config)

    assert expected_ts is not None
    # With 60-day duration and 30-day periods, should only get 2 periods
    assert len(expected_ts) == 2


def test_compute_expected_timestamps_no_time_range() -> None:
    """Test compute_expected_timestamps returns None when window has no time_range."""
    mock_storage = MagicMock()
    window = Window(
        storage=mock_storage,
        group="test",
        name="test_window",
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 100, 100),
        time_range=None,  # No time range
    )

    layer_config = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.UINT8, bands=["band"])],
        data_source=DataSourceConfig(
            class_path="rslearn.data_sources.sentinel2.Sentinel2",
            query_config=QueryConfig(
                space_mode=SpaceMode.PER_PERIOD_MOSAIC,
                period_duration=timedelta(days=30),
                max_matches=4,
            ),
        ),
    )

    expected_ts = compute_expected_timestamps(window, layer_config)
    assert expected_ts is None


def test_compute_expected_timestamps_no_data_source() -> None:
    """Test compute_expected_timestamps returns None when layer has no data_source."""
    mock_storage = MagicMock()
    window = Window(
        storage=mock_storage,
        group="test",
        name="test_window",
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 100, 100),
        time_range=(datetime(2025, 1, 1), datetime(2025, 4, 1)),
    )

    # Layer config without data_source
    layer_config = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.UINT8, bands=["band"])],
        data_source=None,
    )

    expected_ts = compute_expected_timestamps(window, layer_config)
    assert expected_ts is None


def test_compute_expected_timestamps_single_timestep() -> None:
    """Test compute_expected_timestamps for single-timestep (max_matches=1) mode."""
    mock_storage = MagicMock()
    window = Window(
        storage=mock_storage,
        group="test",
        name="test_window",
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 100, 100),
        time_range=(datetime(2025, 1, 1), datetime(2025, 2, 1)),
    )

    # MOSAIC mode with max_matches=1 (default)
    layer_config = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.UINT8, bands=["band"])],
        data_source=DataSourceConfig(
            class_path="rslearn.data_sources.sentinel2.Sentinel2",
            query_config=QueryConfig(
                space_mode=SpaceMode.MOSAIC,
                max_matches=1,
            ),
        ),
    )

    expected_ts = compute_expected_timestamps(window, layer_config)

    assert expected_ts is not None
    assert len(expected_ts) == 1
    assert expected_ts[0] == (datetime(2025, 1, 1), datetime(2025, 2, 1))
