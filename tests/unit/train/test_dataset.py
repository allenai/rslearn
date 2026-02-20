"""Unit tests for rslearn.train.dataset."""

import json
import random
import warnings
from collections.abc import Callable
from datetime import datetime

import numpy as np
import pytest
import torch
import torch.utils.data
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.train.dataset import (
    DataInput,
    IndexMode,
    ModelDataset,
    RetryDataset,
    SplitConfig,
    check_window,
    read_data_input,
)
from rslearn.train.dataset_index import INDEX_DIR_NAME
from rslearn.train.model_context import RasterImage
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.train.transforms.concatenate import Concatenate
from rslearn.utils.raster_array import RasterArray
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


def test_read_data_input_timestamps(tmp_path: UPath) -> None:
    """Test that read_data_input reads timestamps from RasterArrays and stacks them.

    Creates two item groups for the same layer, each with a distinct timestamp.
    With load_all_layers + load_all_item_groups, both should be read and the
    timestamps should be concatenated in the returned RasterImage.
    """
    ds_path = UPath(tmp_path)
    ds_path.mkdir(parents=True, exist_ok=True)

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
    with (ds_path / "config.json").open("w") as f:
        json.dump(dataset_config, f)

    dataset = Dataset(ds_path)

    window = Window(
        storage=dataset.storage,
        name="test_window",
        group="default",
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 4, 4),
        time_range=None,
    )
    window.save()

    ts1 = (datetime(2024, 1, 5), datetime(2024, 1, 10))
    ts2 = (datetime(2024, 1, 15), datetime(2024, 1, 20))

    image1 = np.ones((1, 4, 4), dtype=np.uint8)
    raster_dir1 = window.get_raster_dir("image", ["band"], group_idx=0)
    GeotiffRasterFormat().encode_raster(
        raster_dir1,
        window.projection,
        window.bounds,
        RasterArray(chw_array=image1, time_range=ts1),
    )

    image2 = 2 * np.ones((1, 4, 4), dtype=np.uint8)
    raster_dir2 = window.get_raster_dir("image", ["band"], group_idx=1)
    GeotiffRasterFormat().encode_raster(
        raster_dir2,
        window.projection,
        window.bounds,
        RasterArray(chw_array=image2, time_range=ts2),
    )

    window.mark_layer_completed("image", group_idx=0)
    window.mark_layer_completed("image", group_idx=1)

    data_input = DataInput(
        "raster",
        ["image"],
        bands=["band"],
        load_all_layers=True,
        load_all_item_groups=True,
    )

    result = read_data_input(
        dataset, window, window.bounds, data_input, random.Random(0)
    )

    assert isinstance(result, RasterImage)
    assert result.image.shape == (1, 2, 4, 4)
    # image1 was 1 everywhere.
    assert torch.all(result.image[:, 0] == 1)
    # image2 was 2 everywhere, it is second item group so should get stacked to the
    # second timestep.
    assert torch.all(result.image[:, 1] == 2)
    # RasterArray should have the stacked timestamps as well.
    assert result.timestamps == [ts1, ts2]


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


class TestCheckWindow:
    """Tests for check_window and CheckWindowResult."""

    def test_passes_when_all_inputs_available(
        self,
        basic_classification_dataset: Dataset,
        add_window_to_basic_classification_dataset: Callable,
    ) -> None:
        """Window is returned with empty result when all required inputs are present."""
        image = np.zeros((1, 4, 4), dtype=np.uint8)
        window = add_window_to_basic_classification_dataset(
            basic_classification_dataset,
            images={("image_layer1", 0): image},
        )
        inputs = {
            "image": DataInput("raster", ["image_layer1"], bands=["band"]),
            "targets": DataInput("vector", ["vector_layer"]),
        }
        result_window, result = check_window(inputs, window)
        assert result_window is window
        assert result.missing_data_input_counts == {}
        assert result.has_output_layer_count == 0

    def test_skipped_for_missing_required_input(
        self,
        basic_classification_dataset: Dataset,
        add_window_to_basic_classification_dataset: Callable,
    ) -> None:
        """Window is skipped and result reports the missing input key."""
        image = np.zeros((1, 4, 4), dtype=np.uint8)
        window = add_window_to_basic_classification_dataset(
            basic_classification_dataset,
            images={("image_layer1", 0): image},
        )
        inputs = {
            "image": DataInput("raster", ["image_layer1"], bands=["band"]),
            "missing_layer": DataInput(
                "raster", ["nonexistent_layer"], bands=["band"], required=True
            ),
        }
        result_window, result = check_window(inputs, window)
        assert result_window is None
        assert result.missing_data_input_counts == {"missing_layer": 1}
        assert result.has_output_layer_count == 0

    def test_skipped_for_existing_output_layer(
        self,
        basic_classification_dataset: Dataset,
        add_window_to_basic_classification_dataset: Callable,
    ) -> None:
        """Window is skipped and result reports existing output layer."""
        image = np.zeros((1, 4, 4), dtype=np.uint8)
        window = add_window_to_basic_classification_dataset(
            basic_classification_dataset,
            images={("image_layer1", 0): image},
        )
        # Mark an output layer as completed.
        layer_dir = window.get_layer_dir("predictions")
        layer_dir.mkdir(parents=True, exist_ok=True)
        window.mark_layer_completed("predictions")

        inputs = {
            "image": DataInput("raster", ["image_layer1"], bands=["band"]),
            "targets": DataInput("vector", ["vector_layer"]),
        }
        result_window, result = check_window(
            inputs,
            window,
            output_layer_name_skip_inference_if_exists="predictions",
        )
        assert result_window is None
        assert result.missing_data_input_counts == {}
        assert result.has_output_layer_count == 1
