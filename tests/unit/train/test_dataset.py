"""Unit tests for rslearn.train.dataset."""

import json
import os
from collections.abc import Callable
from datetime import datetime

import numpy as np
import pytest
import shapely
import torch
import torch.utils.data
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import Item
from rslearn.dataset import Dataset, Window
from rslearn.dataset.window import WindowLayerData
from rslearn.train.dataset import (
    DataInput,
    ModelDataset,
    RetryDataset,
    SplitConfig,
    _compute_cache_key,
    read_layer_time_range,
)
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


def test_compute_cache_key() -> None:
    """Test that cache keys are computed correctly based on inputs and split config."""
    inputs1 = {
        "image": DataInput("raster", ["image_layer1"], bands=["band"]),
        "targets": DataInput("vector", ["vector_layer"], is_target=True),
    }
    inputs2 = {
        "image": DataInput("raster", ["image_layer1"], bands=["band"]),
        "targets": DataInput("vector", ["vector_layer"], is_target=True),
    }
    inputs3 = {
        "image": DataInput("raster", ["image_layer2"], bands=["band"]),
        "targets": DataInput("vector", ["vector_layer"], is_target=True),
    }

    split_config1 = SplitConfig(groups=["train"])
    split_config2 = SplitConfig(groups=["train"])
    split_config3 = SplitConfig(groups=["val"])

    # Same inputs and split config should produce the same key.
    key1 = _compute_cache_key(inputs1, split_config1)
    key2 = _compute_cache_key(inputs2, split_config2)
    assert key1 == key2

    # Different layer in inputs should produce different key.
    key3 = _compute_cache_key(inputs3, split_config1)
    assert key1 != key3

    # Different split config should produce different key.
    key4 = _compute_cache_key(inputs1, split_config3)
    assert key1 != key4


def test_dataset_index_caching(
    basic_classification_dataset: Dataset,
    add_window_to_basic_classification_dataset: Callable,
    tmp_path: UPath,
) -> None:
    """Test that dataset index caching works correctly."""
    image = np.zeros((1, 4, 4), dtype=np.uint8)
    add_window_to_basic_classification_dataset(
        basic_classification_dataset,
        images={
            ("image_layer1", 0): image,
        },
    )

    cache_path = str(tmp_path / "cache" / "dataset_index.json")
    inputs = {
        "image": DataInput("raster", ["image_layer1"], bands=["band"], passthrough=True),
        "targets": DataInput("vector", ["vector_layer"]),
    }

    # First run: cache should be created.
    dataset1 = ModelDataset(
        basic_classification_dataset,
        split_config=SplitConfig(),
        task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
        workers=0,
        inputs=inputs,
        cache_index_path=cache_path,
    )
    assert len(dataset1) == 1

    # Verify the cache file was created.
    assert os.path.exists(cache_path)
    with open(cache_path) as f:
        cache_data = json.load(f)
    assert "cache_key" in cache_data
    assert "windows" in cache_data
    assert len(cache_data["windows"]) == 1

    # Second run: should load from cache.
    dataset2 = ModelDataset(
        basic_classification_dataset,
        split_config=SplitConfig(),
        task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
        workers=0,
        inputs=inputs,
        cache_index_path=cache_path,
    )
    assert len(dataset2) == 1


def test_dataset_index_cache_invalidation(
    basic_classification_dataset: Dataset,
    add_window_to_basic_classification_dataset: Callable,
    tmp_path: UPath,
) -> None:
    """Test that cache is invalidated when inputs change."""
    image = np.zeros((1, 4, 4), dtype=np.uint8)
    add_window_to_basic_classification_dataset(
        basic_classification_dataset,
        images={
            ("image_layer1", 0): image,
            ("image_layer2", 0): image,
        },
    )

    cache_path = str(tmp_path / "cache" / "dataset_index.json")
    inputs1 = {
        "image": DataInput("raster", ["image_layer1"], bands=["band"], passthrough=True),
        "targets": DataInput("vector", ["vector_layer"]),
    }

    # First run: cache should be created with inputs1.
    dataset1 = ModelDataset(
        basic_classification_dataset,
        split_config=SplitConfig(),
        task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
        workers=0,
        inputs=inputs1,
        cache_index_path=cache_path,
    )
    assert len(dataset1) == 1

    # Get the original cache key.
    with open(cache_path) as f:
        original_cache_key = json.load(f)["cache_key"]

    # Second run with different inputs: cache should be invalidated and recreated.
    inputs2 = {
        "image": DataInput("raster", ["image_layer2"], bands=["band"], passthrough=True),
        "targets": DataInput("vector", ["vector_layer"]),
    }
    dataset2 = ModelDataset(
        basic_classification_dataset,
        split_config=SplitConfig(),
        task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
        workers=0,
        inputs=inputs2,
        cache_index_path=cache_path,
    )
    assert len(dataset2) == 1

    # Verify the cache key was updated.
    with open(cache_path) as f:
        new_cache_key = json.load(f)["cache_key"]
    assert original_cache_key != new_cache_key


def test_dataset_without_cache(
    basic_classification_dataset: Dataset,
    add_window_to_basic_classification_dataset: Callable,
) -> None:
    """Test that dataset works correctly without caching (cache_index_path=None)."""
    image = np.zeros((1, 4, 4), dtype=np.uint8)
    add_window_to_basic_classification_dataset(
        basic_classification_dataset,
        images={
            ("image_layer1", 0): image,
        },
    )

    inputs = {
        "image": DataInput("raster", ["image_layer1"], bands=["band"], passthrough=True),
        "targets": DataInput("vector", ["vector_layer"]),
    }

    # Run without cache_index_path - should work normally.
    dataset = ModelDataset(
        basic_classification_dataset,
        split_config=SplitConfig(),
        task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
        workers=0,
        inputs=inputs,
        cache_index_path=None,
    )
    assert len(dataset) == 1
