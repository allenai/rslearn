"""Unit tests for rslearn.train.dataset."""

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
