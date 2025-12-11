"""Unit tests for rslearn.train.dataset."""

import numpy as np
import pytest
import torch.utils.data

from rslearn.dataset import Dataset
from rslearn.train.dataset import (
    DataInput,
    ModelDataset,
    RetryDataset,
    SplitConfig,
)
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.train.transforms.concatenate import Concatenate
from tests.unit.train.conftest import add_window


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


def test_basic_time_series(basic_classification_dataset: Dataset) -> None:
    # Create a window with two images in the first layer to make sure we will be able
    # to load it when explicitly adding a DataInput for it.
    image = np.zeros((1, 4, 4), dtype=np.uint8)
    add_window(
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
    assert inputs["image"].shape == (2, 4, 4)


def test_load_all_layers(basic_classification_dataset: Dataset) -> None:
    """Make sure we can load a time series by using load_all_layers option."""
    # Create a window with two images in the first layer to make sure we will be able
    # to load it when explicitly adding a DataInput for it.
    image = np.zeros((1, 4, 4), dtype=np.uint8)
    add_window(
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
    assert inputs["image"].shape == (2, 4, 4)


def test_load_two_layers(basic_classification_dataset: Dataset) -> None:
    """Make sure when load_all_layers is passed we load all of the layer options."""
    # We create a window with two images in the first layer and one image in the second
    # layer. Then in the DataInput we only refer to the second image in the first layer
    # and the only image in the second layer. With load_all_layers but not
    # load_all_item_groups, just these two images should be read.
    add_window(
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
    assert inputs["image"].shape == (2, 4, 4)
    assert torch.all(inputs["image"][0] == 1)
    assert torch.all(inputs["image"][1] == 2)
