import json
import pathlib

import numpy as np
import numpy.typing as npt
import pytest
import shapely
import torch.utils.data
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.train.dataset import DataInput, ModelDataset, RetryDataset, SplitConfig
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.train.transforms.concatenate import Concatenate
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import STGeometry
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat


class TestException(Exception):
    pass


@pytest.fixture
def basic_classification_dataset(tmp_path: pathlib.Path) -> Dataset:
    """Create an empty dataset setup for image classification."""
    ds_path = UPath(tmp_path)
    dataset_config = {
        "layers": {
            "image_layer1": {
                "type": "raster",
                "band_sets": [
                    {
                        "dtype": "uint8",
                        "bands": ["band"],
                    }
                ],
            },
            "image_layer2": {
                "type": "raster",
                "band_sets": [
                    {
                        "dtype": "uint8",
                        "bands": ["band"],
                    }
                ],
            },
            "vector_layer": {"type": "vector"},
        },
    }
    ds_path.mkdir(parents=True, exist_ok=True)
    with (ds_path / "config.json").open("w") as f:
        json.dump(dataset_config, f)
    return Dataset(ds_path)


def add_window(
    dataset: Dataset,
    name: str = "default",
    group: str = "default",
    images: dict[tuple[str, int], npt.NDArray] = {},
) -> Window:
    """Add a window to the dataset.

    Args:
        dataset: the dataset to add to.
        name: the name of the window.
        group: the group of the window.
        images: map from (layer_name, group_idx) to the image content, which should be
            1x4x4 since that is the window size.
    """
    window_path = Window.get_window_root(dataset.path, name, group)
    window = Window(
        path=window_path,
        group=name,
        name=group,
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 4, 4),
        time_range=None,
    )
    window.save()

    for (layer_name, group_idx), image in images.items():
        raster_dir = window.get_raster_dir(layer_name, ["band"], group_idx=group_idx)
        GeotiffRasterFormat().encode_raster(
            raster_dir, window.projection, window.bounds, image
        )
        window.mark_layer_completed(layer_name, group_idx=group_idx)

    # Add label.
    feature = Feature(
        STGeometry(window.projection, shapely.Point(1, 1), None),
        {
            "label": 1,
        },
    )
    layer_dir = window.get_layer_dir("vector_layer")
    GeojsonVectorFormat().encode_vector(
        layer_dir,
        [feature],
    )
    window.mark_layer_completed("vector_layer")

    return window


class TestDataset(torch.utils.data.Dataset):
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
    dataset = TestDataset(failures=3)
    dataset = RetryDataset(dataset, retries=3, delay=0.01)
    for _ in dataset:
        pass

    # Now try with 4 failures, it should fail.
    dataset = TestDataset(failures=4)
    dataset = RetryDataset(dataset, retries=3, delay=0.01)
    with pytest.raises(TestException):
        for _ in dataset:
            pass


def test_dataset_covers_border(image_to_class_dataset: Dataset) -> None:
    # Make sure that, when loading all patches, the border of the raster is included in
    # the generated windows.
    # The image_to_class_dataset window is 4x4 so 3x3 patch will ensure irregular window
    # at the border.
    split_config = SplitConfig(
        patch_size=3,
        load_all_patches=True,
    )
    image_data_input = DataInput("raster", ["image"], bands=["band"], passthrough=True)
    target_data_input = DataInput("vector", ["label"])
    task = ClassificationTask("label", ["cls0", "cls1"], read_class_id=True)
    dataset = ModelDataset(
        image_to_class_dataset,
        split_config=split_config,
        task=task,
        workers=1,
        inputs={
            "image": image_data_input,
            "targets": target_data_input,
        },
    )

    point_coverage = {}
    for col in range(4):
        for row in range(4):
            point_coverage[(col, row)] = False

    # There should be 4 windows with top-left at:
    # - (0, 0)
    # - (0, 3)
    # - (3, 0)
    # - (3, 3)
    assert len(dataset) == 4

    for _, _, metadata in dataset:
        bounds = metadata["bounds"]
        for col, row in list(point_coverage.keys()):
            if col < bounds[0] or col >= bounds[2]:
                continue
            if row < bounds[1] or row >= bounds[3]:
                continue
            point_coverage[(col, row)] = True

    assert all(point_coverage.values())

    # Test with overlap_ratio=0.5 for 2x2 patches
    split_config_with_overlap = SplitConfig(
        patch_size=2,
        load_all_patches=True,
        overlap_ratio=0.5,
    )
    dataset_with_overlap = ModelDataset(
        image_to_class_dataset,
        split_config=split_config_with_overlap,
        task=task,
        workers=1,
        inputs={
            "image": image_data_input,
            "targets": target_data_input,
        },
    )

    point_coverage = {}
    for col in range(4):
        for row in range(4):
            point_coverage[(col, row)] = False

    # With overlap_ratio=0.5, there should be 16 windows given that overlap is 1 pixel.
    assert len(dataset_with_overlap) == 16

    for _, _, metadata in dataset:
        bounds = metadata["bounds"]

        for col, row in list(point_coverage.keys()):
            if col < bounds[0] or col >= bounds[2]:
                continue
            if row < bounds[1] or row >= bounds[3]:
                continue
            point_coverage[(col, row)] = True

    assert all(point_coverage.values())


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
            ("image_layer2", 1): 2 * np.ones((1, 4, 4), dtype=np.uint8),
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
    assert np.all(inputs["image"][0] == 1)
    assert np.all(inputs["image"][1] == 2)
