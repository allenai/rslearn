"""Unit tests for rslearn.train.dataset."""

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
from rslearn.train.all_patches_dataset import (
    InMemoryAllPatchesDataset,
    IterableAllPatchesDataset,
)
from rslearn.train.dataset import (
    DataInput,
    ModelDataset,
    RetryDataset,
    SplitConfig,
)
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.train.tasks.segmentation import SegmentationTask
from rslearn.train.transforms.concatenate import Concatenate
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import PixelBounds, STGeometry, ResolutionFactor
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
    bounds: PixelBounds = (0, 0, 4, 4),
) -> Window:
    """Add a window to the dataset.

    Args:
        dataset: the dataset to add to.
        name: the name of the window.
        group: the group of the window.
        images: map from (layer_name, group_idx) to the image content, which should be
            1x4x4 since that is the window size.
    """
    window = Window(
        storage=dataset.storage,
        name=name,
        group=group,
        projection=WGS84_PROJECTION,
        bounds=bounds,
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
        STGeometry(window.projection, shapely.box(*bounds), None),
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


def test_dataset_covers_border(image_to_class_dataset: Dataset) -> None:
    # Make sure that, when loading all patches, the border of the raster is included in
    # the generated windows.
    # The image_to_class_dataset window is 4x4 so 3x3 patch will ensure irregular window
    # at the border.
    patch_size = 3
    split_config = SplitConfig(
        patch_size=patch_size,
        load_all_patches=True,
    )
    image_data_input = DataInput("raster", ["image"], bands=["band"], passthrough=True)
    target_data_input = DataInput("vector", ["label"])
    task = ClassificationTask("label", ["cls0", "cls1"], read_class_id=True)
    model_dataset = ModelDataset(
        image_to_class_dataset,
        split_config=split_config,
        task=task,
        workers=1,
        inputs={
            "image": image_data_input,
            "targets": target_data_input,
        },
    )
    dataset = IterableAllPatchesDataset(model_dataset, (patch_size, patch_size))

    point_coverage = {}
    for col in range(4):
        for row in range(4):
            point_coverage[(col, row)] = False

    # There should be 4 windows with top-left at:
    # - (0, 0)
    # - (0, 1)
    # - (1, 0)
    # - (1, 1)
    assert len(list(dataset)) == 4

    for _, _, metadata in dataset:
        bounds = metadata.patch_bounds
        for col, row in list(point_coverage.keys()):
            if col < bounds[0] or col >= bounds[2]:
                continue
            if row < bounds[1] or row >= bounds[3]:
                continue
            point_coverage[(col, row)] = True

    assert all(point_coverage.values())

    # Test with overlap_ratio=0.5 for 2x2 patches
    dataset_with_overlap = IterableAllPatchesDataset(
        model_dataset, (2, 2), overlap_ratio=0.5
    )

    point_coverage = {}
    for col in range(4):
        for row in range(4):
            point_coverage[(col, row)] = False

    # With overlap_ratio=0.5, there should be 9 windows given that overlap is 1 pixel.
    assert len(list(dataset_with_overlap)) == 9

    for _, _, metadata in dataset:
        bounds = metadata.patch_bounds

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


class TestIterableAllPatchesDataset:
    """Tests for IterableAllPatchesDataset."""

    def test_one_window_per_worker(self, basic_classification_dataset: Dataset) -> None:
        """Verify that things work with one window per worker."""
        add_window(basic_classification_dataset, name="window0")
        add_window(basic_classification_dataset, name="window1")
        add_window(basic_classification_dataset, name="window2")
        add_window(basic_classification_dataset, name="window3")
        model_dataset = ModelDataset(
            basic_classification_dataset,
            split_config=SplitConfig(),
            task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
            workers=1,
            inputs={
                "targets": DataInput("vector", ["vector_layer"]),
            },
        )
        world_size = 4
        window_names = set()
        for rank in range(world_size):
            all_patches_dataset = IterableAllPatchesDataset(
                model_dataset, (4, 4), rank=rank, world_size=world_size
            )
            samples = list(all_patches_dataset)
            assert len(samples) == 1
            window_names.add(samples[0][2].window_name)
        assert len(window_names) == 4

    def test_different_window_sizes(
        self, basic_classification_dataset: Dataset
    ) -> None:
        """Verify that rank padding works with different window sizes."""
        # One rank should get the second window.
        # While the other rank should get first window and needs to repeat it.
        add_window(basic_classification_dataset, name="window0", bounds=(0, 0, 4, 4))
        add_window(basic_classification_dataset, name="window1", bounds=(0, 0, 8, 8))
        model_dataset = ModelDataset(
            basic_classification_dataset,
            split_config=SplitConfig(),
            task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
            workers=1,
            inputs={
                "targets": DataInput("vector", ["vector_layer"]),
            },
        )
        world_size = 2
        seen_patches: dict[tuple[str, PixelBounds], int] = {}
        for rank in range(world_size):
            all_patches_dataset = IterableAllPatchesDataset(
                model_dataset, (4, 4), rank=rank, world_size=world_size
            )
            samples = list(all_patches_dataset)
            assert len(samples) == 4
            for sample in samples:
                patch_id = (sample[2].window_name, sample[2].patch_bounds)
                seen_patches[patch_id] = seen_patches.get(patch_id, 0) + 1

        assert len(seen_patches) == 5
        assert seen_patches[("window0", (0, 0, 4, 4))] == 4
        assert seen_patches[("window1", (0, 0, 4, 4))] == 1
        assert seen_patches[("window1", (0, 4, 4, 8))] == 1
        assert seen_patches[("window1", (4, 0, 8, 4))] == 1
        assert seen_patches[("window1", (4, 4, 8, 8))] == 1

    def test_empty_dataset(self, basic_classification_dataset: Dataset) -> None:
        """Verify that IterableAllPatchesDataset works with no windows."""
        model_dataset = ModelDataset(
            basic_classification_dataset,
            split_config=SplitConfig(),
            task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
            workers=1,
            inputs={
                "targets": DataInput("vector", ["vector_layer"]),
            },
        )
        world_size = 2
        for rank in range(world_size):
            all_patches_dataset = IterableAllPatchesDataset(
                model_dataset, (4, 4), rank=rank, world_size=world_size
            )
            samples = list(all_patches_dataset)
            assert len(samples) == 0


    def test_resolution_factor_cropping(self, basic_classification_dataset: Dataset) -> None:
        """Verify that cropping works correctly with different resolution factors."""

        add_window(
            basic_classification_dataset,
            name="window0",
            bounds=(0, 0, 8, 8),
            images={
                ("image_layer1", 0): np.ones((1, 8, 8), dtype=np.uint8),
                ("image_layer2", 0): np.ones((1, 4, 4), dtype=np.uint8),
            },
        )
        
        image_data_input = DataInput("raster", ["image_layer1"], bands=["band"], passthrough=True)
        target_data_input = DataInput(
            "raster", 
            ["image_layer2"], 
            bands=["band"],
            resolution_factor=ResolutionFactor(numerator=1, denominator=2),  # 1/2 resolution
            is_target=True,
        )
        
        model_dataset = ModelDataset(
            basic_classification_dataset,
            split_config=SplitConfig(patch_size=4, load_all_patches=True),
            task=SegmentationTask(num_classes=2),
            workers=1,
            inputs={"image": image_data_input, "targets": target_data_input},
        )
        
        dataset = IterableAllPatchesDataset(
            model_dataset, (4, 4), rank=0, world_size=1
        )
        
        # Verify we actually have samples to test
        sample_count = 0
        for inputs, targets, metadata in dataset:
            sample_count += 1
            assert inputs["image"].shape[-2:] == (4, 4)
            assert targets["classes"].shape == (2, 2)
        assert sample_count > 0, "No samples were generated - test is not valid"


class TestInMemoryAllPatchesDataset:
    """Tests for InMemoryAllPatchesDataset."""

    def test_iterable_equal(self, basic_classification_dataset: Dataset) -> None:
        """Verify that InMemoryAllPatchesDataset and IterableAllPatchesDataset are equivalent."""
        # Create a couple of windows with different sizes to exercise patching.
        add_window(basic_classification_dataset, name="w0", bounds=(0, 0, 4, 4))
        add_window(basic_classification_dataset, name="w1", bounds=(0, 0, 8, 8))

        # Build a minimal ModelDataset (only targets needed for this comparison).
        model_dataset = ModelDataset(
            basic_classification_dataset,
            split_config=SplitConfig(),
            task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
            workers=1,
            inputs={
                "targets": DataInput("vector", ["vector_layer"]),
            },
        )

        # Construct iterable and regular versions.
        patch_size = (3, 3)
        iterable_ds = IterableAllPatchesDataset(
            model_dataset, patch_size, rank=0, world_size=1
        )
        regular_ds = InMemoryAllPatchesDataset(model_dataset, patch_size)

        iterable_samples = list(iterable_ds)
        regular_samples = [regular_ds[i] for i in range(len(regular_ds))]

        # Compare metadata (last element of each tuple) index-by-index.
        assert len(iterable_samples) == len(regular_samples)
        for i in range(len(iterable_samples)):
            assert iterable_samples[i][-1] == regular_samples[i][-1]

    def test_resolution_factor_cropping(self, basic_classification_dataset: Dataset) -> None:
        """Verify that cropping works correctly with different resolution factors."""

        add_window(
            basic_classification_dataset,
            name="window0",
            bounds=(0, 0, 8, 8),
            images={
                ("image_layer1", 0): np.ones((1, 8, 8), dtype=np.uint8),
                ("image_layer2", 0): np.ones((1, 4, 4), dtype=np.uint8),
            },
        )
        
        image_data_input = DataInput("raster", ["image_layer1"], bands=["band"], passthrough=True)
        target_data_input = DataInput(
            "raster", 
            ["image_layer2"], 
            bands=["band"],
            resolution_factor=ResolutionFactor(numerator=1, denominator=2),  # 1/2 resolution
            is_target=True,
        )
        
        model_dataset = ModelDataset(
            basic_classification_dataset,
            split_config=SplitConfig(patch_size=4, load_all_patches=True),
            task=SegmentationTask(num_classes=2),
            workers=1,
            inputs={"image": image_data_input, "targets": target_data_input},
        )
        
        dataset = InMemoryAllPatchesDataset(model_dataset, (4, 4))
        assert len(dataset) > 0, "No samples were generated - test is not valid"
        
        for i in range(len(dataset)):
            inputs, targets, metadata = dataset[i]
            # Target patch should have half the resolution of the input patch
            assert inputs["image"].shape[-2:] == (4, 4)
            assert targets["classes"].shape == (2, 2)