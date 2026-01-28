"""Tests for rslearn.train.all_patches_dataset."""

from collections.abc import Callable

import numpy as np

from rslearn.dataset import Dataset
from rslearn.train.all_patches_dataset import (
    InMemoryAllPatchesDataset,
    IterableAllPatchesDataset,
)
from rslearn.train.dataset import (
    DataInput,
    ModelDataset,
    SplitConfig,
)
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.train.tasks.segmentation import SegmentationTask
from rslearn.utils.geometry import PixelBounds, ResolutionFactor


def test_dataset_covers_border(image_to_class_dataset: Dataset) -> None:
    # Make sure that, when loading all patches, the border of the raster is included in
    # the generated windows.
    # The image_to_class_dataset window is 4x4 so 3x3 crop will ensure irregular window
    # at the border.
    crop_size = 3
    split_config = SplitConfig(
        crop_size=crop_size,
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
    dataset = IterableAllPatchesDataset(model_dataset, (crop_size, crop_size))

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

    # Test with overlap_pixels=1 for 2x2 crops
    dataset_with_overlap = IterableAllPatchesDataset(
        model_dataset, (2, 2), overlap_pixels=1
    )

    point_coverage = {}
    for col in range(4):
        for row in range(4):
            point_coverage[(col, row)] = False

    # With overlap_pixels=1, there should be 9 windows given that overlap is 1 pixel.
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


class TestIterableAllPatchesDataset:
    """Tests for IterableAllPatchesDataset."""

    def test_one_window_per_worker(
        self,
        basic_classification_dataset: Dataset,
        add_window_to_basic_classification_dataset: Callable,
    ) -> None:
        """Verify that things work with one window per worker."""
        add_window_to_basic_classification_dataset(
            basic_classification_dataset, name="window0"
        )
        add_window_to_basic_classification_dataset(
            basic_classification_dataset, name="window1"
        )
        add_window_to_basic_classification_dataset(
            basic_classification_dataset, name="window2"
        )
        add_window_to_basic_classification_dataset(
            basic_classification_dataset, name="window3"
        )
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
                model_dataset, crop_size=(4, 4), rank=rank, world_size=world_size
            )
            samples = list(all_patches_dataset)
            assert len(samples) == 1
            window_names.add(samples[0][2].window_name)
        assert len(window_names) == 4

    def test_different_window_sizes(
        self,
        basic_classification_dataset: Dataset,
        add_window_to_basic_classification_dataset: Callable,
    ) -> None:
        """Verify that rank padding works with different window sizes."""
        # One rank should get the second window.
        # While the other rank should get first window and needs to repeat it.
        add_window_to_basic_classification_dataset(
            basic_classification_dataset, name="window0", bounds=(0, 0, 4, 4)
        )
        add_window_to_basic_classification_dataset(
            basic_classification_dataset, name="window1", bounds=(0, 0, 8, 8)
        )
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
                model_dataset, crop_size=(4, 4), rank=rank, world_size=world_size
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
                model_dataset, crop_size=(4, 4), rank=rank, world_size=world_size
            )
            samples = list(all_patches_dataset)
            assert len(samples) == 0

    def test_small_window(
        self,
        basic_classification_dataset: Dataset,
        add_window_to_basic_classification_dataset: Callable,
    ) -> None:
        """Verify that it works when the window is smaller than the crop size."""
        add_window_to_basic_classification_dataset(
            basic_classification_dataset, name="window", bounds=(0, 0, 2, 2)
        )
        model_dataset = ModelDataset(
            basic_classification_dataset,
            split_config=SplitConfig(),
            task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
            workers=1,
            inputs={
                "targets": DataInput("vector", ["vector_layer"]),
            },
        )
        all_patches_dataset = IterableAllPatchesDataset(
            dataset=model_dataset,
            crop_size=(4, 4),
        )
        samples = list(all_patches_dataset)
        assert len(samples) == 1
        _, _, metadata = samples[0]
        assert metadata.window_bounds == (0, 0, 2, 2)
        assert metadata.patch_bounds == (0, 0, 4, 4)

    def test_resolution_factor_cropping(
        self,
        basic_classification_dataset: Dataset,
        add_window_to_basic_classification_dataset: Callable,
    ) -> None:
        """Verify that cropping works correctly with different resolution factors."""

        add_window_to_basic_classification_dataset(
            basic_classification_dataset,
            name="window0",
            bounds=(0, 0, 8, 8),
            images={
                ("image_layer1", 0): np.ones((1, 8, 8), dtype=np.uint8),
                ("image_layer2", 0): np.ones((1, 4, 4), dtype=np.uint8),
            },
        )

        image_data_input = DataInput(
            "raster", ["image_layer1"], bands=["band"], passthrough=True
        )
        target_data_input = DataInput(
            "raster",
            ["image_layer2"],
            bands=["band"],
            resolution_factor=ResolutionFactor(
                numerator=1, denominator=2
            ),  # 1/2 resolution
            is_target=True,
        )

        model_dataset = ModelDataset(
            basic_classification_dataset,
            split_config=SplitConfig(crop_size=4, load_all_patches=True),
            task=SegmentationTask(num_classes=2),
            workers=1,
            inputs={"image": image_data_input, "targets": target_data_input},
        )

        dataset = IterableAllPatchesDataset(
            model_dataset, crop_size=(4, 4), rank=0, world_size=1
        )

        # Verify we actually have samples to test
        sample_count = 0
        for inputs, targets, metadata in dataset:
            sample_count += 1
            assert inputs["image"].shape[-2:] == (4, 4)
            assert targets["classes"].get_hw_tensor().shape == (2, 2)
        assert sample_count > 0, "No samples were generated - test is not valid"


class TestInMemoryAllPatchesDataset:
    """Tests for InMemoryAllPatchesDataset."""

    def test_iterable_equal(
        self,
        basic_classification_dataset: Dataset,
        add_window_to_basic_classification_dataset: Callable,
    ) -> None:
        """Verify that InMemoryAllPatchesDataset and IterableAllPatchesDataset are equivalent."""
        # Create a couple of windows with different sizes to exercise patching.
        add_window_to_basic_classification_dataset(
            basic_classification_dataset, name="w0", bounds=(0, 0, 4, 4)
        )
        add_window_to_basic_classification_dataset(
            basic_classification_dataset, name="w1", bounds=(0, 0, 8, 8)
        )

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
        crop_size = (3, 3)
        iterable_ds = IterableAllPatchesDataset(
            model_dataset, crop_size=crop_size, rank=0, world_size=1
        )
        regular_ds = InMemoryAllPatchesDataset(model_dataset, crop_size=crop_size)

        iterable_samples = list(iterable_ds)
        regular_samples = [regular_ds[i] for i in range(len(regular_ds))]

        # Compare metadata (last element of each tuple) index-by-index.
        assert len(iterable_samples) == len(regular_samples)
        for i in range(len(iterable_samples)):
            assert iterable_samples[i][-1] == regular_samples[i][-1]

    def test_resolution_factor_cropping(
        self,
        basic_classification_dataset: Dataset,
        add_window_to_basic_classification_dataset: Callable,
    ) -> None:
        """Verify that cropping works correctly with different resolution factors."""

        add_window_to_basic_classification_dataset(
            basic_classification_dataset,
            name="window0",
            bounds=(0, 0, 8, 8),
            images={
                ("image_layer1", 0): np.ones((1, 8, 8), dtype=np.uint8),
                ("image_layer2", 0): np.ones((1, 4, 4), dtype=np.uint8),
            },
        )

        image_data_input = DataInput(
            "raster", ["image_layer1"], bands=["band"], passthrough=True
        )
        target_data_input = DataInput(
            "raster",
            ["image_layer2"],
            bands=["band"],
            resolution_factor=ResolutionFactor(
                numerator=1, denominator=2
            ),  # 1/2 resolution
            is_target=True,
        )

        model_dataset = ModelDataset(
            basic_classification_dataset,
            split_config=SplitConfig(crop_size=4, load_all_patches=True),
            task=SegmentationTask(num_classes=2),
            workers=1,
            inputs={"image": image_data_input, "targets": target_data_input},
        )

        dataset = InMemoryAllPatchesDataset(model_dataset, crop_size=(4, 4))
        assert len(dataset) > 0, "No samples were generated - test is not valid"

        for i in range(len(dataset)):
            inputs, targets, metadata = dataset[i]
            # Target patch should have half the resolution of the input patch
            assert inputs["image"].shape[-2:] == (4, 4)
            assert targets["classes"].get_hw_tensor().shape == (2, 2)
