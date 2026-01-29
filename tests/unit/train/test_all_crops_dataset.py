"""Tests for rslearn.train.all_crops_dataset."""

from collections.abc import Callable

import numpy as np

from rslearn.dataset import Dataset
from rslearn.train.all_crops_dataset import (
    InMemoryAllCropsDataset,
    IterableAllCropsDataset,
)
from rslearn.train.dataset import (
    DataInput,
    ModelDataset,
    SplitConfig,
)
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.train.tasks.segmentation import SegmentationTask
from rslearn.utils.geometry import PixelBounds, ResolutionFactor


class TestIterableAllCropsDataset:
    """Tests for IterableAllCropsDataset."""

    def test_crop_positions_first_and_last_only(
        self,
        basic_classification_dataset: Dataset,
        add_window_to_basic_classification_dataset: Callable,
    ) -> None:
        """Test crop positions when window only needs first/last crops (no intermediate crops).

        With a 6x6 window, crop_size=4, overlap_pixels=2:
        - First crop on each dimension covers [0, 4)
        - Last crop on each dimension covers [2, 6)

        So we get 2x2 = 4 crops starting at positions [0, 2] x [0, 2].
        """
        window_size = 6
        crop_size = 4
        overlap_pixels = 2

        add_window_to_basic_classification_dataset(
            basic_classification_dataset,
            name="window0",
            bounds=(0, 0, window_size, window_size),
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
        all_crops_dataset = IterableAllCropsDataset(
            model_dataset,
            crop_size=(crop_size, crop_size),
            overlap_pixels=overlap_pixels,
        )
        samples = list(all_crops_dataset)

        # Should have 2x2 = 4 crops (only first and last, no middle)
        assert len(samples) == 4

        # Verify exact crop bounds
        crop_bounds = set(sample[2].crop_bounds for sample in samples)
        expected_bounds = {
            (0, 0, 4, 4),
            (0, 2, 4, 6),
            (2, 0, 6, 4),
            (2, 2, 6, 6),
        }
        assert crop_bounds == expected_bounds, (
            f"Expected {expected_bounds}, got {crop_bounds}"
        )

    def test_crop_positions_with_middle_crops(
        self,
        basic_classification_dataset: Dataset,
        add_window_to_basic_classification_dataset: Callable,
    ) -> None:
        """Test crop positions when window needs middle crops between first/last.

        With a 15x15 window, crop_size=8, overlap_pixels=2:
        - First crop at 0
        - Middle crops at range(6, 7, 6) = [6] (one middle crop)
        - Last crop at 15 - 8 = 7

        So we get 3x3 = 9 crops at positions [0, 6, 7] x [0, 6, 7].
        """
        window_size = 15
        crop_size = 8
        overlap_pixels = 2

        add_window_to_basic_classification_dataset(
            basic_classification_dataset,
            name="window0",
            bounds=(0, 0, window_size, window_size),
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
        all_crops_dataset = IterableAllCropsDataset(
            model_dataset,
            crop_size=(crop_size, crop_size),
            overlap_pixels=overlap_pixels,
        )
        samples = list(all_crops_dataset)

        # Should have 3x3 = 9 crops (first, middle, and last)
        assert len(samples) == 9

        # Verify exact crop bounds
        crop_bounds = set(sample[2].crop_bounds for sample in samples)
        expected_bounds = {
            (col, row, col + crop_size, row + crop_size)
            for col in [0, 6, 7]
            for row in [0, 6, 7]
        }
        assert crop_bounds == expected_bounds, (
            f"Expected {expected_bounds}, got {crop_bounds}"
        )

    def test_distributed_one_window_per_worker(
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
            all_crops_dataset = IterableAllCropsDataset(
                model_dataset, crop_size=(4, 4), rank=rank, world_size=world_size
            )
            samples = list(all_crops_dataset)
            assert len(samples) == 1
            window_names.add(samples[0][2].window_name)
        assert len(window_names) == 4

    def test_distributed_different_window_sizes(
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
            all_crops_dataset = IterableAllCropsDataset(
                model_dataset, crop_size=(4, 4), rank=rank, world_size=world_size
            )
            samples = list(all_crops_dataset)
            assert len(samples) == 4
            for sample in samples:
                patch_id = (sample[2].window_name, sample[2].crop_bounds)
                seen_patches[patch_id] = seen_patches.get(patch_id, 0) + 1

        assert len(seen_patches) == 5
        assert seen_patches[("window0", (0, 0, 4, 4))] == 4
        assert seen_patches[("window1", (0, 0, 4, 4))] == 1
        assert seen_patches[("window1", (0, 4, 4, 8))] == 1
        assert seen_patches[("window1", (4, 0, 8, 4))] == 1
        assert seen_patches[("window1", (4, 4, 8, 8))] == 1

    def test_empty_dataset(self, basic_classification_dataset: Dataset) -> None:
        """Verify that IterableAllCropsDataset works with no windows."""
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
            all_crops_dataset = IterableAllCropsDataset(
                model_dataset, crop_size=(4, 4), rank=rank, world_size=world_size
            )
            samples = list(all_crops_dataset)
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
        all_crops_dataset = IterableAllCropsDataset(
            dataset=model_dataset,
            crop_size=(4, 4),
        )
        samples = list(all_crops_dataset)
        assert len(samples) == 1
        _, _, metadata = samples[0]
        assert metadata.window_bounds == (0, 0, 2, 2)
        assert metadata.crop_bounds == (0, 0, 4, 4)

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
            split_config=SplitConfig(crop_size=4, load_all_crops=True),
            task=SegmentationTask(num_classes=2),
            workers=1,
            inputs={"image": image_data_input, "targets": target_data_input},
        )

        dataset = IterableAllCropsDataset(
            model_dataset, crop_size=(4, 4), rank=0, world_size=1
        )

        # Verify we actually have samples to test
        sample_count = 0
        for inputs, targets, metadata in dataset:
            sample_count += 1
            assert inputs["image"].shape[-2:] == (4, 4)
            assert targets["classes"].get_hw_tensor().shape == (2, 2)
        assert sample_count > 0, "No samples were generated - test is not valid"


class TestInMemoryAllCropsDataset:
    """Tests for InMemoryAllCropsDataset."""

    def test_iterable_equal(
        self,
        basic_classification_dataset: Dataset,
        add_window_to_basic_classification_dataset: Callable,
    ) -> None:
        """Verify that InMemoryAllCropsDataset and IterableAllCropsDataset are equivalent."""
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
        iterable_ds = IterableAllCropsDataset(
            model_dataset, crop_size=crop_size, rank=0, world_size=1
        )
        regular_ds = InMemoryAllCropsDataset(model_dataset, crop_size=crop_size)

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
            split_config=SplitConfig(crop_size=4, load_all_crops=True),
            task=SegmentationTask(num_classes=2),
            workers=1,
            inputs={"image": image_data_input, "targets": target_data_input},
        )

        dataset = InMemoryAllCropsDataset(model_dataset, crop_size=(4, 4))
        assert len(dataset) > 0, "No samples were generated - test is not valid"

        for i in range(len(dataset)):
            inputs, targets, metadata = dataset[i]
            # Target patch should have half the resolution of the input patch
            assert inputs["image"].shape[-2:] == (4, 4)
            assert targets["classes"].get_hw_tensor().shape == (2, 2)
