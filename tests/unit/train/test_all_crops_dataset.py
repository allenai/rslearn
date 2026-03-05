"""Tests for rslearn.train.all_crops_dataset."""

from collections.abc import Callable

import numpy as np
import torch

from rslearn.dataset import Dataset
from rslearn.train.all_crops_dataset import (
    InMemoryAllCropsDataset,
    IterableAllCropsDataset,
    get_window_crop_options,
)
from rslearn.train.dataset import (
    DataInput,
    ModelDataset,
    SplitConfig,
)
from rslearn.train.model_context import RasterImage, SampleMetadata
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.train.tasks.segmentation import SegmentationTask
from rslearn.train.tasks.task import BasicTask
from rslearn.utils.geometry import PixelBounds, Projection, ResolutionFactor


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

    def test_small_window_raster_padded_to_crop_size(
        self,
        basic_classification_dataset: Dataset,
        add_window_to_basic_classification_dataset: Callable,
    ) -> None:
        """Verify that raster inputs are padded when the window is smaller than crop size.

        This is important for sliding-window inference (load_all_crops), where
        get_window_crop_options may generate crops that extend beyond window bounds.
        """
        crop_size = 32
        tiny_bounds = (0, 0, 24, 4)  # width/height smaller than crop_size

        add_window_to_basic_classification_dataset(
            basic_classification_dataset,
            name="window",
            bounds=tiny_bounds,
            images={
                ("image_layer1", 0): np.ones(
                    (
                        1,
                        tiny_bounds[3] - tiny_bounds[1],
                        tiny_bounds[2] - tiny_bounds[0],
                    ),
                    dtype=np.uint8,
                ),
            },
        )

        image_data_input = DataInput(
            "raster", ["image_layer1"], bands=["band"], passthrough=True
        )

        model_dataset = ModelDataset(
            basic_classification_dataset,
            split_config=SplitConfig(
                crop_size=crop_size, load_all_crops=True, skip_targets=True
            ),
            task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
            workers=1,
            inputs={"image": image_data_input},
        )

        all_crops_dataset = IterableAllCropsDataset(
            dataset=model_dataset,
            crop_size=(crop_size, crop_size),
        )

        samples = list(all_crops_dataset)
        assert len(samples) == 1
        inputs, _, metadata = samples[0]

        assert metadata.window_bounds == tiny_bounds
        assert metadata.crop_bounds == (0, 0, crop_size, crop_size)
        assert isinstance(inputs["image"], RasterImage)
        assert inputs["image"].shape[-2:] == (crop_size, crop_size)
        assert inputs["image"].image.shape[:2] == (1, 1)  # C, T

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


def _make_metadata(
    window_bounds: PixelBounds,
    crop_bounds: PixelBounds,
    effective_bounds: PixelBounds | None = None,
) -> SampleMetadata:
    """Helper to create a SampleMetadata for testing."""
    return SampleMetadata(
        window_group="test",
        window_name="w",
        window_bounds=window_bounds,
        crop_bounds=crop_bounds,
        crop_idx=0,
        num_crops_in_window=1,
        time_range=None,
        projection=Projection("EPSG:3857", 1, 1),
        dataset_source=None,
        effective_bounds=effective_bounds,
    )


class TestEffectiveBoundsInCropOptions:
    """Tests for effective_bounds returned by get_window_crop_options."""

    def test_no_overlap(self) -> None:
        """When window is an exact multiple of crop_size, effective == crop bounds."""
        result = get_window_crop_options((50, 50), (0, 0), (0, 0, 200, 200))
        for crop, eff in result:
            assert eff == crop

    def test_last_crop_overlap_trimmed(self) -> None:
        """The last crop's effective start is pushed past the previous crop's end."""
        # 260px window, 50px crop, stride 50 → last crop shifted back to 210
        result = get_window_crop_options((50, 50), (0, 0), (0, 0, 260, 260))

        eff_list = [eff for _, eff in result]
        # Collect effective col/row starts per crop
        eff_col_starts = sorted(set(e[0] for e in eff_list))
        eff_row_starts = sorted(set(e[1] for e in eff_list))
        # Last effective col start should be 250 (prev crop ends at 200+50=250)
        assert eff_col_starts[-1] == 250
        assert eff_row_starts[-1] == 250

        # No two crops should share any pixel.
        # Build a coverage grid and verify each pixel is covered exactly once.
        coverage = np.zeros((260, 260), dtype=int)
        for ex0, ey0, ex1, ey1 in eff_list:
            coverage[ey0:ey1, ex0:ex1] += 1
        assert (coverage == 1).all(), "Some pixels are covered more than once or missed"

    def test_single_crop(self) -> None:
        """A single crop that fills the window should be unchanged."""
        result = get_window_crop_options((50, 50), (0, 0), (0, 0, 50, 50))
        assert len(result) == 1
        crop, eff = result[0]
        assert crop == (0, 0, 50, 50)
        assert eff == (0, 0, 50, 50)

    def test_with_intentional_overlap(self) -> None:
        """With overlap_size > 0, effective bounds still avoid double-counting."""
        result = get_window_crop_options((8, 8), (2, 2), (0, 0, 15, 15))

        # Each pixel should be covered exactly once by effective bounds.
        coverage = np.zeros((15, 15), dtype=int)
        for _, (ex0, ey0, ex1, ey1) in result:
            # Clip to window since crops can extend beyond
            cx0, cy0 = max(0, ex0), max(0, ey0)
            cx1, cy1 = min(15, ex1), min(15, ey1)
            coverage[cy0:cy1, cx0:cx1] += 1
        assert (coverage == 1).all()


class TestWindowValidMaskWithEffectiveBounds:
    """Tests that _get_window_valid_mask honours effective_bounds."""

    def test_without_effective_bounds(self) -> None:
        """Without effective_bounds the entire crop is valid."""
        meta = _make_metadata(
            window_bounds=(0, 0, 260, 260),
            crop_bounds=(200, 200, 250, 250),
        )
        ref = torch.zeros(50, 50)
        mask = BasicTask._get_window_valid_mask(ref, meta)
        assert mask.sum().item() == 50 * 50

    def test_with_effective_bounds_restricts_valid(self) -> None:
        """With effective_bounds, only the non-overlapping region is valid."""
        meta = _make_metadata(
            window_bounds=(0, 0, 260, 260),
            crop_bounds=(210, 210, 260, 260),
            effective_bounds=(250, 250, 260, 260),
        )
        ref = torch.zeros(50, 50)
        mask = BasicTask._get_window_valid_mask(ref, meta)
        # Only 10×10 pixels should be valid
        assert mask.sum().item() == 10 * 10
        # Valid region should be bottom-right corner of the crop
        assert mask[40:, 40:].sum().item() == 10 * 10
        assert mask[:40, :].sum().item() == 0

    def test_effective_bounds_intersects_window_bounds(self) -> None:
        """effective_bounds and window_bounds are both applied."""
        # effective_bounds extends past window_bounds — should be clipped
        meta = _make_metadata(
            window_bounds=(0, 0, 255, 255),
            crop_bounds=(210, 210, 260, 260),
            effective_bounds=(250, 250, 260, 260),
        )
        ref = torch.zeros(50, 50)
        mask = BasicTask._get_window_valid_mask(ref, meta)
        # window ends at 255, effective starts at 250 → only 5×5 valid
        assert mask.sum().item() == 5 * 5


class TestEffectiveBoundsPropagation:
    """Verify that effective_bounds is set on metadata from AllCropsDatasets."""

    def test_iterable_sets_effective_bounds(
        self,
        basic_classification_dataset: Dataset,
        add_window_to_basic_classification_dataset: Callable,
    ) -> None:
        """IterableAllCropsDataset should set effective_bounds on each sample."""
        add_window_to_basic_classification_dataset(
            basic_classification_dataset,
            name="window0",
            bounds=(0, 0, 6, 6),
        )
        model_dataset = ModelDataset(
            basic_classification_dataset,
            split_config=SplitConfig(),
            task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
            workers=1,
            inputs={"targets": DataInput("vector", ["vector_layer"])},
        )
        ds = IterableAllCropsDataset(model_dataset, crop_size=(4, 4))
        for _, _, meta in ds:
            assert meta.effective_bounds is not None

    def test_in_memory_sets_effective_bounds(
        self,
        basic_classification_dataset: Dataset,
        add_window_to_basic_classification_dataset: Callable,
    ) -> None:
        """InMemoryAllCropsDataset should set effective_bounds on each sample."""
        add_window_to_basic_classification_dataset(
            basic_classification_dataset,
            name="window0",
            bounds=(0, 0, 6, 6),
        )
        model_dataset = ModelDataset(
            basic_classification_dataset,
            split_config=SplitConfig(),
            task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
            workers=1,
            inputs={"targets": DataInput("vector", ["vector_layer"])},
        )
        ds = InMemoryAllCropsDataset(model_dataset, crop_size=(4, 4))
        for i in range(len(ds)):
            _, _, meta = ds[i]
            assert meta.effective_bounds is not None
