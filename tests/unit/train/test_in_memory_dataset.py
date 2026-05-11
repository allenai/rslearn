"""Tests for rslearn.train.in_memory_dataset."""

from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import numpy as np

from rslearn.dataset import Dataset
from rslearn.train.dataset import (
    DataInput,
    ModelDataset,
    SplitConfig,
)
from rslearn.train.in_memory_dataset import (
    InMemoryRandomCropDataset,
)
from rslearn.train.model_context import SampleMetadata
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.utils.geometry import PixelBounds


class TestInMemoryRandomCropDataset:
    """Tests for InMemoryRandomCropDataset."""

    def test_len_equals_num_windows(
        self,
        basic_classification_dataset: Dataset,
        add_window_to_basic_classification_dataset: Callable,
    ) -> None:
        """len() should equal the number of windows regardless of crop size."""
        add_window_to_basic_classification_dataset(
            basic_classification_dataset, name="w0", bounds=(0, 0, 8, 8)
        )
        add_window_to_basic_classification_dataset(
            basic_classification_dataset, name="w1", bounds=(0, 0, 16, 16)
        )
        model_dataset = ModelDataset(
            basic_classification_dataset,
            split_config=SplitConfig(load_all_crops=True),
            task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
            workers=1,
            inputs={"targets": DataInput("vector", ["vector_layer"])},
        )
        ds = InMemoryRandomCropDataset(model_dataset, crop_size=(4, 4))
        assert len(ds) == 2

    def test_random_crop_within_bounds(
        self,
        basic_classification_dataset: Dataset,
        add_window_to_basic_classification_dataset: Callable,
    ) -> None:
        """Every returned crop_bounds should be crop_size and fit inside the window."""
        window_bounds: PixelBounds = (0, 0, 16, 16)
        crop_size = (4, 4)
        add_window_to_basic_classification_dataset(
            basic_classification_dataset, name="w0", bounds=window_bounds
        )
        model_dataset = ModelDataset(
            basic_classification_dataset,
            split_config=SplitConfig(load_all_crops=True),
            task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
            workers=1,
            inputs={"targets": DataInput("vector", ["vector_layer"])},
        )
        ds = InMemoryRandomCropDataset(model_dataset, crop_size=crop_size)

        for _ in range(20):
            _, _, metadata = ds[0]
            cb = metadata.crop_bounds
            assert cb[2] - cb[0] == crop_size[0]
            assert cb[3] - cb[1] == crop_size[1]
            assert cb[0] >= window_bounds[0]
            assert cb[1] >= window_bounds[1]
            assert cb[2] <= window_bounds[2]
            assert cb[3] <= window_bounds[3]

    def test_fix_crop_pick_deterministic(
        self,
        basic_classification_dataset: Dataset,
        add_window_to_basic_classification_dataset: Callable,
    ) -> None:
        """With fix_crop_pick=True, same index always yields the same crop."""
        add_window_to_basic_classification_dataset(
            basic_classification_dataset, name="w0", bounds=(0, 0, 16, 16)
        )
        model_dataset = ModelDataset(
            basic_classification_dataset,
            split_config=SplitConfig(load_all_crops=True),
            task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
            workers=1,
            inputs={"targets": DataInput("vector", ["vector_layer"])},
        )
        ds = InMemoryRandomCropDataset(
            model_dataset, crop_size=(4, 4), fix_crop_pick=True
        )
        first = ds[0][2].crop_bounds
        for _ in range(10):
            assert ds[0][2].crop_bounds == first

    def test_random_crop_varies(
        self,
        basic_classification_dataset: Dataset,
        add_window_to_basic_classification_dataset: Callable,
    ) -> None:
        """With fix_crop_pick=False, repeated calls should eventually differ."""
        add_window_to_basic_classification_dataset(
            basic_classification_dataset, name="w0", bounds=(0, 0, 64, 64)
        )
        model_dataset = ModelDataset(
            basic_classification_dataset,
            split_config=SplitConfig(load_all_crops=True),
            task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
            workers=1,
            inputs={"targets": DataInput("vector", ["vector_layer"])},
        )
        ds = InMemoryRandomCropDataset(
            model_dataset, crop_size=(4, 4), fix_crop_pick=False
        )
        bounds_seen = set()
        for _ in range(50):
            bounds_seen.add(ds[0][2].crop_bounds)
        assert len(bounds_seen) > 1

    def test_window_loaded_once(
        self,
        basic_classification_dataset: Dataset,
        add_window_to_basic_classification_dataset: Callable,
    ) -> None:
        """The underlying ModelDataset.get_raw_inputs should be called once per window."""
        add_window_to_basic_classification_dataset(
            basic_classification_dataset, name="w0", bounds=(0, 0, 8, 8)
        )
        model_dataset = ModelDataset(
            basic_classification_dataset,
            split_config=SplitConfig(load_all_crops=True),
            task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
            workers=1,
            inputs={"targets": DataInput("vector", ["vector_layer"])},
        )
        ds = InMemoryRandomCropDataset(model_dataset, crop_size=(4, 4))

        original_get_raw = ModelDataset.get_raw_inputs
        call_count = 0

        def counting_get_raw(
            self_inner: ModelDataset, idx: int
        ) -> tuple[dict[str, Any], dict[str, Any], SampleMetadata]:
            nonlocal call_count
            call_count += 1
            return original_get_raw(self_inner, idx)

        with patch.object(ModelDataset, "get_raw_inputs", counting_get_raw):
            for _ in range(10):
                ds[0]

        assert call_count == 1

    def test_small_window_crop_larger_than_window(
        self,
        basic_classification_dataset: Dataset,
        add_window_to_basic_classification_dataset: Callable,
    ) -> None:
        """When crop_size > window_size, the crop should still be returned."""
        add_window_to_basic_classification_dataset(
            basic_classification_dataset,
            name="w0",
            bounds=(0, 0, 2, 2),
            images={("image_layer1", 0): np.ones((1, 2, 2), dtype=np.uint8)},
        )
        model_dataset = ModelDataset(
            basic_classification_dataset,
            split_config=SplitConfig(load_all_crops=True, skip_targets=True),
            task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
            workers=1,
            inputs={
                "image": DataInput(
                    "raster", ["image_layer1"], bands=["band"], passthrough=True
                ),
            },
        )
        ds = InMemoryRandomCropDataset(model_dataset, crop_size=(4, 4))
        inputs, _, metadata = ds[0]
        cb = metadata.crop_bounds
        assert cb[2] - cb[0] == 4
        assert cb[3] - cb[1] == 4

    def test_no_crop_size_returns_full_window(
        self,
        basic_classification_dataset: Dataset,
        add_window_to_basic_classification_dataset: Callable,
    ) -> None:
        """With crop_size=None, returns the full window bounds and caches correctly."""
        bounds: PixelBounds = (0, 0, 8, 8)
        add_window_to_basic_classification_dataset(
            basic_classification_dataset, name="w0", bounds=bounds
        )
        add_window_to_basic_classification_dataset(
            basic_classification_dataset, name="w1", bounds=(0, 0, 16, 16)
        )
        model_dataset = ModelDataset(
            basic_classification_dataset,
            split_config=SplitConfig(load_all_crops=True),
            task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
            workers=1,
            inputs={"targets": DataInput("vector", ["vector_layer"])},
        )
        ds = InMemoryRandomCropDataset(model_dataset, crop_size=None)
        assert len(ds) == 2

        _, _, meta0 = ds[0]
        _, _, meta1 = ds[1]
        # Should return full window bounds unchanged
        assert meta0.crop_bounds == meta0.window_bounds
        assert meta1.crop_bounds == meta1.window_bounds
