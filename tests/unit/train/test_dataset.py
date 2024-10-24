import pytest
from torch.utils.data import Dataset

from rslearn.train.dataset import DataInput, ModelDataset, RetryDataset, SplitConfig
from rslearn.train.tasks.classification import ClassificationTask


class TestException(Exception):
    pass


class TestDataset(Dataset):
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
