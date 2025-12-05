import pytest

from rslearn.const import WGS84_PROJECTION
from rslearn.train.model_context import SampleMetadata


@pytest.fixture
def empty_sample_metadata() -> SampleMetadata:
    """Many functions don't actually use the metadata, so this empty metadata can be used."""
    return SampleMetadata(
        window_group="",
        window_name="",
        window_bounds=(0, 0, 0, 0),
        patch_bounds=(0, 0, 0, 0),
        patch_idx=0,
        num_patches_in_window=1,
        time_range=None,
        projection=WGS84_PROJECTION,
        dataset_source=None,
    )
