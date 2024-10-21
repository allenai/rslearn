import os
from pathlib import Path

import pytest

from .fixtures.datasets.image_to_class import image_to_class_dataset
from .fixtures.datasets.local_files_dataset import local_files_dataset
from .fixtures.geometries.seattle2020 import seattle2020

__all__ = [
    "image_to_class_dataset",
    "local_files_dataset",
    "seattle2020",
]


@pytest.fixture(scope="session")
def test_bucket():
    test_bucket = os.environ.get("TEST_BUCKET", "test-bucket-rslearn")
    print(f"test_bucket: {test_bucket}")
    return test_bucket


@pytest.fixture(scope="session")
def test_prefix():
    return os.environ.get("TEST_PREFIX", "tests/")
