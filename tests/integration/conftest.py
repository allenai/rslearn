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


@pytest.fixture(scope="session", autouse=True)
def test_bucket():
    os.environ.setdefault("TEST_BUCKET", "test-bucket-rslearn")
    test_bucket = os.environ["TEST_BUCKET"]
    print(f"test_bucket: {test_bucket}")
    return test_bucket


@pytest.fixture(scope="session", autouse=True)
def test_prefix():
    os.environ.setdefault("TEST_PREFIX", "tests/")
    return os.environ["TEST_PREFIX"]
