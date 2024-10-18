import os

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
def set_storage_emulator_host():
    os.environ.setdefault("STORAGE_EMULATOR_HOST", "http://localhost:4443")


@pytest.fixture(scope="session")
def test_bucket():
    return os.environ.get("TEST_BUCKET", "test-bucket3")
