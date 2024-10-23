import os

import pytest
from google.cloud import storage

from .fixtures.datasets.local_files_dataset import local_files_dataset
from .fixtures.geometries.seattle2020 import seattle2020

__all__ = [
    "local_files_dataset",
    "seattle2020",
]


# maybe I don't want to explictly autouse this
@pytest.fixture(scope="session", autouse=True)
def test_bucket() -> str:
    os.environ.setdefault("TEST_BUCKET", "test-bucket-rslearn")
    test_bucket = os.environ["TEST_BUCKET"]
    print(f"test_bucket: {test_bucket}")
    storage_client = storage.Client()
    try:
        storage_client.get_bucket(test_bucket)
        print(f"Bucket {test_bucket} exists.")
    except Exception as e:
        raise AssertionError(f"Bucket {test_bucket} does not exist: {str(e)}")
    return test_bucket


@pytest.fixture(scope="session", autouse=True)
def test_prefix() -> str:
    os.environ.setdefault("TEST_PREFIX", "tests/")
    return os.environ["TEST_PREFIX"]
