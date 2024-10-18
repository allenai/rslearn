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
    return os.environ.get("TEST_BUCKET", "test-bucket1")


@pytest.fixture(scope="session")
def test_prefix():
    return os.environ.get("TEST_PREFIX", "tests/")


@pytest.fixture(scope="session")
def test_bucket_path(test_bucket, test_prefix):
    """For local testing we can use a real GCS bucket, but for CI we need to use a fake one."""
    host_prefix = os.environ.get("STORAGE_EMULATOR_HOST")
    if host_prefix == "gcs://":
        # We're using a real GCS bucket
        return f"{host_prefix}{test_bucket}/{test_prefix}"
    else:
        # We're using the emulator, so we need to create the bucket
        from google.cloud import storage

        storage_client = storage.Client()
        if not storage_client.bucket(test_bucket).exists():
            bucket = storage_client.create_bucket("test-bucket")
            print(f"Bucket {bucket.name} created.")

    return f"{host_prefix}{test_bucket}/{test_prefix}"
