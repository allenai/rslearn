import logging
import multiprocessing

import pytest

from .fixtures.datasets.image_to_class import image_to_class_dataset

logging.basicConfig()


@pytest.fixture(scope="session", autouse=True)
def always_spawn():
    multiprocessing.set_start_method("forkserver")


__all__ = [
    "image_to_class_dataset",
]
