import logging
import multiprocessing

import pytest

from .fixtures.datasets.image_to_class import (
    image_to_class_data_module,
    image_to_class_dataset,
    image_to_class_model,
)

logging.basicConfig()


@pytest.fixture(scope="session", autouse=True)
def always_spawn() -> None:
    multiprocessing.set_start_method("forkserver", force=True)


__all__ = [
    "image_to_class_dataset",
    "image_to_class_data_module",
    "image_to_class_model",
]
