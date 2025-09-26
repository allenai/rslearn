import logging
import multiprocessing
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from .fixtures.datasets.image_to_class import (
    image_to_class_data_module,
    image_to_class_dataset,
    image_to_class_model,
)

logging.basicConfig()


@pytest.fixture(scope="session")
def load_repository_env() -> None:
    """Expose repository-level environment variables to tests that need them."""
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@pytest.fixture(scope="session", autouse=True)
def always_spawn() -> None:
    multiprocessing.set_start_method("forkserver")


__all__ = [
    "image_to_class_dataset",
    "image_to_class_data_module",
    "image_to_class_model",
]
