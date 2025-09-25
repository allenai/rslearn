import logging
import multiprocessing
from pathlib import Path

import pytest
from dotenv import load_dotenv

from .fixtures.datasets.image_to_class import (
    image_to_class_data_module,
    image_to_class_dataset,
    image_to_class_model,
)

logging.basicConfig()


# Load environment variables from the repository-level .env so integration tests have
# access to credentials required by external data sources (e.g., EarthDaily).
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@pytest.fixture(scope="session", autouse=True)
def always_spawn() -> None:
    multiprocessing.set_start_method("forkserver")


__all__ = [
    "image_to_class_dataset",
    "image_to_class_data_module",
    "image_to_class_model",
]
