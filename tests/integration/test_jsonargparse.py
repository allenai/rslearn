"""Test jsonargparse parsing."""

import jsonargparse
import pytest
from lightning import LightningDataModule

from rslearn.dataset.dataset import Dataset
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.utils.jsonargparse import init_jsonargparse


@pytest.fixture(autouse=True)
def setup_fixture() -> None:
    init_jsonargparse()


def test_simple_data_config(local_files_dataset: Dataset) -> None:
    """Make sure a simple data module config is parsed correctly."""
    # We define a simple data module that has two inputs using ClassificationTask.
    # We need the local_files_dataset since it will need actual dataset on disk when
    # instantiating the RslearnDataModule.
    cfg_dict = {
        "data": {
            "class_path": "rslearn.train.data_module.RslearnDataModule",
            "init_args": {
                "path": str(local_files_dataset.path),
                "inputs": {
                    "image": {
                        "data_type": "raster",
                        "layers": ["image"],
                        "bands": ["R", "G", "B"],
                        "passthrough": True,
                        "dtype": "FLOAT32",
                    },
                    "targets": {
                        "data_type": "vector",
                        "layers": ["label"],
                        "is_target": True,
                    },
                },
                "task": {
                    "class_path": "rslearn.train.tasks.classification.ClassificationTask",
                    "init_args": {
                        "property_name": "category",
                        "classes": ["class0", "class1"],
                    },
                },
                "batch_size": 8,
            },
        }
    }
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--data", type=LightningDataModule)
    cfg = parser.parse_object(cfg_dict)
    data_module = parser.instantiate_classes(cfg).data
    assert isinstance(data_module, RslearnDataModule)
    assert isinstance(data_module.task, ClassificationTask)
    assert data_module.task.property_name == "category"
