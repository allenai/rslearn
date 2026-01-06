"""Test jsonargparse parsing."""

import json
from pathlib import Path

import jsonargparse
import pytest
from lightning import LightningDataModule

from rslearn.arg_parser import RslearnArgumentParser
from rslearn.dataset.dataset import Dataset
from rslearn.lightning_cli import RslearnLightningCLI
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.lightning_module import RslearnLightningModule
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


def test_simple_model_config(
    image_to_class_dataset: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Make sure the overall parsing of model config works.

    Among other things this ensures RslearnArgumentParser is working properly.
    """
    # We define a model config for classification task, corresponding to the
    # image_to_class_dataset from tests/fixtures/datasets/image_to_class.py.
    cfg = {
        "model": {
            "class_path": "rslearn.train.lightning_module.RslearnLightningModule",
            "init_args": {
                "model": {
                    "class_path": "rslearn.models.singletask.SingleTaskModel",
                    "init_args": {
                        # Apply Swin feature extractor with one band input, for classification.
                        "encoder": [
                            {
                                "class_path": "rslearn.models.swin.Swin",
                                "init_args": {
                                    "input_channels": 1,
                                    "num_outputs": 2,
                                },
                            }
                        ],
                        # Apply classification head on the FeatureVector output from Swin model.
                        "decoder": [
                            {
                                "class_path": "rslearn.train.tasks.classification.ClassificationHead"
                            },
                        ],
                    },
                },
                "optimizer": {"class_path": "rslearn.train.optimizer.AdamW"},
            },
        },
        "data": {
            "class_path": "rslearn.train.data_module.RslearnDataModule",
            "init_args": {
                # We will get the DATASET_PATH from environment.
                "path": "${DATASET_PATH}",
                # The inputs and task options correspond to image_to_class_dataset.
                "inputs": {
                    "image": {
                        "data_type": "raster",
                        "layers": ["image"],
                        "bands": ["band"],
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
                        "property_name": "label",
                        "classes": ["class0", "class1"],
                        "read_class_id": True,
                    },
                },
                "batch_size": 1,
            },
        },
    }

    # We could pass the config in args, but we pass it as file because that is more
    # like typical usage and it has a different code path that we want to make sure
    # works.
    tmp_fname = tmp_path / "model.yaml"
    with tmp_fname.open("w") as f:
        json.dump(cfg, f)

    monkeypatch.setenv("DATASET_PATH", str(image_to_class_dataset.path))
    cli = RslearnLightningCLI(
        model_class=RslearnLightningModule,
        datamodule_class=RslearnDataModule,
        args=["--config", str(tmp_fname)],
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        parser_class=RslearnArgumentParser,
        run=False,
    )
    assert isinstance(cli.datamodule, RslearnDataModule)
    assert isinstance(cli.model, RslearnLightningModule)
    assert isinstance(cli.datamodule.task, ClassificationTask)
    assert cli.datamodule.task.property_name == "label"

    # Make sure dataset path was good from environment (using RslearnArgumentParser).
    assert cli.datamodule.path == image_to_class_dataset.path
