"""Integration tests for RslearnLightningCLI and model management features."""

import json
import pathlib

import lightning.pytorch as L
import numpy as np
import pytest
import shapely
import torch
from upath import UPath

from rslearn.arg_parser import RslearnArgumentParser
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.lightning_cli import RslearnLightningCLI
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import STGeometry
from rslearn.utils.jsonargparse import init_jsonargparse
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat

CLASSES = ["cat", "dog"]
PROPERTY_NAME = "category"


class CrashAtEpochCallback(L.Callback):
    """Callback that raises an exception at a specified epoch.

    This prevents on_train_end from running, ensuring that tests exercise
    checkpoint-saving behavior during training rather than at shutdown.
    """

    def __init__(self, crash_at_epoch: int = 2) -> None:
        """Initialize CrashAtEpochCallback.

        Args:
            crash_at_epoch: the epoch (0-indexed) at which to crash.
        """
        self.crash_at_epoch = crash_at_epoch

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Raise if we reached the crash epoch.

        Args:
            trainer: the trainer.
            pl_module: the lightning module.
        """
        if trainer.current_epoch >= self.crash_at_epoch:
            raise RuntimeError("Intentional crash for testing")


@pytest.fixture
def classification_dataset(tmp_path: pathlib.Path) -> Dataset:
    """Create a minimal classification dataset with one 32x32 window."""
    ds_path = UPath(tmp_path / "dataset")

    dataset_config = {
        "layers": {
            "image": {
                "type": "raster",
                "band_sets": [{"dtype": "uint8", "bands": ["band"]}],
            },
            "targets": {"type": "vector"},
        },
    }
    ds_path.mkdir(parents=True, exist_ok=True)
    with (ds_path / "config.json").open("w") as f:
        json.dump(dataset_config, f)
    dataset = Dataset(ds_path)

    window = Window(
        storage=dataset.storage,
        group="default",
        name="default",
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 32, 32),
        time_range=None,
    )
    window.save()

    image = np.random.randint(0, 255, size=(1, 32, 32), dtype=np.uint8)
    layer_dir = window.get_layer_dir("image")
    GeotiffRasterFormat().encode_raster(
        layer_dir / "band",
        window.projection,
        window.bounds,
        RasterArray(chw_array=image),
    )
    window.mark_layer_completed("image")

    feature = Feature(
        STGeometry(WGS84_PROJECTION, shapely.Point(16, 16), None),
        {PROPERTY_NAME: CLASSES[0]},
    )
    layer_dir = window.get_layer_dir("targets")
    GeojsonVectorFormat().encode_vector(layer_dir, [feature])
    window.mark_layer_completed("targets")

    return dataset


def test_save_last_every_epoch_and_best_when_metric_improves(
    classification_dataset: Dataset,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With model manegement, save last.ckpt on every epoch and best.ckpt when the metric improves.

    We crash at epoch 2 to avoid on_train_end saving last.ckpt at shutdown. With lr=0
    the model never updates so val_loss is constant, so save_top_k should only save the
    epoch-0 checkpoint, but save_last should update last.ckpt every epoch.
    """
    init_jsonargparse()

    management_dir = tmp_path / "management"
    project_name = "test_project"
    run_name = "test_run"

    cfg = {
        "model": {
            "class_path": "rslearn.train.lightning_module.RslearnLightningModule",
            "init_args": {
                "model": {
                    "class_path": "rslearn.models.singletask.SingleTaskModel",
                    "init_args": {
                        "encoder": [
                            {
                                "class_path": "rslearn.models.swin.Swin",
                                "init_args": {
                                    "arch": "swin_t",
                                    "input_channels": 1,
                                    "num_outputs": len(CLASSES),
                                },
                            }
                        ],
                        "decoder": [
                            {
                                "class_path": "rslearn.train.tasks.classification.ClassificationHead",
                            },
                        ],
                    },
                },
                "optimizer": {
                    "class_path": "rslearn.train.optimizer.AdamW",
                    "init_args": {"lr": 0},
                },
            },
        },
        "data": {
            "class_path": "rslearn.train.data_module.RslearnDataModule",
            "init_args": {
                "path": "${DATASET_PATH}",
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
                        "layers": ["targets"],
                    },
                },
                "task": {
                    "class_path": "rslearn.train.tasks.classification.ClassificationTask",
                    "init_args": {
                        "property_name": PROPERTY_NAME,
                        "classes": CLASSES,
                    },
                },
                "batch_size": 1,
            },
        },
        "trainer": {
            "max_epochs": 3,
            "accelerator": "cpu",
            "callbacks": [
                {
                    "class_path": "tests.integration.test_lightning_cli.CrashAtEpochCallback",
                    "init_args": {"crash_at_epoch": 2},
                },
                {
                    "class_path": "rslearn.train.callbacks.checkpointing.ManagedBestLastCheckpoint",
                    "init_args": {"monitor": "val_loss", "mode": "min"},
                },
            ],
        },
        "management_dir": str(management_dir),
        "project_name": project_name,
        "run_name": run_name,
        "log_mode": "no",
    }

    tmp_fname = tmp_path / "config.yaml"
    with tmp_fname.open("w") as f:
        json.dump(cfg, f)

    monkeypatch.setenv("DATASET_PATH", str(classification_dataset.path))
    with pytest.raises(RuntimeError, match="Intentional crash"):
        RslearnLightningCLI(
            model_class=RslearnLightningModule,
            datamodule_class=RslearnDataModule,
            args=["fit", "--config", str(tmp_fname)],
            subclass_mode_model=True,
            subclass_mode_data=True,
            save_config_kwargs={"overwrite": True},
            parser_class=RslearnArgumentParser,
        )

    project_dir = management_dir / project_name / run_name

    # The best checkpoint should be from epoch 0 (the first and only time the metric
    # was recorded as "best", since lr=0 means val_loss is constant).
    best_ckpt_path = project_dir / "best.ckpt"
    assert best_ckpt_path.exists(), "best.ckpt should exist"
    best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    assert best_ckpt["epoch"] == 0, (
        f"Best checkpoint should be from epoch 0, but got epoch {best_ckpt['epoch']}"
    )

    # last.ckpt should be from epoch 1 (the last fully completed epoch before crash).
    last_ckpt_path = project_dir / "last.ckpt"
    assert last_ckpt_path.exists(), "last.ckpt should exist"
    last_ckpt = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
    assert last_ckpt["epoch"] == 1, (
        f"last.ckpt should be from epoch 1, but got epoch {last_ckpt['epoch']}"
    )
