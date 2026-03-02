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


def _make_classification_config(
    management_dir: pathlib.Path,
    callbacks: list[dict],
    project_name: str = "test_project",
    run_name: str = "test_run",
) -> dict:
    """Build a classification training config with the given callbacks."""
    return {
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
            "callbacks": callbacks,
        },
        "management_dir": str(management_dir),
        "project_name": project_name,
        "run_name": run_name,
        "log_mode": "no",
    }


def _run_cli_fit(
    cfg: dict,
    tmp_path: pathlib.Path,
    dataset_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write config to disk and run RslearnLightningCLI fit."""
    init_jsonargparse()
    tmp_fname = tmp_path / "config.yaml"
    with tmp_fname.open("w") as f:
        json.dump(cfg, f)
    monkeypatch.setenv("DATASET_PATH", dataset_path)
    RslearnLightningCLI(
        model_class=RslearnLightningModule,
        datamodule_class=RslearnDataModule,
        args=["fit", "--config", str(tmp_fname)],
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        parser_class=RslearnArgumentParser,
    )


def test_save_last_every_epoch_and_best_when_metric_improves(
    classification_dataset: Dataset,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With model management, save last.ckpt on every epoch and best.ckpt when the metric improves.

    We crash at epoch 2 to avoid on_train_end saving last.ckpt at shutdown. With lr=0
    the model never updates so val_loss is constant, so save_top_k should only save the
    epoch-0 checkpoint, but save_last should update last.ckpt every epoch.
    """
    management_dir = tmp_path / "management"
    cfg = _make_classification_config(
        management_dir,
        callbacks=[
            {
                "class_path": "tests.integration.test_lightning_cli.CrashAtEpochCallback",
                "init_args": {"crash_at_epoch": 2},
            },
            {
                "class_path": "rslearn.train.callbacks.checkpointing.ManagedBestLastCheckpoint",
                "init_args": {"monitor": "val_loss", "mode": "min"},
            },
        ],
    )

    with pytest.raises(RuntimeError, match="Intentional crash"):
        _run_cli_fit(cfg, tmp_path, str(classification_dataset.path), monkeypatch)

    project_dir = management_dir / "test_project" / "test_run"

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


def test_checkpoint_monitors_validation_metric(
    classification_dataset: Dataset,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checkpointing on a validation metric (val_accuracy) should work.

    Previously there was a bug where monitoring val_loss worked but metrics weren't
    computed yet.
    """
    management_dir = tmp_path / "management"
    cfg = _make_classification_config(
        management_dir,
        callbacks=[
            {
                "class_path": "tests.integration.test_lightning_cli.CrashAtEpochCallback",
                "init_args": {"crash_at_epoch": 2},
            },
            {
                "class_path": "rslearn.train.callbacks.checkpointing.ManagedBestLastCheckpoint",
                "init_args": {"monitor": "val_accuracy", "mode": "max"},
            },
        ],
    )

    with pytest.raises(RuntimeError, match="Intentional crash"):
        _run_cli_fit(cfg, tmp_path, str(classification_dataset.path), monkeypatch)

    project_dir = management_dir / "test_project" / "test_run"
    assert (project_dir / "last.ckpt").exists(), "last.ckpt should exist"
    assert (project_dir / "best.ckpt").exists(), "best.ckpt should exist"


def test_best_checkpointing_remembers_best_metric_across_resume(
    classification_dataset: Dataset,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checkpointing should be stable across resumes.

    With lr=0 the loss is constant, so best.ckpt is saved once at epoch 0.
    On resume the callback's _best_value must be restored from the checkpoint
    so that it doesn't re-save best.ckpt at a later epoch.
    """
    management_dir = tmp_path / "management"
    project_dir = management_dir / "test_project" / "test_run"

    def make_cfg(crash_at_epoch: int) -> dict:
        cfg = _make_classification_config(
            management_dir,
            callbacks=[
                {
                    "class_path": "tests.integration.test_lightning_cli.CrashAtEpochCallback",
                    "init_args": {"crash_at_epoch": crash_at_epoch},
                },
                {
                    "class_path": "rslearn.train.callbacks.checkpointing.ManagedBestLastCheckpoint",
                    "init_args": {"monitor": "val_loss", "mode": "min"},
                },
            ],
        )
        cfg["trainer"]["max_epochs"] = 3
        return cfg

    # Run 1: train epoch 0, crash at epoch 1.
    cfg1 = make_cfg(crash_at_epoch=1)
    with pytest.raises(RuntimeError, match="Intentional crash"):
        _run_cli_fit(cfg1, tmp_path, str(classification_dataset.path), monkeypatch)

    best_ckpt = torch.load(
        project_dir / "best.ckpt", map_location="cpu", weights_only=False
    )
    assert best_ckpt["epoch"] == 0

    # Run 2: resume from last.ckpt (auto), train epoch 1, crash at epoch 2.
    cfg2 = make_cfg(crash_at_epoch=2)
    with pytest.raises(RuntimeError, match="Intentional crash"):
        _run_cli_fit(cfg2, tmp_path, str(classification_dataset.path), monkeypatch)

    # best.ckpt must still be from epoch 0 (loss is constant with lr=0, so no
    # later epoch should be considered "better").
    best_ckpt = torch.load(
        project_dir / "best.ckpt", map_location="cpu", weights_only=False
    )
    assert best_ckpt["epoch"] == 0, (
        f"best.ckpt was incorrectly overwritten after resume: expected epoch 0, got epoch {best_ckpt['epoch']}"
    )


def test_checkpoint_raises_on_invalid_metric(
    classification_dataset: Dataset,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ManagedBestLastCheckpoint should raise ValueError when the monitored metric doesn't exist."""
    management_dir = tmp_path / "management"
    cfg = _make_classification_config(
        management_dir,
        callbacks=[
            {
                "class_path": "rslearn.train.callbacks.checkpointing.ManagedBestLastCheckpoint",
                "init_args": {"monitor": "nonexistent_metric", "mode": "min"},
            },
        ],
    )

    with pytest.raises(ValueError, match="did not find nonexistent_metric metric"):
        _run_cli_fit(cfg, tmp_path, str(classification_dataset.path), monkeypatch)
