"""Tests for rslearn.train.callbacks."""

import os
import pathlib
from unittest.mock import MagicMock

import torch
from lightning.pytorch.trainer.states import TrainerFn

from rslearn.train.callbacks.checkpointing import (
    BestLastCheckpoint,
)


def _make_trainer(fn: TrainerFn = TrainerFn.FITTING) -> MagicMock:
    """Create a mock trainer whose save_checkpoint writes a small file."""
    trainer = MagicMock()
    trainer.sanity_checking = False
    trainer.is_global_zero = True
    trainer.state.fn = fn

    def save_checkpoint(path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"ckpt")

    trainer.save_checkpoint.side_effect = save_checkpoint
    return trainer


class TestBestLastCheckpoint:
    def test_saves_last_every_validation(self, tmp_path: pathlib.Path) -> None:
        dirpath = str(tmp_path / "checkpoints")
        cb = BestLastCheckpoint(dirpath=dirpath, monitor="val_loss", mode="min")

        trainer = _make_trainer()
        trainer.callback_metrics = {"val_loss": torch.tensor(0.5)}
        pl_module = MagicMock()

        cb.on_validation_end(trainer, pl_module)

        last_path = os.path.join(dirpath, "last.ckpt")
        trainer.save_checkpoint.assert_any_call(last_path + ".tmp")
        assert os.path.exists(last_path)
        assert not os.path.exists(last_path + ".tmp")

    def test_saves_best_when_metric_improves(self, tmp_path: pathlib.Path) -> None:
        dirpath = str(tmp_path / "checkpoints")
        cb = BestLastCheckpoint(dirpath=dirpath, monitor="val_loss", mode="min")

        trainer = _make_trainer()
        pl_module = MagicMock()

        # First call: val_loss=0.5, should save best
        trainer.callback_metrics = {"val_loss": torch.tensor(0.5)}
        cb.on_validation_end(trainer, pl_module)
        assert trainer.save_checkpoint.call_count == 2  # last + best

        trainer.save_checkpoint.reset_mock()

        # Second call: val_loss=0.3 (improved), should save best
        trainer.callback_metrics = {"val_loss": torch.tensor(0.3)}
        cb.on_validation_end(trainer, pl_module)
        assert trainer.save_checkpoint.call_count == 2  # last + best

        trainer.save_checkpoint.reset_mock()

        # Third call: val_loss=0.4 (worse), should NOT save best
        trainer.callback_metrics = {"val_loss": torch.tensor(0.4)}
        cb.on_validation_end(trainer, pl_module)
        assert trainer.save_checkpoint.call_count == 1  # only last

    def test_max_mode(self, tmp_path: pathlib.Path) -> None:
        dirpath = str(tmp_path / "checkpoints")
        cb = BestLastCheckpoint(dirpath=dirpath, monitor="val_acc", mode="max")

        trainer = _make_trainer()
        pl_module = MagicMock()

        # First: 0.8 -> save best
        trainer.callback_metrics = {"val_acc": torch.tensor(0.8)}
        cb.on_validation_end(trainer, pl_module)
        assert trainer.save_checkpoint.call_count == 2

        trainer.save_checkpoint.reset_mock()

        # Second: 0.7 (worse for max) -> no best
        trainer.callback_metrics = {"val_acc": torch.tensor(0.7)}
        cb.on_validation_end(trainer, pl_module)
        assert trainer.save_checkpoint.call_count == 1

        trainer.save_checkpoint.reset_mock()

        # Third: 0.9 (better) -> save best
        trainer.callback_metrics = {"val_acc": torch.tensor(0.9)}
        cb.on_validation_end(trainer, pl_module)
        assert trainer.save_checkpoint.call_count == 2

    def test_skips_save_outside_fit(self, tmp_path: pathlib.Path) -> None:
        dirpath = str(tmp_path / "checkpoints")
        cb = BestLastCheckpoint(dirpath=dirpath, monitor="val_loss", mode="min")

        trainer = _make_trainer(fn=TrainerFn.VALIDATING)
        trainer.callback_metrics = {"val_loss": torch.tensor(0.5)}
        cb.on_validation_end(trainer, MagicMock())

        assert trainer.save_checkpoint.call_count == 0
        assert not os.path.exists(os.path.join(dirpath, "last.ckpt"))

    def test_existing_last_kept_if_save_fails(self, tmp_path: pathlib.Path) -> None:
        dirpath = tmp_path / "checkpoints"
        dirpath.mkdir()
        last_path = dirpath / "last.ckpt"
        last_path.write_bytes(b"previous")
        cb = BestLastCheckpoint(dirpath=str(dirpath), monitor="val_loss", mode="min")

        trainer = _make_trainer()
        trainer.callback_metrics = {"val_loss": torch.tensor(0.5)}

        def failing_save(path: str) -> None:
            with open(path, "wb") as f:
                f.write(b"part")
            raise RuntimeError("preempted")

        trainer.save_checkpoint.side_effect = failing_save
        try:
            cb.on_validation_end(trainer, MagicMock())
        except RuntimeError:
            pass

        assert last_path.read_bytes() == b"previous"

    def test_remote_path_saved_directly(self) -> None:
        cb = BestLastCheckpoint(dirpath="gs://bucket/run", monitor="val_loss")

        trainer = MagicMock()
        trainer.sanity_checking = False
        trainer.state.fn = TrainerFn.FITTING
        trainer.callback_metrics = {"val_loss": torch.tensor(0.5)}
        cb.on_validation_end(trainer, MagicMock())

        trainer.save_checkpoint.assert_any_call("gs://bucket/run/last.ckpt")
        trainer.save_checkpoint.assert_any_call("gs://bucket/run/best.ckpt")
