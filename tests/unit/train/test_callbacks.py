"""Tests for rslearn.train.callbacks."""

import os
import pathlib
from unittest.mock import MagicMock

import torch

from rslearn.train.callbacks.checkpointing import (
    BestLastCheckpoint,
)


class TestBestLastCheckpoint:
    def test_saves_last_every_validation(self, tmp_path: pathlib.Path) -> None:
        dirpath = str(tmp_path / "checkpoints")
        cb = BestLastCheckpoint(dirpath=dirpath, monitor="val_loss", mode="min")

        trainer = MagicMock()
        trainer.sanity_checking = False
        trainer.callback_metrics = {"val_loss": torch.tensor(0.5)}
        pl_module = MagicMock()

        cb.on_validation_end(trainer, pl_module)

        trainer.save_checkpoint.assert_any_call(os.path.join(dirpath, "last.ckpt"))

    def test_saves_best_when_metric_improves(self, tmp_path: pathlib.Path) -> None:
        dirpath = str(tmp_path / "checkpoints")
        cb = BestLastCheckpoint(dirpath=dirpath, monitor="val_loss", mode="min")

        trainer = MagicMock()
        trainer.sanity_checking = False
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

        trainer = MagicMock()
        trainer.sanity_checking = False
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
