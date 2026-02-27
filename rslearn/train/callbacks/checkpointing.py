"""Checkpoint callbacks for rslearn."""

import os
from typing import Literal

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

from rslearn.log_utils import get_logger

logger = get_logger(__name__)


class BestLastCheckpoint(Callback):
    """Saves both last.ckpt (every validation) and best.ckpt (when metric improves).

    Replaces the common pattern of configuring two separate ModelCheckpoint callbacks.
    Uses trainer.save_checkpoint() which handles distributed saving and cloud paths.
    """

    def __init__(
        self,
        dirpath: str,
        monitor: str = "val_loss",
        mode: Literal["min", "max"] = "min",
    ) -> None:
        """Create a new BestLastCheckpoint.

        Args:
            dirpath: directory to save checkpoints in (local or cloud path).
            monitor: metric key to monitor for best checkpoint.
            mode: "min" if lower is better, "max" if higher is better.
        """
        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self._best_value: float | None = None
        self._is_sanity_check = False

    def on_sanity_check_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Mark sanity check as active."""
        self._is_sanity_check = True

    def on_sanity_check_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Mark sanity check as inactive."""
        self._is_sanity_check = False

    @rank_zero_only
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Save last.ckpt always, and best.ckpt when the monitored metric improves."""
        if self._is_sanity_check:
            return

        os.makedirs(self.dirpath, exist_ok=True)

        last_path = os.path.join(self.dirpath, "last.ckpt")
        trainer.save_checkpoint(last_path)
        logger.info("saved checkpoint to %s", last_path)

        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            logger.warning(
                "monitored metric '%s' not found in callback_metrics, "
                "skipping best checkpoint",
                self.monitor,
            )
            return

        current_val = current.item()
        is_better = self._best_value is None or (
            (self.mode == "min" and current_val < self._best_value)
            or (self.mode == "max" and current_val > self._best_value)
        )
        if is_better:
            self._best_value = current_val
            best_path = os.path.join(self.dirpath, "best.ckpt")
            trainer.save_checkpoint(best_path)
            logger.info(
                "saved best checkpoint to %s (%s=%s)",
                best_path,
                self.monitor,
                current_val,
            )


class ManagedBestLastCheckpoint(BestLastCheckpoint):
    """BestLastCheckpoint that resolves dirpath from trainer.default_root_dir.

    Use with project management: the CLI sets trainer.default_root_dir to the
    project directory, and this callback picks it up at setup time.
    """

    def __init__(
        self, monitor: str = "val_loss", mode: Literal["min", "max"] = "min"
    ) -> None:
        """Create a new ManagedBestLastCheckpoint.

        Args:
            monitor: metric key to monitor for best checkpoint.
            mode: "min" if lower is better, "max" if higher is better.
        """
        super().__init__(dirpath="", monitor=monitor, mode=mode)

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: str | None = None
    ) -> None:
        """Resolve dirpath from trainer.default_root_dir."""
        self.dirpath = trainer.default_root_dir
        logger.info("ManagedBestLastCheckpoint using dirpath=%s", self.dirpath)
