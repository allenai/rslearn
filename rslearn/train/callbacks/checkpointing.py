"""Checkpoint callbacks for rslearn."""

import os
from typing import Any, Literal

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer.states import TrainerFn

from rslearn.log_utils import get_logger

logger = get_logger(__name__)


def _is_local_path(path: str) -> bool:
    """Return whether path refers to the local filesystem."""
    return "://" not in path or path.startswith("file://")


def _save_checkpoint_atomic(trainer: Trainer, path: str) -> None:
    """Save a checkpoint so that a crash never leaves a partial file at path.

    For local paths, Lightning stages the checkpoint in the system temp directory and
    then moves it into place. When the temp directory is on a different filesystem
    (e.g. /tmp vs a network filesystem), the move becomes a copy, so a job that is
    preempted mid-copy leaves a truncated checkpoint behind, and auto-resume then fails
    to load it. Here we instead save to a temporary file next to the target and rename
    it over the target, which is atomic within a directory.

    Remote paths (e.g. gs://) are saved directly since object writes are already atomic.
    """
    if not _is_local_path(path):
        trainer.save_checkpoint(path)
        return

    local_path = path.removeprefix("file://")
    tmp_path = f"{local_path}.tmp"
    # trainer.save_checkpoint writes on rank 0 and ends with a barrier across ranks.
    trainer.save_checkpoint(tmp_path)
    if trainer.is_global_zero:
        os.replace(tmp_path, local_path)


class BestLastCheckpoint(Callback):
    """Saves both last.ckpt (every validation) and best.ckpt (when metric improves)."""

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

    def state_dict(self) -> dict[str, Any]:
        """Persist best value so it survives checkpoint resume."""
        return {"best_value": self._best_value}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore best value from checkpoint on resume."""
        self._best_value = state_dict.get("best_value")

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save last.ckpt always, and best.ckpt when the monitored metric improves."""
        if trainer.sanity_checking:
            return
        # Only save during fit. Standalone validation (e.g. `model validate` after
        # training) must not overwrite last.ckpt/best.ckpt with a checkpoint that does
        # not carry the fit loop state, since a later auto-resume would then restart
        # training from the beginning.
        if trainer.state.fn != TrainerFn.FITTING:
            return

        # Check metric value to handle best.ckpt.
        if self.monitor not in trainer.callback_metrics:
            raise ValueError(
                f"did not find {self.monitor} metric (keys are {trainer.callback_metrics.keys()})"
            )
        current_val = trainer.callback_metrics[self.monitor].item()

        is_better = self._best_value is None or (
            (self.mode == "min" and current_val < self._best_value)
            or (self.mode == "max" and current_val > self._best_value)
        )
        if is_better:
            self._best_value = current_val
            best_path = os.path.join(self.dirpath, "best.ckpt")
            _save_checkpoint_atomic(trainer, best_path)

            # Previously this entire function was decorated with @rank_zero_only, but this resulted
            # in NCCL timeout because all ranks must call trainer.save_checkpoint. Log only on rank 0.
            if trainer.is_global_zero:
                logger.info(
                    "saved best checkpoint to %s (%s=%s)",
                    best_path,
                    self.monitor,
                    current_val,
                )

        # Save last.ckpt after best.ckpt so that auto-resuming from last.ckpt is always
        # correct.
        # - If we crash after writing best.ckpt, we will auto-resume from previous epoch
        #   and write the same best.ckpt again.
        # - If we crash after writing last.ckpt, we will restore the correct
        #   self._best_value and won't override the checkpoint incorrectly.
        last_path = os.path.join(self.dirpath, "last.ckpt")
        _save_checkpoint_atomic(trainer, last_path)
        if trainer.is_global_zero:
            logger.info("saved checkpoint to %s", last_path)


class ManagedBestLastCheckpoint(BestLastCheckpoint):
    """BestLastCheckpoint that resolves dirpath from trainer.default_root_dir.

    Use with project management: the CLI sets trainer.default_root_dir to the project
    directory ({MANAGEMENT_DIR}/{PROJECT_NAME}/{RUN_NAME}), and this callback picks it
    up at setup time.
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
        if trainer.is_global_zero:
            logger.info("ManagedBestLastCheckpoint using dirpath=%s", self.dirpath)
