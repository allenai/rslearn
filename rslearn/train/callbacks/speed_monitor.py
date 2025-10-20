"""Speed monitoring callback for tracking data loading and forward pass times."""

import time
from typing import Any

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer import Trainer
from torch.nn import Module

from rslearn.log_utils import get_logger

logger = get_logger(__name__)

THROUGHPUT_PREFIX = "throughput"

class SpeedMonitor(Callback):
    """Monitor time spent on data loading and forward pass.

    Tracks the time spent on:
    - Data loading: Time from end of previous batch to start of current batch
    - Forward pass: Time from batch start to before backward pass

    Logs both absolute times and percentages to wandb.
    """

    def __init__(self) -> None:
        """Initialize the callback."""
        self.batch_end_time: float | None = None
        self.batch_start_time: float | None = None
        self.forward_end_time: float | None = None
        self.data_loading_time: float | None = None

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: Module, batch: Any, batch_idx: int
    ) -> None:
        """Mark the start of batch processing (end of data loading).

        Args:
            trainer: The trainer object.
            pl_module: The module object.
            batch: The current batch.
            batch_idx: The batch index.
        """
        self.batch_start_time = time.time()

        # Calculate data loading time if we have previous batch end time
        if self.batch_end_time is not None:
            self.data_loading_time = self.batch_start_time - self.batch_end_time
            pl_module.log(
                f"{THROUGHPUT_PREFIX}/data_loading_time",
                self.data_loading_time,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
        else:
            self.data_loading_time = None

    def on_before_backward(self, trainer: Trainer, pl_module: Module, loss: Any) -> None:
        """Mark the end of forward pass (before backward).

        Args:
            trainer: The trainer object.
            pl_module: The module object.
            loss: The loss value.
        """
        self.forward_end_time = time.time()

        # Calculate forward pass time
        if self.batch_start_time is not None:
            forward_time = self.forward_end_time - self.batch_start_time
            pl_module.log(
                f"{THROUGHPUT_PREFIX}/forward_pass_time",
                forward_time,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: Module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Mark the end of batch processing and compute total time and percentages.

        Args:
            trainer: The trainer object.
            pl_module: The module object.
            outputs: The outputs from the training step.
            batch: The current batch.
            batch_idx: The batch index.
        """
        self.batch_end_time = time.time()

        # Calculate total batch time and percentages
        if self.batch_start_time is not None and self.forward_end_time is not None:
            # Time breakdown
            forward_time = self.forward_end_time - self.batch_start_time
            backward_optimizer_time = self.batch_end_time - self.forward_end_time
            total_time = self.batch_end_time - self.batch_start_time

            # Log total time
            pl_module.log(
                f"{THROUGHPUT_PREFIX}/total_batch_time",
                total_time,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

            # Calculate and log percentages
            if total_time > 0:
                forward_percent = (forward_time / total_time) * 100
                backward_optimizer_percent = (backward_optimizer_time / total_time) * 100

                pl_module.log(
                    f"{THROUGHPUT_PREFIX}/forward_pass_percent",
                    forward_percent,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                )
                pl_module.log(
                    f"{THROUGHPUT_PREFIX}/backward_optimizer_percent",
                    backward_optimizer_percent,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                )

            # Log data loading percentage if available
            if self.data_loading_time is not None:
                total_time_with_loading = self.data_loading_time + total_time
                if total_time_with_loading > 0:
                    data_loading_percent = (self.data_loading_time / total_time_with_loading) * 100
                    pl_module.log(
                        "speed/data_loading_percent",
                        data_loading_percent,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=False,
                    )
