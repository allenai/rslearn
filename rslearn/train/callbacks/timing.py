"""Timing callback for measuring prediction speed."""

import time
from typing import Any

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from rslearn.log_utils import get_logger

logger = get_logger(__name__)


class PredictionTimingCallback(Callback):
    """Callback to time prediction forward passes.

    This callback measures the time taken for each prediction batch and reports
    statistics at the end. It skips the first N batches to allow for warmup,
    which is especially important when using torch.compile.
    """

    def __init__(self, skip_first_n: int = 5):
        """Initialize timing callback.

        Args:
            skip_first_n: Number of initial batches to skip for warmup.
                When using torch.compile with max-autotune, consider setting
                this to 10 or higher as compilation happens on first batches.
        """
        self.skip_first_n = skip_first_n
        self.times: list[float] = []
        self.batch_count = 0
        self.start_time: float = 0.0

    def on_predict_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Record start time before prediction batch."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Record end time after prediction batch."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start_time
        self.batch_count += 1

        if self.batch_count > self.skip_first_n:
            self.times.append(elapsed)

        # Log progress every 100 batches
        if self.batch_count % 100 == 0:
            if self.times:
                avg_ms = sum(self.times) / len(self.times) * 1000
                logger.info(
                    f"Batch {self.batch_count}: running avg {avg_ms:.2f}ms/batch"
                )

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Report timing statistics at the end of prediction."""
        if self.times:
            avg_ms = sum(self.times) / len(self.times) * 1000
            min_ms = min(self.times) * 1000
            max_ms = max(self.times) * 1000
            total_batches = len(self.times)
            throughput = 1000 / avg_ms if avg_ms > 0 else 0

            logger.info(
                f"\n{'=' * 60}\n"
                f"Prediction Timing Summary\n"
                f"{'=' * 60}\n"
                f"  Warmup batches skipped: {self.skip_first_n}\n"
                f"  Measured batches: {total_batches}\n"
                f"  Average time: {avg_ms:.2f} ms/batch\n"
                f"  Min time: {min_ms:.2f} ms/batch\n"
                f"  Max time: {max_ms:.2f} ms/batch\n"
                f"  Throughput: {throughput:.2f} batches/sec\n"
                f"{'=' * 60}"
            )
        else:
            logger.warning(
                f"No timing data collected. Only {self.batch_count} batches were run, "
                f"but {self.skip_first_n} were skipped for warmup."
            )
