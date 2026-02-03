"""Metric output classes for non-scalar metrics that need special logging."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import wandb

from rslearn.log_utils import get_logger

logger = get_logger(__name__)


@dataclass
class NonScalarMetricOutput(ABC):
    """Base class for non-scalar metric outputs that need special logging.

    Subclasses should implement the _do_log_to_wandb method to define how the metric
    should be logged to Weights & Biases.
    """

    def log_to_wandb(self, name: str) -> None:
        """Log this metric to wandb.

        Args:
            name: the metric name
        """
        if not wandb.run:
            logger.warning(f"wandb is not initialized, skipping {name}")
            return
        self._do_log_to_wandb(name)

    @abstractmethod
    def _do_log_to_wandb(self, name: str) -> None:
        """Subclass hook to perform the actual wandb logging.

        Args:
            name: the metric name
        """
        pass


@dataclass
class ConfusionMatrixOutput(NonScalarMetricOutput):
    """Confusion matrix metric output.

    Args:
        confusion_matrix: confusion matrix of shape (num_classes, num_classes)
            where cm[i, j] is the count of samples with true label i and predicted
            label j.
        class_names: optional list of class names for axis labels
    """

    confusion_matrix: torch.Tensor
    class_names: list[str] | None = None

    def _expand_confusion_matrix(self) -> tuple[list[int], list[int]]:
        """Expand confusion matrix to (preds, labels) pairs for wandb.

        Returns:
            Tuple of (preds, labels) as lists of integers.
        """
        cm = self.confusion_matrix.detach().cpu()

        # Handle extra dimensions from distributed reduction
        if cm.dim() > 2:
            cm = cm.sum(dim=0)

        total = cm.sum().item()
        if total == 0:
            return [], []

        preds = []
        labels = []
        for true_label in range(cm.shape[0]):
            for pred_label in range(cm.shape[1]):
                count = cm[true_label, pred_label].item()
                if count > 0:
                    preds.extend([pred_label] * int(count))
                    labels.extend([true_label] * int(count))

        return preds, labels

    def _do_log_to_wandb(self, name: str) -> None:
        """Log confusion matrix to wandb.

        Args:
            name: the metric name (e.g., "val_confusion_matrix")
        """
        preds, labels = self._expand_confusion_matrix()

        if len(preds) == 0:
            logger.warning(f"No samples to log for {name}")
            return

        num_classes = self.confusion_matrix.shape[0]
        if self.class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        else:
            class_names = self.class_names

        wandb.log(
            {
                name: wandb.plot.confusion_matrix(
                    preds=preds,
                    y_true=labels,
                    class_names=class_names,
                    title=name,
                ),
            },
        )
