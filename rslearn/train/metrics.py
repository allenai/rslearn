"""Metric output classes for non-scalar metrics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import wandb
from torchmetrics import Metric

from rslearn.log_utils import get_logger

logger = get_logger(__name__)


@dataclass
class NonScalarMetricOutput(ABC):
    """Base class for non-scalar metric outputs that need special logging.

    Subclasses should implement the log_to_wandb method to define how the metric
    should be logged (only supports logging to Weights & Biases).
    """

    @abstractmethod
    def log_to_wandb(self, name: str) -> None:
        """Log this metric to wandb.

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

    def log_to_wandb(self, name: str) -> None:
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


class ConfusionMatrixMetric(Metric):
    """Confusion matrix metric that works on flattened inputs.

    Expects preds of shape (N, C) and labels of shape (N,).
    Should be wrapped by ClassificationMetric or SegmentationMetric
    which handle the task-specific preprocessing.

    Args:
        num_classes: number of classes
        class_names: optional list of class names for labeling
    """

    def __init__(
        self,
        num_classes: int,
        class_names: list[str] | None = None,
    ):
        """Initialize a new ConfusionMatrixMetric.

        Args:
            num_classes: number of classes
            class_names: optional list of class names for labeling
        """
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names
        self.add_state(
            "confusion_matrix",
            default=torch.zeros(num_classes, num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """Update metric.

        Args:
            preds: predictions of shape (N, C) - probabilities
            labels: ground truth of shape (N,) - class indices
        """
        if len(preds) == 0:
            return

        pred_classes = preds.argmax(dim=1)  # (N,)

        for true_label in range(self.num_classes):
            for pred_label in range(self.num_classes):
                count = ((labels == true_label) & (pred_classes == pred_label)).sum()
                self.confusion_matrix[true_label, pred_label] += count

    def compute(self) -> ConfusionMatrixOutput:
        """Returns the confusion matrix wrapped in ConfusionMatrixOutput."""
        return ConfusionMatrixOutput(
            confusion_matrix=self.confusion_matrix,
            class_names=self.class_names,
        )

    def reset(self) -> None:
        """Reset metric."""
        super().reset()
