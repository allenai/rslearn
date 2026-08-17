"""Metric output classes for non-scalar metrics."""

from dataclasses import dataclass
from typing import Any

import torch
import wandb
from torchmetrics import Metric
from torchmetrics.classification import MulticlassConfusionMatrix

from rslearn.log_utils import get_logger

logger = get_logger(__name__)


@dataclass
class NonScalarMetricOutput:
    """Base class for non-scalar metric outputs that need special logging.

    Subclasses should implement the platform-specific methods for the loggers they
    support. Unsupported loggers are skipped with a warning.
    """

    def log_to_wandb(self, name: str) -> None:
        """Log this metric to wandb.

        Args:
            name: the metric name
        """
        logger.warning("W&B logging is not implemented for metric %s", name)

    def log_to_mlflow(self, name: str, client: Any, run_id: str) -> None:
        """Log this metric to MLflow.

        Args:
            name: the metric name.
            client: the MLflow client associated with the Lightning logger.
            run_id: the MLflow run ID.
        """
        logger.warning("MLflow logging is not implemented for metric %s", name)


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

    def log_to_mlflow(self, name: str, client: Any, run_id: str) -> None:
        """Log the confusion matrix as MLflow table and figure artifacts.

        Args:
            name: the metric name (e.g., ``val_confusion_matrix``).
            client: the MLflow client associated with the Lightning logger.
            run_id: the MLflow run ID.
        """
        cm = self.confusion_matrix.detach().cpu()
        if cm.dim() > 2:
            cm = cm.sum(dim=0)

        num_classes = cm.shape[0]
        class_names = self.class_names or [str(i) for i in range(num_classes)]
        data: dict[str, list[Any]] = {
            "true_class": [],
            "predicted_class": [],
            "count": [],
        }
        for true_label in range(cm.shape[0]):
            for pred_label in range(cm.shape[1]):
                data["true_class"].append(class_names[true_label])
                data["predicted_class"].append(class_names[pred_label])
                data["count"].append(int(cm[true_label, pred_label].item()))

        client.log_table(
            run_id=run_id,
            data=data,
            artifact_file=f"{name}.json",
        )

        # TorchMetrics provides the confusion-matrix visualization; MLflow stores the
        # resulting Matplotlib figure as an artifact.
        plotter = MulticlassConfusionMatrix(num_classes=num_classes)
        figure, axes = plotter.plot(val=cm, labels=class_names)
        # Import lazily because Matplotlib is only required by the optional MLflow path.
        from matplotlib import pyplot as plt

        try:
            axes.set_title(name)
            client.log_figure(
                run_id=run_id,
                figure=figure,
                artifact_file=f"{name}.png",
            )
        finally:
            plt.close(figure)


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
