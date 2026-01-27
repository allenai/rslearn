"""Metric output classes for non-scalar metrics that need special logging."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import wandb


@dataclass
class NonScalarMetricOutput(ABC):
    """Base class for non-scalar metric outputs that need special logging.

    Subclasses should implement the log_to_wandb method to define how the metric
    should be logged to Weights & Biases.
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
        probs: accumulated probability predictions (N, num_classes)
        labels: accumulated ground truth labels (N,)
        class_names: optional list of class names for axis labels
    """

    probs: torch.Tensor
    labels: torch.Tensor
    class_names: list[str] | None = None

    def log_to_wandb(self, name: str) -> None:
        """Log confusion matrix to wandb.

        Args:
            name: the metric name (e.g., "val_confusion_matrix")
        """
        probs = self.probs.detach().cpu().numpy()
        labels = self.labels.detach().cpu().numpy()
        num_classes = probs.shape[1]

        if self.class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        else:
            class_names = self.class_names

        # Log the interactive confusion matrix with probabilities
        wandb.log(
            {
                name: wandb.plot.confusion_matrix(
                    probs=probs,
                    y_true=labels.tolist(),
                    class_names=class_names,
                    title=name,
                ),
            },
        )
