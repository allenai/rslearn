"""Metric output classes for non-scalar metrics that need special logging."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

matplotlib.use("Agg")  # Non-interactive backend for server environments


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
        matrix: the confusion matrix tensor (num_classes x num_classes)
        class_names: optional list of class names for axis labels
    """

    matrix: torch.Tensor
    class_names: list[str] | None = None

    def _create_figure(
        self, cm: np.ndarray, class_names: list[str]
    ) -> plt.Figure:
        """Create a matplotlib figure for the confusion matrix.

        Args:
            cm: confusion matrix as numpy array (num_classes x num_classes)
            class_names: list of class names

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

        # Create heatmap
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        # Set ticks and labels
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True label",
            xlabel="Predicted label",
            title="Confusion Matrix",
        )

        # Rotate x labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm[i, j]
                ax.text(
                    j,
                    i,
                    str(int(value)),
                    ha="center",
                    va="center",
                    color="white" if value > thresh else "black",
                    fontsize=8,
                )

        fig.tight_layout()
        return fig

    def log_to_wandb(self, name: str) -> None:
        """Log confusion matrix to wandb.

        Args:
            name: the metric name (e.g., "val_confusion_matrix")
        """
        cm = self.matrix.detach().cpu().numpy()
        num_classes = cm.shape[0]

        if self.class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        else:
            class_names = self.class_names

        # Create matplotlib figure for image logging
        fig = self._create_figure(cm, class_names)

        # Reconstruct preds and y_true from the confusion matrix
        preds = []
        y_true = []
        for true_class in range(num_classes):
            for pred_class in range(num_classes):
                count = int(cm[true_class, pred_class])
                preds.extend([pred_class] * count)
                y_true.extend([true_class] * count)

        # Log both the interactive confusion matrix and the image
        wandb.log(
            {
                name: wandb.plot.confusion_matrix(
                    preds=preds,
                    y_true=y_true,
                    class_names=class_names,
                    title=name,
                ),
                f"{name}_image": wandb.Image(fig),
            },
        )

        plt.close(fig)
