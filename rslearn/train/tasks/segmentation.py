"""Segmentation task."""

from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torchmetrics.classification
from torchmetrics import Metric, MetricCollection

from rslearn.utils import Feature

from .task import BasicTask

DEFAULT_COLORS = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
    [0, 128, 0],
    [255, 160, 122],
    [139, 69, 19],
    [128, 128, 128],
    [255, 255, 255],
    [143, 188, 143],
    [95, 158, 160],
    [255, 200, 0],
    [128, 0, 0],
]


class SegmentationTask(BasicTask):
    """A segmentation (per-pixel classification) task."""

    def __init__(
        self,
        num_classes: int,
        colors: list[tuple[int, int, int]] = DEFAULT_COLORS,
        **kwargs,
    ):
        """Initialize a new SegmentationTask.

        Args:
            num_classes: the number of classes to predict
            colors: optional colors for each class
            kwargs: additional arguments to pass to BasicTask
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.colors = colors

    def process_inputs(
        self, raw_inputs: dict[str, Union[npt.NDArray[Any], list[Feature]]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Processes the data into targets.

        Args:
            raw_inputs: raster or vector data to process

        Returns:
            tuple (input_dict, target_dict) containing the processed inputs and targets
                that are compatible with both metrics and loss functions
        """
        assert raw_inputs["targets"].shape[0] == 1
        labels = raw_inputs["targets"][0, :, :]
        return {}, {
            "classes": torch.tensor(labels, dtype=torch.int64),
            "valid": torch.ones(labels.shape, dtype=torch.float32),
        }

    def visualize(
        self,
        input_dict: dict[str, Any],
        target_dict: Optional[dict[str, Any]],
        output: Any,
    ) -> dict[str, npt.NDArray[Any]]:
        """Visualize the outputs and targets.

        Args:
            input_dict: the input dict from process_inputs
            target_dict: the target dict from process_inputs
            output: the prediction

        Returns:
            a dictionary mapping image name to visualization image
        """
        image = super().visualize(input_dict, target_dict, output)["image"]
        gt_classes = target_dict["classes"].cpu().numpy()
        pred_classes = output.cpu().numpy().argmax(axis=0)
        gt_vis = np.zeros((gt_classes.shape[0], gt_classes.shape[1], 3), dtype=np.uint8)
        pred_vis = np.zeros(
            (pred_classes.shape[0], pred_classes.shape[1], 3), dtype=np.uint8
        )
        for class_id in range(self.num_classes):
            color = self.colors[class_id % len(self.colors)]
            gt_vis[gt_classes == class_id] = color
            pred_vis[pred_classes == class_id] = color

        return {
            "image": np.array(image),
            "gt": gt_vis,
            "pred": pred_vis,
        }

    def get_metrics(self) -> MetricCollection:
        """Get the metrics for this task."""
        metrics = {}
        metrics["accuracy"] = SegmentationMetric(
            torchmetrics.classification.MulticlassAccuracy(
                num_classes=self.num_classes,
            )
        )
        return MetricCollection(metrics)


class SegmentationHead(torch.nn.Module):
    """Head for segmentation task."""

    def forward(
        self,
        logits: torch.Tensor,
        inputs: list[dict[str, Any]],
        targets: Optional[list[dict[str, Any]]] = None,
    ):
        """Compute the segmentation outputs from logits and targets.

        Args:
            logits: tensor that is (BatchSize, NumClasses, Height, Width) in shape.
            inputs: original inputs (ignored).
            targets: should contain classes key that stores the per-pixel class labels.

        Returns:
            tuple of outputs and loss dict
        """
        outputs = torch.nn.functional.softmax(logits, dim=1)

        loss = None
        if targets:
            labels = torch.stack([target["classes"] for target in targets], dim=0)
            mask = torch.stack([target["valid"] for target in targets], dim=0)
            loss = (
                torch.nn.functional.cross_entropy(logits, labels, reduction="none")
                * mask
            )
            loss = torch.mean(loss)

        return outputs, {"cls": loss}


class SegmentationMetric(Metric):
    """Metric for segmentation task."""

    def __init__(self, metric: Metric):
        """Initialize a new SegmentationMetric."""
        super().__init__()
        self.metric = metric

    def update(self, preds: list[Any], targets: list[dict[str, Any]]) -> None:
        """Update metric.

        Args:
            preds: the predictions
            targets: the targets
        """
        preds = torch.stack(preds)
        labels = torch.stack([target["classes"] for target in targets])

        # Sub-select the valid labels.
        # We flatten the prediction and label images at valid pixels.
        # Prediction is changed from BCHW to BHWC so we can select the valid BHW mask.
        mask = torch.stack([target["valid"] > 0 for target in targets])
        preds = preds.permute(0, 2, 3, 1)[mask]
        labels = labels[mask]
        if len(preds) == 0:
            return

        self.metric.update(preds, labels)

    def compute(self) -> Any:
        """Returns the computed metric."""
        return self.metric.compute()

    def plot(self, *args: list[Any], **kwargs: dict[str, Any]) -> Any:
        """Returns a plot of the metric."""
        return self.metric.plot(*args, **kwargs)
