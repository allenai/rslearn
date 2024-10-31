"""Classification task."""

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torchmetrics
from PIL import Image, ImageDraw
from torchmetrics import Metric, MetricCollection

from rslearn.utils import Feature

from .task import BasicTask


class RegressionTask(BasicTask):
    """A window regression task."""

    def __init__(
        self,
        property_name: str,
        filters: list[tuple[str, str]] | None,
        allow_invalid: bool = False,
        scale_factor: float = 1,
        metric_mode: str = "mse",
        **kwargs: Any,
    ) -> None:
        """Initialize a new RegressionTask.

        Args:
            property_name: the property from which to extract the regression value. The
                value is read from the first matching feature.
            filters: optional list of (property_name, property_value) to only consider
                features with matching properties.
            allow_invalid: instead of throwing error when no regression label is found
                at a window, simply mark the example invalid for this task
            scale_factor: multiply the label value by this factor
            metric_mode: what metric to use, either mse or l1
            kwargs: other arguments to pass to BasicTask
        """
        super().__init__(**kwargs)
        self.property_name = property_name
        self.filters = filters
        self.allow_invalid = allow_invalid
        self.scale_factor = scale_factor
        self.metric_mode = metric_mode

        if not self.filters:
            self.filters = []

    def process_inputs(
        self,
        raw_inputs: dict[str, torch.Tensor | list[Feature]],
        metadata: dict[str, Any],
        load_targets: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Processes the data into targets.

        Args:
            raw_inputs: raster or vector data to process
            metadata: metadata about the patch being read
            load_targets: whether to load the targets or only inputs

        Returns:
            tuple (input_dict, target_dict) containing the processed inputs and targets
                that are compatible with both metrics and loss functions
        """
        if not load_targets:
            return {}, {}

        data = raw_inputs["targets"]
        for feat in data:
            if feat.properties is None or self.filters is None:
                continue
            for property_name, property_value in self.filters:
                if feat.properties.get(property_name) != property_value:
                    continue
            if self.property_name not in feat.properties:
                continue
            value = float(feat.properties[self.property_name]) * self.scale_factor
            return {}, {
                "value": torch.tensor(value, dtype=torch.float32),
                "valid": torch.tensor(1, dtype=torch.float32),
            }

        if not self.allow_invalid:
            raise Exception("no feature found providing regression label")

        return {}, {
            "value": torch.tensor(0, dtype=torch.float32),
            "valid": torch.tensor(0, dtype=torch.float32),
        }

    def visualize(
        self,
        input_dict: dict[str, Any],
        target_dict: dict[str, Any] | None,
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
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        if target_dict is None:
            raise ValueError("target_dict is required for visualization")
        target = target_dict["value"] / self.scale_factor
        output = output / self.scale_factor
        text = f"Label: {target:.2f}\nOutput: {output:.2f}"
        box = draw.textbbox(xy=(0, 0), text=text, font_size=12)
        draw.rectangle(xy=box, fill=(0, 0, 0))
        draw.text(xy=(0, 0), text=text, font_size=12, fill=(255, 255, 255))
        return {
            "image": np.array(image),
        }

    def get_metrics(self) -> MetricCollection:
        """Get the metrics for this task."""
        if self.metric_mode == "mse":
            metric = torchmetrics.MeanSquaredError()
        elif self.metric_mode == "l1":
            metric = torchmetrics.MeanAbsoluteError()
        return MetricCollection(
            {
                self.metric_mode: RegressionMetricWrapper(
                    metric=metric, scale_factor=self.scale_factor
                )
            }
        )


class RegressionHead(torch.nn.Module):
    """Head for regression task."""

    def __init__(self, loss_mode: str = "mse", use_sigmoid: bool = False):
        """Initialize a new RegressionHead.

        Args:
            loss_mode: the loss function to use, either "mse" or "l1".
            use_sigmoid: whether to apply a sigmoid activation on the output. This
                requires targets to be between 0-1.
        """
        super().__init__()
        self.loss_mode = loss_mode
        self.use_sigmoid = use_sigmoid

    def forward(
        self,
        logits: torch.Tensor,
        inputs: list[dict[str, Any]],
        targets: list[dict[str, Any]] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute the regression outputs and loss from logits and targets.

        Args:
            logits: tensor that is (BatchSize, 1) or (BatchSize) in shape.
            inputs: original inputs (ignored).
            targets: should contain target key that stores the regression label.

        Returns:
            tuple of outputs and loss dict
        """
        assert len(logits.shape) in [1, 2]
        if len(logits.shape) == 2:
            assert logits.shape[1] == 1
            logits = logits[:, 0]

        if self.use_sigmoid:
            outputs = torch.nn.functional.sigmoid(logits)
        else:
            outputs = logits

        loss = None
        if targets:
            labels = torch.stack([target["value"] for target in targets])
            mask = torch.stack([target["valid"] for target in targets])
            if self.loss_mode == "mse":
                loss = torch.mean(torch.square(outputs - labels) * mask)
            elif self.loss_mode == "l1":
                loss = torch.mean(torch.abs(outputs - labels) * mask)
            else:
                assert False

        return outputs, {"regress": loss}


class RegressionMetricWrapper(Metric):
    """Metric for regression task."""

    def __init__(self, metric: Metric, scale_factor: float, **kwargs: Any) -> None:
        """Initialize a new RegressionMetricWrapper.

        Args:
            metric: the underlying torchmetric to apply, which should accept a flat
                tensor of predicted values followed by a flat tensor of target values
            scale_factor: scale factor to undo so that metric is based on original
                values
            kwargs: other arguments to pass to super constructor
        """
        super().__init__(**kwargs)
        self.metric = metric
        self.scale_factor = scale_factor

    def update(self, preds: list[Any], targets: list[dict[str, Any]]) -> None:
        """Update metric.

        Args:
            preds: the predictions
            targets: the targets
        """
        preds = torch.stack(preds)
        labels = torch.stack([target["value"] for target in targets])

        # Sub-select the valid labels.
        mask = torch.stack([target["valid"] > 0 for target in targets])
        preds = preds[mask] / self.scale_factor
        labels = labels[mask] / self.scale_factor
        if len(preds) == 0:
            return

        self.metric.update(preds, labels)

    def compute(self) -> Any:
        """Returns the computed metric."""
        return self.metric.compute()

    def reset(self) -> None:
        """Reset metric."""
        super().reset()
        self.metric.reset()

    def plot(self, *args: list[Any], **kwargs: dict[str, Any]) -> Any:
        """Returns a plot of the metric."""
        return self.metric.plot(*args, **kwargs)
