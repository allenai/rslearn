"""Classification task."""

from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torchmetrics
from PIL import Image, ImageDraw
from torchmetrics import Metric

from rslearn.utils import Feature

from .task import BasicTask


class RegressionTask(BasicTask):
    """A window regression task."""

    def __init__(
        self,
        property_name: str,
        filters: Optional[list[tuple[str, str]]],
        allow_invalid: bool = False,
        scale_factor: float = 1,
        **kwargs,
    ):
        """Initialize a new RegressionTask.

        Args:
            property_name: the property from which to extract the regression value. The
                value is read from the first matching feature.
            filters: optional list of (property_name, property_value) to only consider
                features with matching properties.
            allow_invalid: instead of throwing error when no regression label is found
                at a window, simply mark the example invalid for this task
            scale_factor: multiply the label value by this factor
            kwargs: other arguments to pass to BasicTask
        """
        super().__init__(**kwargs)
        self.property_name = property_name
        self.filters = filters
        self.allow_invalid = allow_invalid
        self.scale_factor = scale_factor

        if not self.filters:
            self.filters = []

    def get_target(self, data: Union[npt.NDArray[Any], list[Feature]]) -> Any:
        """Extracts targets from vector data.

        Args:
            data: vector data to process

        Returns:
            the label
        """
        for feat in data:
            for property_name, property_value in self.filters:
                if feat.properties.get(property_name) != property_value:
                    continue
            if self.property_name not in feat.properties:
                continue
            value = float(feat.properties[self.property_name]) * self.scale_factor
            return {
                "value": torch.tensor(value, dtype=torch.float32),
                "valid": torch.tensor(1, dtype=torch.float32),
            }

        if not self.allow_invalid:
            raise Exception("no feature found providing regression label")

        return {
            "value": torch.tensor(0, dtype=torch.float32),
            "valid": torch.tensor(0, dtype=torch.float32),
        }

    def visualize(
        self, input_dict: dict[str, Any], output: Any, target: Optional[Any]
    ) -> dict[str, npt.NDArray[Any]]:
        """Visualize the outputs and targets.

        Args:
            input_dict: the input
            output: the prediction
            target: the target label from get_target

        Returns:
            a dictionary mapping image name to visualization image
        """
        image = super().visualize(input_dict, output, target)["image"]
        image = image.repeat(repeats=8, axis=0).repeat(repeats=8, axis=1)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        target = target["value"] / self.scale_factor
        output = output / self.scale_factor
        text = f"Label: {target:.2f}\nOutput: {output:.2f}"
        box = draw.textbbox(xy=(0, 0), text=text, font_size=12)
        draw.rectangle(xy=box, fill=(0, 0, 0))
        draw.text(xy=(0, 0), text=text, font_size=12, fill=(255, 255, 255))
        return {
            "image": np.array(image),
        }


class RegressionHead(torch.nn.Module):
    """Head for regression task."""

    def __init__(self, loss_mode: str = "mse"):
        """Initialize a new RegressionHead.

        Args:
            loss_mode: the loss function to use, either "mse" or "l1".
        """
        super().__init__()
        self.loss_mode = loss_mode

    def forward(
        self, logits: torch.Tensor, targets: Optional[list[dict[str, Any]]] = None
    ):
        """Compute the regression outputs and loss from logits and targets.

        Args:
            logits: tensor that is (BatchSize, 1) or (BatchSize) in shape.
            targets: should contain target key that stores the regression label.

        Returns:
            tuple of outputs and loss dict
        """
        assert len(logits.shape) in [1, 2]
        if len(logits.shape) == 2:
            assert logits.shape[1] == 1
            logits = logits[:, 0]
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


class RegressionMetric(Metric):
    """Metric for regression task."""

    def __init__(self, mode: str = "mse", **kwargs):
        """Initialize a new RegressionMetric.

        Args:
            mode: either "mse" or "l1"
            kwargs: other arguments to pass to super constructor
        """
        super().__init__(**kwargs)
        if mode == "mse":
            self.metric = torchmetrics.MeanSquaredError()
        elif mode == "l1":
            self.metric = torchmetrics.MeanAbsoluteError()
        else:
            assert False

    def update(self, preds: torch.Tensor, targets: list[dict[str, Any]]) -> None:
        """Update metric.

        Args:
            preds: the predictions
            targets: the targets
        """
        labels = torch.stack([target["value"] for target in targets])

        # Sub-select the valid labels.
        mask = torch.stack([target["valid"] > 0 for target in targets])
        preds = preds[mask]
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
