"""Classification task."""

from typing import Any, Optional, Union

import numpy.typing as npt
import torch
import torchmetrics
from torchmetrics import Metric

from rslearn.utils import Feature

from .task import Task


class RegressionTask(Task):
    """A window regression task."""

    def __init__(
        self,
        property_name: str,
        filters: Optional[list[tuple[str, str]]],
        allow_invalid: bool = False,
    ):
        """Initialize a new RegressionTask.

        Args:
            property_name: the property from which to extract the regression value. The
                value is read from the first matching feature.
            filters: optional list of (property_name, property_value) to only consider
                features with matching properties.
            allow_invalid: instead of throwing error when no regression label is found
                at a window, simply mark the example invalid for this task
        """
        self.property_name = property_name
        self.filters = filters
        self.allow_invalid = allow_invalid

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
            value = float(feat.properties[self.property_name]) / 400
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
        outputs = torch.nn.functional.sigmoid(logits)

        loss = None
        if targets:
            labels = torch.stack([target["value"] for target in targets])
            mask = torch.stack([target["valid"] for target in targets])
            print(outputs, labels)
            if self.loss_mode == "mse":
                loss = torch.mean(torch.square(outputs - labels) * mask)
                print(
                    logits[0],
                    outputs[0],
                    loss,
                    torch.square(outputs[0] - labels[0]) * mask[0],
                )
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
