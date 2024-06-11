"""Classification task."""

from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torchmetrics.classification
from PIL import Image, ImageDraw
from torchmetrics import Metric, MetricCollection

from rslearn.utils import Feature

from .task import BasicTask


class ClassificationTask(BasicTask):
    """A window classification task."""

    def __init__(
        self,
        property_name: str,
        classes: list[str],
        filters: Optional[list[tuple[str, str]]],
        read_class_id: bool = False,
        allow_invalid: bool = False,
    ):
        """Initialize a new ClassificationTask.

        Args:
            property_name: the property from which to extract the class name. The class
                is read from the first matching feature.
            classes: a list of class names.
            filters: optional list of (property_name, property_value) to only consider
                features with matching properties.
            read_class_id: whether to read an integer class ID instead of the class
                name.
            allow_invalid: instead of throwing error when no regression label is found
                at a window, simply mark the example invalid for this task
        """
        self.property_name = property_name
        self.classes = classes
        self.filters = filters
        self.read_class_id = read_class_id
        self.allow_invalid = allow_invalid

        if not self.filters:
            self.filters = []

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
        data = raw_inputs["targets"]
        for feat in data:
            for property_name, property_value in self.filters:
                if feat.properties.get(property_name) != property_value:
                    continue
            if self.property_name not in feat.properties:
                continue

            if self.read_class_id:
                class_id = int(feat.properties[self.property_name])
            else:
                class_id = self.classes.index(feat.properties[self.property_name])

            return {}, {
                "class": torch.tensor(class_id, dtype=torch.int64),
                "valid": torch.tensor(1, dtype=torch.float32),
            }

        if not self.allow_invalid:
            raise Exception("no feature found providing class label")

        return {}, {
            "class": torch.tensor(0, dtype=torch.int64),
            "valid": torch.tensor(0, dtype=torch.float32),
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
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        target_class = self.classes[target_dict["value"]]
        output_class = self.classes[output.argmax()]
        text = f"Label: {target_class}\nOutput: {output_class}"
        box = draw.textbbox(xy=(0, 0), text=text, font_size=12)
        draw.rectangle(xy=box, fill=(0, 0, 0))
        draw.text(xy=(0, 0), text=text, font_size=12, fill=(255, 255, 255))
        return {
            "image": np.array(image),
        }

    def get_metrics(self) -> MetricCollection:
        """Get the metrics for this task."""
        return MetricCollection([ClassificationMetric(num_classes=len(self.classes))])


class ClassificationHead(torch.nn.Module):
    """Head for classification task."""

    def forward(
        self, logits: torch.Tensor, targets: Optional[list[dict[str, Any]]] = None
    ):
        """Compute the classification outputs and loss from logits and targets.

        Args:
            logits: tensor that is (BatchSize, NumClasses) in shape.
            targets: should contain class key that stores the class label.

        Returns:
            tuple of outputs and loss dict
        """
        outputs = torch.nn.functional.softmax(logits, dim=1)

        loss = None
        if targets:
            class_labels = torch.stack([target["class"] for target in targets], dim=0)
            mask = torch.stack([target["valid"] for target in targets], dim=0)
            loss = (
                torch.nn.functional.cross_entropy(
                    logits, class_labels, reduction="none"
                )
                * mask
            )
            loss = torch.mean(loss)

        return outputs, {"cls": loss}


class ClassificationMetric(Metric):
    """Metric for regression task."""

    def __init__(self, num_classes: int):
        """Initialize a new ClassificationMetric."""
        super().__init__()
        self.metric = torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes
        )

    def update(self, preds: list[Any], targets: list[dict[str, Any]]) -> None:
        """Update metric.

        Args:
            preds: the predictions
            targets: the targets
        """
        preds = torch.stack(preds)
        labels = torch.stack([target["class"] for target in targets])

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
