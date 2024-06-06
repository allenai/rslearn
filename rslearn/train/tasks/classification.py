"""Classification task."""

from typing import Any, Optional, Union

import numpy.typing as npt
import torch

from rslearn.utils import Feature

from .task import Task


class ClassificationTask(Task):
    """A window classification task."""

    def __init__(
        self,
        property_name: str,
        classes: list[str],
        filters: Optional[list[tuple[str, str]]],
    ):
        """Initialize a new ClassificationTask.

        Args:
            property_name: the property from which to extract the class name. The class
                is read from the first matching feature.
            classes: a list of class names.
            filters: optional list of (property_name, property_value) to only consider
                features with matching properties.
        """
        self.property_name = property_name
        self.classes = classes
        self.filters = filters

        if not self.filters:
            self.filters = []

    def get_target(self, data: Union[npt.NDArray[Any], list[Feature]]) -> Any:
        """Extracts classification targets from vector data.

        Args:
            data: vector data to process

        Returns:
            the category label
        """
        for feat in data:
            for property_name, property_value in self.filters:
                if feat.properties.get(property_name) != property_value:
                    continue
            if self.property_name not in feat.properties:
                continue
            idx = self.classes.index(feat.properties[self.property_name])
            return torch.tensor(idx, dtype=torch.int64)
        raise Exception("no feature found providing class label")


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
        outputs = None
        if not self.training:
            outputs = torch.nn.functional.softmax(logits, dim=1)

        loss = None
        if targets:
            class_labels = torch.stack([target["class"] for target in targets], dim=0)
            loss = torch.nn.functional.cross_entropy(
                logits, class_labels, reduction="none"
            )

        return outputs, {"cls": loss}
