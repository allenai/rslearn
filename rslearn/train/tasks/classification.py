"""Classification task."""

from typing import Any

import numpy as np
import numpy.typing as npt
import shapely
import torch
from PIL import Image, ImageDraw
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from rslearn.utils import Feature, STGeometry

from .task import BasicTask


class ClassificationTask(BasicTask):
    """A window classification task."""

    def __init__(
        self,
        property_name: str,
        classes: list,  # TODO: Should this be a list of str or int or can it be both?
        filters: list[tuple[str, str]] = [],
        read_class_id: bool = False,
        allow_invalid: bool = False,
        skip_unknown_categories: bool = False,
        prob_property: str | None = None,
        metric_kwargs: dict[str, Any] = {},
        enable_f1_metric: bool = False,
        f1_metric_kwargs: dict[str, Any] = {},
        positive_class: str | None = None,
        positive_class_threshold: float = 0.5,
        **kwargs: Any,
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
            skip_unknown_categories: whether to skip examples with categories that are
                not passed via classes, instead of throwing error
            prob_property: when predicting, write probabilities in addition to class ID
                under this property name.
            metric_kwargs: additional arguments to pass to underlying metric, see
                torchmetrics.classification.MulticlassAccuracy.
            enable_f1_metric: whether to compute F1 (default false)
            f1_metric_kwargs: extra arguments to pass to F1 metric.
            positive_class: positive class name.
            positive_class_threshold: threshold for classifying the positive class in
                binary classification (default 0.5).
            kwargs: other arguments to pass to BasicTask
        """
        super().__init__(**kwargs)
        self.property_name = property_name
        self.classes = classes
        self.filters = filters
        self.read_class_id = read_class_id
        self.allow_invalid = allow_invalid
        self.skip_unknown_categories = skip_unknown_categories
        self.prob_property = prob_property
        self.metric_kwargs = metric_kwargs
        self.enable_f1_metric = enable_f1_metric
        self.f1_metric_kwargs = f1_metric_kwargs
        self.positive_class = positive_class
        self.positive_class_threshold = positive_class_threshold

        if self.positive_class_threshold != 0.5:
            # Must be binary classification
            if len(self.classes) != 2:
                raise ValueError(
                    "positive_class_threshold is only applicable for binary classification."
                )
            if positive_class is None:
                raise ValueError(
                    "Please set positive_class when using positive_class_threshold."
                )
            elif self.positive_class not in self.classes:
                raise ValueError(
                    f"Positive class '{self.positive_class}' not found in classes {self.classes}."
                )
            else:
                self.positive_class_id = self.classes.index(self.positive_class)

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
            if feat.properties is None:
                continue
            for property_name, property_value in self.filters:
                if feat.properties.get(property_name) != property_value:
                    continue
            if self.property_name not in feat.properties:
                continue

            v = feat.properties[self.property_name]
            if self.read_class_id:
                class_id = int(v)
            else:
                if v in self.classes:
                    class_id = self.classes.index(v)
                else:
                    class_id = -1

            if class_id < 0 or class_id >= len(self.classes):
                # Throw error if unknown categories are not acceptable.
                assert self.skip_unknown_categories
                # Otherwise, skip this example.
                continue

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

    def process_output(
        self, raw_output: Any, metadata: dict[str, Any]
    ) -> npt.NDArray[Any] | list[Feature]:
        """Processes an output into raster or vector data.

        Args:
            raw_output: the output from prediction head.
            metadata: metadata about the patch being read

        Returns:
            either raster or vector data.
        """
        probs = raw_output.cpu().numpy()
        if len(self.classes) == 2 and self.positive_class_threshold != 0.5:
            positive_class_prob = probs[self.positive_class_id]
            if positive_class_prob >= self.positive_class_threshold:
                class_idx = self.positive_class_id
            else:
                class_idx = 1 - self.positive_class_id
        else:
            # For multiclass classification or when using the default threshold
            class_idx = probs.argmax()

        if not self.read_class_id:
            value = self.classes[class_idx]  # type: ignore
        else:
            value = class_idx

        feature = Feature(
            STGeometry(
                metadata["projection"],
                shapely.Point(metadata["bounds"][0], metadata["bounds"][1]),
                None,
            ),
            {
                self.property_name: value,
            },
        )
        if self.prob_property is not None and feature.properties is not None:
            feature.properties[self.prob_property] = probs.tolist()
        return [feature]

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
        target_class = self.classes[target_dict["class"]]
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
        metrics = {}
        metric_kwargs = {"num_classes": len(self.classes)}
        metric_kwargs.update(self.metric_kwargs)
        metrics["accuracy"] = ClassificationMetric(MulticlassAccuracy(**metric_kwargs))

        if self.enable_f1_metric:
            kwargs = {
                "num_classes": len(self.classes),
                "average": None,
            }  # Default to per-class metrics
            kwargs.update(self.f1_metric_kwargs)

            # Check the 'average' parameter
            average_mode = kwargs.get("average", None)
            if average_mode in (None, "none"):
                # Generate per-class metrics with class-specific names
                for class_id, class_name in enumerate(self.classes):
                    metrics[f"{class_name}_recall"] = ClassificationMetric(
                        MulticlassRecall(**kwargs), class_id
                    )
                    metrics[f"{class_name}_precision"] = ClassificationMetric(
                        MulticlassPrecision(**kwargs), class_id
                    )
                    metrics[f"{class_name}_f1"] = ClassificationMetric(
                        MulticlassF1Score(**kwargs), class_id
                    )
            else:
                metrics["recall"] = ClassificationMetric(MulticlassRecall(**kwargs))
                metrics["precision"] = ClassificationMetric(
                    MulticlassPrecision(**kwargs)
                )
                metrics["f1"] = ClassificationMetric(MulticlassF1Score(**kwargs))

        return MetricCollection(metrics)


class ClassificationHead(torch.nn.Module):
    """Head for classification task."""

    def forward(
        self,
        logits: torch.Tensor,
        inputs: list[dict[str, Any]],
        targets: list[dict[str, Any]] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the classification outputs and loss from logits and targets.

        Args:
            logits: tensor that is (BatchSize, NumClasses) in shape.
            inputs: original inputs (ignored).
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
    """Metric for classification task."""

    def __init__(self, metric: Metric, class_idx: int | None = None):
        """Initialize a new ClassificationMetric.

        Args:
            metric: the metric to wrap
            class_idx: optional class index to return metric value for. If set, the
                metric should return a dict from class index to metric value.
        """
        super().__init__()
        self.metric = metric
        self.class_idx = class_idx

    def update(
        self, preds: list[Any] | torch.Tensor, targets: list[dict[str, Any]]
    ) -> None:
        """Update metric.

        Args:
            preds: the predictions
            targets: the targets
        """
        if not isinstance(preds, torch.Tensor):
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
        result = self.metric.compute()
        if self.class_idx is None:
            return result
        else:
            return result[self.class_idx]

    def reset(self) -> None:
        """Reset metric."""
        super().reset()
        self.metric.reset()

    def plot(self, *args: list[Any], **kwargs: dict[str, Any]) -> Any:
        """Returns a plot of the metric."""
        return self.metric.plot(*args, **kwargs)
