"""Raster-based classification task."""

from typing import Any, Literal

import numpy.typing as npt
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
)

from rslearn.models.component import FeatureVector, Predictor
from rslearn.train.model_context import (
    ModelContext,
    ModelOutput,
    RasterImage,
    SampleMetadata,
)

from .task import BasicTask


class RasterClassificationTask(BasicTask):
    """A classification task that reads labels from raster data.

    This task converts a raster label map to a single class label by applying
    an aggregation function (max or any). Useful for binary classification
    tasks where the label is "positive if any pixel is positive".
    """

    def __init__(
        self,
        num_classes: int,
        aggregation: Literal["max", "any"] = "max",
        nodata_value: int | None = None,
        enable_f1_metric: bool = False,
        metric_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> None:
        """Initialize a new RasterClassificationTask.

        Args:
            num_classes: the number of classes.
            aggregation: how to aggregate pixel labels to a single class.
                "max" - class = max pixel value (for ordinal classes)
                "any" - class = 1 if any pixel is non-zero, else 0 (for binary)
            nodata_value: optional value to treat as nodata/invalid.
            enable_f1_metric: whether to compute F1 score.
            metric_kwargs: additional arguments for metrics.
            kwargs: additional arguments to pass to BasicTask.
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.aggregation = aggregation
        self.nodata_value = nodata_value
        self.enable_f1_metric = enable_f1_metric
        self.metric_kwargs = metric_kwargs

    def process_inputs(
        self,
        raw_inputs: dict[str, RasterImage],
        metadata: SampleMetadata,
        load_targets: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Processes raster data into a single classification target.

        Args:
            raw_inputs: raster data to process
            metadata: metadata about the patch being read
            load_targets: whether to load the targets or only inputs

        Returns:
            tuple (input_dict, target_dict) where target_dict contains:
                - "class": scalar tensor with the class label
                - "valid": scalar tensor indicating if the example is valid
        """
        if not load_targets:
            return {}, {}

        assert isinstance(raw_inputs["targets"], RasterImage)
        assert raw_inputs["targets"].image.shape[0] == 1
        assert raw_inputs["targets"].image.shape[1] == 1
        labels = raw_inputs["targets"].image[0, 0, :, :].long()

        # Handle nodata
        if self.nodata_value is not None:
            valid_mask = labels != self.nodata_value
            if not valid_mask.any():
                # All pixels are nodata - mark as invalid
                return {}, {
                    "class": torch.tensor(0, dtype=torch.int64),
                    "valid": torch.tensor(0, dtype=torch.float32),
                }
            valid_labels = labels[valid_mask]
        else:
            valid_labels = labels.flatten()

        # Aggregate to single class
        if self.aggregation == "max":
            class_id = torch.amax(valid_labels).item()
        elif self.aggregation == "any":
            # Binary: 1 if any pixel is non-zero, else 0
            class_id = 1 if (valid_labels > 0).any() else 0
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return {}, {
            "class": torch.tensor(class_id, dtype=torch.int64),
            "valid": torch.tensor(1, dtype=torch.float32),
        }

    def process_output(
        self, raw_output: Any, metadata: SampleMetadata
    ) -> npt.NDArray[Any]:
        """Processes model output.

        Args:
            raw_output: the output from prediction head (probabilities).
            metadata: metadata about the patch being read

        Returns:
            numpy array of probabilities
        """
        if isinstance(raw_output, torch.Tensor):
            return raw_output.cpu().numpy()
        return raw_output

    def visualize(
        self,
        input_dict: dict[str, Any],
        target_dict: dict[str, Any] | None,
        output: Any,
    ) -> dict[str, npt.NDArray[Any]]:
        """Visualize is not implemented for classification."""
        return {}

    def get_metrics(self) -> MetricCollection:
        """Returns metrics for this task."""
        kwargs = {"num_classes": self.num_classes}
        kwargs.update(self.metric_kwargs)

        metrics = {
            "Accuracy": MulticlassAccuracy(**kwargs),
        }

        if self.enable_f1_metric:
            metrics["F1"] = MulticlassF1Score(
                num_classes=self.num_classes, average="macro"
            )
            # Per-class F1
            for i in range(self.num_classes):
                metrics[f"F1_cls_{i}"] = PerClassF1(
                    num_classes=self.num_classes, class_idx=i
                )

        return MetricCollection(metrics)


class PerClassF1(MulticlassF1Score):
    """Wrapper to report F1 for a specific class."""

    def __init__(self, num_classes: int, class_idx: int, **kwargs: Any):
        super().__init__(num_classes=num_classes, average="none", **kwargs)
        self.class_idx = class_idx

    def compute(self) -> torch.Tensor:
        f1_per_class = super().compute()
        return f1_per_class[self.class_idx]


class RasterClassificationHead(Predictor):
    """Head for raster classification task with weighted cross-entropy loss."""

    def __init__(self, weights: list[float] | None = None):
        """Initialize a new RasterClassificationHead.

        Args:
            weights: optional weights for cross-entropy loss. Should be a list
                with one weight per class, e.g. [0.05, 0.95] for binary
                classification with class imbalance.
        """
        super().__init__()
        if weights is not None:
            self.register_buffer("weights", torch.tensor(weights))
        else:
            self.weights = None

    def forward(
        self,
        intermediates: Any,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Compute classification outputs and loss.

        Args:
            intermediates: output from previous model component, must be a
                FeatureVector with shape (BatchSize, NumClasses).
            context: the model context.
            targets: must contain "class" key with the class label and "valid" key.

        Returns:
            ModelOutput with outputs and loss dict.
        """
        if not isinstance(intermediates, FeatureVector):
            raise ValueError(
                "input to RasterClassificationHead must be a FeatureVector"
            )

        logits = intermediates.feature_vector
        outputs = torch.nn.functional.softmax(logits, dim=1)

        losses = {}
        if targets:
            class_labels = torch.stack([target["class"] for target in targets], dim=0)
            mask = torch.stack([target["valid"] for target in targets], dim=0)
            loss = (
                torch.nn.functional.cross_entropy(
                    logits, class_labels, weight=self.weights, reduction="none"
                )
                * mask
            )
            losses["cls"] = torch.mean(loss)

        return ModelOutput(
            outputs=outputs,
            loss_dict=losses,
        )
