"""Regression task."""

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import shapely
import torch
import torchmetrics
from PIL import Image, ImageDraw
from torchmetrics import Metric, MetricCollection

from rslearn.models.component import FeatureVector, Predictor
from rslearn.train.model_context import ModelContext, ModelOutput, SampleMetadata
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import STGeometry
from rslearn.utils.raster_format import RasterImage

from .task import BasicTask


class RegressionTask(BasicTask):
    """A window regression task."""

    def __init__(
        self,
        property_name: str,
        filters: list[tuple[str, str]] | None = None,
        allow_invalid: bool = False,
        scale_factor: float = 1,
        metric_mode: Literal["mse", "l1"] = "mse",
        use_accuracy_metric: bool = False,
        within_factor: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initialize a new RegressionTask.

        Args:
            property_name: the property from which to extract the ground truth
                regression value. The value is read from the first matching feature.
            filters: optional list of (property_name, property_value) to only consider
                features with matching properties.
            allow_invalid: instead of throwing error when no regression label is found
                at a window, simply mark the example invalid for this task
            scale_factor: multiply the label value by this factor for training
            metric_mode: what metric to use, either "mse" (default) or "l1"
            use_accuracy_metric: include metric that reports percentage of
                examples where output is within a factor of the ground truth.
            within_factor: the factor for accuracy metric. If it's 0.2, and ground
                truth is 5.0, then values from 5.0*0.8 to 5.0*1.2 are accepted.
            kwargs: other arguments to pass to BasicTask
        """
        super().__init__(**kwargs)
        self.property_name = property_name
        self.filters = filters
        self.allow_invalid = allow_invalid
        self.scale_factor = scale_factor
        self.metric_mode = metric_mode
        self.use_accuracy_metric = use_accuracy_metric
        self.within_factor = within_factor

        if not self.filters:
            self.filters = []

    def process_inputs(
        self,
        raw_inputs: dict[str, RasterImage | list[Feature]],
        metadata: SampleMetadata,
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
        assert isinstance(data, list)
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

    def process_output(
        self, raw_output: Any, metadata: SampleMetadata
    ) -> list[Feature]:
        """Processes an output into raster or vector data.

        Args:
            raw_output: the output from prediction head, which must be a scalar tensor.
            metadata: metadata about the patch being read

        Returns:
            a list with a single Feature corresponding to the patch extent and with a
                property containing the predicted value.
        """
        if not isinstance(raw_output, torch.Tensor) or len(raw_output.shape) != 0:
            raise ValueError("output for RegressionTask must be a scalar Tensor")

        output = raw_output.item() / self.scale_factor
        feature = Feature(
            STGeometry(
                metadata.projection,
                shapely.Point(metadata.patch_bounds[0], metadata.patch_bounds[1]),
                None,
            ),
            {
                self.property_name: output,
            },
        )
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
        metric_dict: dict[str, Metric] = {}

        if self.metric_mode == "mse":
            metric_dict["mse"] = RegressionMetricWrapper(
                metric=torchmetrics.MeanSquaredError(), scale_factor=self.scale_factor
            )
        elif self.metric_mode == "l1":
            metric_dict["l1"] = RegressionMetricWrapper(
                metric=torchmetrics.MeanAbsoluteError(), scale_factor=self.scale_factor
            )

        if self.use_accuracy_metric:
            metric_dict["accuracy"] = RegressionMetricWrapper(
                metric=RegressionAccuracy(self.within_factor),
                scale_factor=self.scale_factor,
            )

        return MetricCollection(metric_dict)


class RegressionHead(Predictor):
    """Head for regression task."""

    def __init__(
        self, loss_mode: Literal["mse", "l1"] = "mse", use_sigmoid: bool = False
    ):
        """Initialize a new RegressionHead.

        Args:
            loss_mode: the loss function to use, either "mse" (default) or "l1".
            use_sigmoid: whether to apply a sigmoid activation on the output. This
                requires targets to be between 0-1.
        """
        super().__init__()
        self.loss_mode = loss_mode
        self.use_sigmoid = use_sigmoid

    def forward(
        self,
        intermediates: Any,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Compute the regression outputs and loss from logits and targets.

        Args:
            intermediates: output from previous model component, which must be a
                FeatureVector with channel dimension size 1 (Bx1).
            context: the model context.
            targets: target dicts, which each must contain a "value" key containing the
                regression label, along with a "valid" key containing a flag indicating
                whether each example is valid for this task.

        Returns:
            the model outputs. The output is a B tensor so that it is split up into a
                scalar for each example.
        """
        if not isinstance(intermediates, FeatureVector):
            raise ValueError("the input to RegressionHead must be a FeatureVector")
        if intermediates.feature_vector.shape[1] != 1:
            raise ValueError(
                f"the input to RegressionHead must have channel dimension size 1, but got shape {intermediates.feature_vector.shape}"
            )

        logits = intermediates.feature_vector[:, 0]

        if self.use_sigmoid:
            outputs = torch.nn.functional.sigmoid(logits)
        else:
            outputs = logits

        losses = {}
        if targets:
            labels = torch.stack([target["value"] for target in targets])
            mask = torch.stack([target["valid"] for target in targets])
            if self.loss_mode == "mse":
                losses["regress"] = torch.mean(torch.square(outputs - labels) * mask)
            elif self.loss_mode == "l1":
                losses["regress"] = torch.mean(torch.abs(outputs - labels) * mask)
            else:
                raise ValueError(f"unknown loss mode {self.loss_mode}")

        return ModelOutput(
            outputs=outputs,
            loss_dict=losses,
        )


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


class RegressionAccuracy(Metric):
    """Percentage of examples with estimate within some factor of ground truth."""

    def __init__(self, factor: float) -> None:
        """Initialize a new RegressionAccuracy.

        Args:
            factor: the factor so if estimate is within this much of ground truth then
                it is marked correct.
        """
        super().__init__()
        self.factor = factor
        self.correct = 0
        self.total = 0

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """Update metric.

        Args:
            preds: the predictions
            labels: the ground truth data
        """
        decisions = (preds >= labels * (1 - self.factor)) & (
            preds <= labels * (1 + self.factor)
        )
        self.correct += torch.count_nonzero(decisions)
        self.total += len(decisions)

    def compute(self) -> Any:
        """Returns the computed metric."""
        return torch.tensor(self.correct / self.total)

    def reset(self) -> None:
        """Reset metric."""
        super().reset()
        self.correct = 0
        self.total = 0

    def plot(self, *args: list[Any], **kwargs: dict[str, Any]) -> Any:
        """Returns a plot of the metric."""
        return None
