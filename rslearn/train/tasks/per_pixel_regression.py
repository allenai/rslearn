"""Per-pixel regression task."""

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
import torchmetrics
from torchmetrics import Metric, MetricCollection

from rslearn.utils.feature import Feature

from .task import BasicTask


class PerPixelRegressionTask(BasicTask):
    """A per-pixel regression task."""

    def __init__(
        self,
        scale_factor: float = 1,
        metric_mode: Literal["mse", "l1"] = "mse",
        nodata_value: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a new PerPixelRegressionTask.

        Args:
            scale_factor: multiply the label value by this factor before using it for
                training.
            metric_mode: what metric to use, either mse or l1
            nodata_value: optional value to treat as invalid
            kwargs: other arguments to pass to BasicTask
        """
        super().__init__(**kwargs)
        self.scale_factor = scale_factor
        self.metric_mode = metric_mode
        self.nodata_value = nodata_value

    def process_inputs(
        self,
        raw_inputs: dict[str, torch.Tensor],
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

        assert raw_inputs["targets"].shape[0] == 1
        labels = raw_inputs["targets"][0, :, :].float() * self.scale_factor

        if self.nodata_value is not None:
            valid = (raw_inputs["targets"][0, :, :] != self.nodata_value).float()
        else:
            valid = torch.ones(labels.shape, dtype=torch.float32)

        return {}, {
            "values": labels,
            "valid": valid,
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
        # Input could be CHW (with single channel) or just HW.
        if len(raw_output.shape) == 2:
            raw_output = raw_output[None, :, :]
        return (raw_output / self.scale_factor).cpu().numpy()

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
        if target_dict is None:
            raise ValueError("target_dict is required for visualization")
        gt_values = target_dict["classes"].cpu().numpy()
        pred_values = output.cpu().numpy()[0, :, :]
        gt_vis = np.clip(gt_values * 255, 0, 255).astype(np.uint8)
        pred_vis = np.clip(pred_values * 255, 0, 255).astype(np.uint8)
        return {
            "image": np.array(image),
            "gt": gt_vis,
            "pred": pred_vis,
        }

    def get_metrics(self) -> MetricCollection:
        """Get the metrics for this task."""
        metric_dict: dict[str, Metric] = {}

        if self.metric_mode == "mse":
            metric_dict["mse"] = PerPixelRegressionMetricWrapper(
                metric=torchmetrics.MeanSquaredError(), scale_factor=self.scale_factor
            )
        elif self.metric_mode == "l1":
            metric_dict["l1"] = PerPixelRegressionMetricWrapper(
                metric=torchmetrics.MeanAbsoluteError(), scale_factor=self.scale_factor
            )

        return MetricCollection(metric_dict)


class PerPixelRegressionHead(torch.nn.Module):
    """Head for per-pixel regression task."""

    def __init__(
        self, loss_mode: Literal["mse", "l1"] = "mse", use_sigmoid: bool = False
    ):
        """Initialize a new RegressionHead.

        Args:
            loss_mode: the loss function to use, either "mse" or "l1".
            use_sigmoid: whether to apply a sigmoid activation on the output. This
                requires targets to be between 0-1.
        """
        super().__init__()

        if loss_mode not in ["mse", "l1"]:
            raise ValueError("invalid loss mode")

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
            logits: BxHxW or BxCxHxW tensor.
            inputs: original inputs (ignored).
            targets: should contain target key that stores the regression labels.

        Returns:
            tuple of outputs and loss dict
        """
        assert len(logits.shape) in [3, 4]
        if len(logits.shape) == 4:
            assert logits.shape[1] == 1
            logits = logits[:, 0, :, :]

        if self.use_sigmoid:
            outputs = torch.nn.functional.sigmoid(logits)
        else:
            outputs = logits

        losses = {}
        if targets:
            labels = torch.stack([target["values"] for target in targets])
            mask = torch.stack([target["valid"] for target in targets])

            if self.loss_mode == "mse":
                scores = torch.square(outputs - labels)
            elif self.loss_mode == "l1":
                scores = torch.abs(outputs - labels)
            else:
                assert False

            # Compute average but only over valid pixels.
            mask_total = mask.sum()
            if mask_total == 0:
                # Just average over all pixels but it will be zero.
                losses["regress"] = (scores * mask).mean()
            else:
                losses["regress"] = (scores * mask).sum() / mask_total

        return outputs, losses


class PerPixelRegressionMetricWrapper(Metric):
    """Metric for per-pixel regression task."""

    def __init__(self, metric: Metric, scale_factor: float, **kwargs: Any) -> None:
        """Initialize a new PerPixelRegressionMetricWrapper.

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
        labels = torch.stack([target["values"] for target in targets])

        # Sub-select the valid labels.
        # We flatten the prediction and label images at valid pixels.
        if len(preds.shape) == 4:
            assert preds.shape[1] == 1
            preds = preds[:, 0, :, :]
        mask = torch.stack([target["valid"] > 0 for target in targets])
        preds = preds[mask]
        labels = labels[mask]
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
