"""Per-pixel regression task."""

import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
import torchmetrics
from torchmetrics import Metric, MetricCollection

from rslearn.models.component import FeatureMaps, Predictor
from rslearn.train.model_context import (
    ModelContext,
    ModelOutput,
    RasterImage,
    SampleMetadata,
)
from rslearn.utils.feature import Feature

from .task import BasicTask


class PerPixelRegressionTask(BasicTask):
    """A per-pixel regression task."""

    def __init__(
        self,
        scale_factor: float = 1,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        metric_mode: (
            Literal["mse", "rmse", "l1", "r2", "mape"]
            | Sequence[Literal["mse", "rmse", "l1", "r2", "mape"]]
            | None
        ) = None,
        nodata_value: float | None = None,
        metrics: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a new PerPixelRegressionTask.

        The target can be rescaled before training in one of two mutually exclusive ways:

        - ``scale_factor``: simple multiplicative scaling (``value * scale_factor``).
        - ``target_mean`` / ``target_std``: z-normalization
          (``(value - target_mean) / target_std``).

        In all cases predictions are mapped back to the original units in
        ``process_output``, and metrics are computed on the reconstructed values (see
        ``PerPixelRegressionMetricWrapper``), so reported numbers remain in the original
        units. The defaults (scale_factor=1, mean=0, std=1) leave the targets unchanged.

        Args:
            scale_factor: multiply ground truth values by this factor before using it for
                training. Mutually exclusive with target_mean/target_std.
            target_mean: mean subtracted from the targets for normalization. Mutually
                exclusive with scale_factor.
            target_std: standard deviation the (mean-subtracted) targets are divided by
                for normalization. Must be positive. Mutually exclusive with scale_factor.
            metric_mode: deprecated; use metrics instead. Will be removed after
                2026-06-01.
            nodata_value: optional value to treat as invalid. The loss will be masked
                at pixels where the ground truth value is equal to nodata_value.
            metrics: metric(s) to compute. Supported values: "mse", "rmse", "l1",
                "r2", "mape".
            kwargs: other arguments to pass to BasicTask
        """
        super().__init__(**kwargs)
        if target_std <= 0:
            raise ValueError(f"target_std must be positive, but got {target_std}")
        uses_scale_factor = scale_factor != 1
        uses_normalization = target_mean != 0.0 or target_std != 1.0
        if uses_scale_factor and uses_normalization:
            raise ValueError(
                "scale_factor and target_mean/target_std are mutually exclusive; "
                "use one or the other (normalization is preferred)."
            )
        self.scale_factor = scale_factor
        self.target_mean = target_mean
        self.target_std = target_std

        if metrics is not None:
            metric_names = list(metrics)
            if metric_mode is not None:
                warnings.warn(
                    "PerPixelRegressionTask.metric_mode is deprecated and ignored when "
                    "`metrics` is set. It will be removed after 2026-06-01.",
                    FutureWarning,
                    stacklevel=2,
                )
        elif metric_mode is not None:
            warnings.warn(
                "PerPixelRegressionTask.metric_mode is deprecated; use `metrics` "
                "instead. It will be removed after 2026-06-01.",
                FutureWarning,
                stacklevel=2,
            )
            if isinstance(metric_mode, str):
                metric_names = [metric_mode]
            else:
                metric_names = list(metric_mode)
        else:
            metric_names = ["mse"]

        if len(metric_names) == 0:
            raise ValueError("metrics must contain at least one metric")
        allowed = {"mse", "rmse", "l1", "r2", "mape"}
        invalid = [m for m in metric_names if m not in allowed]
        if invalid:
            raise ValueError(f"invalid metrics entries: {invalid}")
        self.metrics = metric_names
        self.nodata_value = nodata_value

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

        assert isinstance(raw_inputs["targets"], RasterImage)
        raw_labels = raw_inputs["targets"].get_hw_tensor()
        labels = (raw_labels.float() * self.scale_factor - self.target_mean) / (
            self.target_std
        )

        window_valid = self._get_window_valid_mask(labels, metadata)
        if self.nodata_value is not None:
            valid = (raw_labels != self.nodata_value).float() * window_valid
        else:
            valid = window_valid

        # Wrap in RasterImage with CTHW format (C=1, T=1) so values and valid can be
        # used in image transforms.
        return {}, {
            "values": RasterImage(labels[None, None, :, :], timestamps=None),
            "valid": RasterImage(valid[None, None, :, :], timestamps=None),
        }

    def process_output(
        self, raw_output: Any, metadata: SampleMetadata
    ) -> npt.NDArray[Any] | list[Feature]:
        """Processes an output into raster or vector data.

        Args:
            raw_output: the output from prediction head, which must be an HW tensor.
            metadata: metadata about the patch being read

        Returns:
            either raster or vector data.
        """
        if not isinstance(raw_output, torch.Tensor):
            raise ValueError("output for PerPixelRegressionTask must be a tensor")
        if len(raw_output.shape) != 2:
            raise ValueError(
                f"PerPixelRegressionTask output must be an HW tensor, but got shape {raw_output.shape}"
            )
        reconstructed = (
            raw_output[None, :, :] * self.target_std + self.target_mean
        ) / self.scale_factor
        return reconstructed.cpu().numpy()

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
        gt_values = target_dict["values"].get_hw_tensor().cpu().numpy()
        pred_values = output.cpu().numpy()[0, :, :]
        # De-normalize back to original target units before rendering.
        gt_values = (gt_values * self.target_std + self.target_mean) / self.scale_factor
        pred_values = (
            pred_values * self.target_std + self.target_mean
        ) / self.scale_factor
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

        metric_kwargs = dict(
            scale_factor=self.scale_factor,
            target_mean=self.target_mean,
            target_std=self.target_std,
        )
        for metric_name in self.metrics:
            if metric_name == "mse":
                metric_dict["mse"] = PerPixelRegressionMetricWrapper(
                    metric=torchmetrics.MeanSquaredError(),
                    **metric_kwargs,
                )
            elif metric_name == "rmse":
                metric_dict["rmse"] = PerPixelRegressionMetricWrapper(
                    metric=torchmetrics.MeanSquaredError(squared=False),
                    **metric_kwargs,
                )
            elif metric_name == "l1":
                metric_dict["l1"] = PerPixelRegressionMetricWrapper(
                    metric=torchmetrics.MeanAbsoluteError(),
                    **metric_kwargs,
                )
            elif metric_name == "r2":
                metric_dict["r2"] = PerPixelRegressionMetricWrapper(
                    metric=torchmetrics.R2Score(),
                    **metric_kwargs,
                )
            elif metric_name == "mape":
                metric_dict["mape"] = PerPixelRegressionMetricWrapper(
                    metric=torchmetrics.MeanAbsolutePercentageError(),
                    **metric_kwargs,
                )
            else:
                raise ValueError(f"unknown metric {metric_name}")

        return MetricCollection(metric_dict)


class PerPixelRegressionHead(Predictor):
    """Head for per-pixel regression task."""

    def __init__(
        self,
        loss_mode: Literal["mse", "l1", "huber"] = "mse",
        use_sigmoid: bool = False,
        huber_delta: float = 1.0,
    ):
        """Initialize a new PerPixelRegressionHead.

        Args:
            loss_mode: the loss function to use: "mse" (default), "l1", or "huber".
            use_sigmoid: whether to apply a sigmoid activation on the output. This
                requires targets to be between 0-1.
            huber_delta: delta parameter for Huber loss (only used when
                loss_mode="huber").
        """
        super().__init__()

        if loss_mode not in ["mse", "l1", "huber"]:
            raise ValueError(f"invalid loss mode {loss_mode}")

        self.loss_mode = loss_mode
        self.use_sigmoid = use_sigmoid
        self.huber_delta = huber_delta

    def forward(
        self,
        intermediates: Any,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Compute the regression outputs and loss from logits and targets.

        Args:
            intermediates: output from previous component, which must be a FeatureMaps
                with one feature map corresponding to the logits. The channel dimension
                size must be 1.
            context: the model context.
            targets: must contain values key that stores the regression labels, and
                valid key containing mask image indicating where the labels are valid.

        Returns:
            tuple of outputs and loss dict. The output is a BHW tensor so that the
                per-sample output is an HW tensor.
        """
        if not isinstance(intermediates, FeatureMaps):
            raise ValueError(
                "the input to PerPixelRegressionHead must be a FeatureMaps"
            )
        if len(intermediates.feature_maps) != 1:
            raise ValueError(
                "the input to PerPixelRegressionHead must have one feature map"
            )
        if intermediates.feature_maps[0].shape[1] != 1:
            raise ValueError(
                f"the input to PerPixelRegressionHead must have channel dimension size 1, but got {intermediates.feature_maps[0].shape}"
            )

        logits = intermediates.feature_maps[0][:, 0, :, :]

        if self.use_sigmoid:
            outputs = torch.nn.functional.sigmoid(logits)
        else:
            outputs = logits

        losses = {}
        if targets:
            labels = torch.stack(
                [target["values"].get_hw_tensor() for target in targets]
            )
            mask = torch.stack([target["valid"].get_hw_tensor() for target in targets])

            if self.loss_mode == "mse":
                scores = torch.square(outputs - labels)
            elif self.loss_mode == "l1":
                scores = torch.abs(outputs - labels)
            elif self.loss_mode == "huber":
                scores = torch.nn.functional.huber_loss(
                    outputs,
                    labels,
                    reduction="none",
                    delta=self.huber_delta,
                )
            else:
                raise ValueError(f"unknown loss mode {self.loss_mode}")

            # Compute average but only over valid pixels.
            mask_total = mask.sum()
            if mask_total == 0:
                # Just average over all pixels but it will be zero.
                losses["regress"] = (scores * mask).mean()
            else:
                losses["regress"] = (scores * mask).sum() / mask_total

        return ModelOutput(
            outputs=outputs,
            loss_dict=losses,
        )


class PerPixelRegressionMetricWrapper(Metric):
    """Metric for per-pixel regression task."""

    def __init__(
        self,
        metric: Metric,
        scale_factor: float,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new PerPixelRegressionMetricWrapper.

        Both predictions and targets arrive in the normalized training space. They are
        de-normalized back to the original units before the underlying metric is updated,
        so reported metrics (e.g. RMSE) are in the original target units rather than the
        scaled/normalized space.

        Args:
            metric: the underlying torchmetric to apply, which should accept a flat
                tensor of predicted values followed by a flat tensor of target values
            scale_factor: scale factor to undo so that metric is based on original
                values
            target_mean: mean to add back when de-normalizing to original units.
            target_std: standard deviation to multiply by when de-normalizing to
                original units.
            kwargs: other arguments to pass to super constructor
        """
        super().__init__(**kwargs)
        self.metric = metric
        self.scale_factor = scale_factor
        self.target_mean = target_mean
        self.target_std = target_std

    def _to_original_units(self, values: torch.Tensor) -> torch.Tensor:
        """De-normalize values from the training space back to the original units."""
        return (values * self.target_std + self.target_mean) / self.scale_factor

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
        labels = torch.stack([target["values"].get_hw_tensor() for target in targets])

        # Sub-select the valid labels.
        # We flatten the prediction and label images at valid pixels.
        if len(preds.shape) == 4:
            assert preds.shape[1] == 1
            preds = preds[:, 0, :, :]
        mask = torch.stack([target["valid"].get_hw_tensor() > 0 for target in targets])
        preds = preds[mask]
        labels = labels[mask]
        if len(preds) == 0:
            return

        # De-normalize both back to original units so the metric is reported in the
        # original target space rather than the normalized training space.
        preds = self._to_original_units(preds)
        labels = self._to_original_units(labels)

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
