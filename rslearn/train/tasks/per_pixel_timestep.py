"""Per-pixel timestep prediction, with labels and outputs as days since the Unix epoch.

The model classifies each pixel into one of the input timesteps (e.g. the image at
which a change occurred), like a SegmentationTask whose classes are the timesteps.
Labels are dates (uint16 days since 1970-01-01 UTC), which the head maps to an input
timestep once the final input timestamps are known. At prediction time, the predicted
timestep is written as the days since 1970-01-01 of the midpoint of that image's time
range, so that neither the labels nor the output depend on which images the model was
given.
"""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torchmetrics import Metric, MetricCollection

from rslearn.models.component import FeatureMaps
from rslearn.train.model_context import (
    ModelContext,
    ModelOutput,
    RasterImage,
    SampleMetadata,
)
from rslearn.utils import Feature

from .segmentation import SegmentationHead, SegmentationTask

UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

# Value used for pixels without a label or prediction.
TIMESTAMP_NODATA_VALUE = np.iinfo(np.uint16).max

# Largest day that can be written without colliding with nodata (2149-06-05).
MAX_DAYS = TIMESTAMP_NODATA_VALUE - 1


class DateToTimestepMode(StrEnum):
    """How to map a labeled date to one of the input timesteps."""

    BEFORE = "BEFORE"
    """The latest timestep on or before the labeled date."""

    AFTER = "AFTER"
    """The earliest timestep on or after the labeled date."""


def _midpoint_days(time_range: tuple[datetime, datetime]) -> int:
    """Days since the Unix epoch of the midpoint of a timezone-aware time range."""
    start, end = time_range
    midpoint = start + (end - start) / 2
    days = (midpoint - UNIX_EPOCH) // timedelta(days=1)
    if days < 0 or days > MAX_DAYS:
        raise ValueError(
            f"time range midpoint {midpoint} cannot be represented as uint16 days since 1970-01-01"
        )
    return days


def days_to_timesteps(
    label_days: torch.Tensor,
    valid: torch.Tensor,
    timestep_days: torch.Tensor,
    mode: DateToTimestepMode,
    fallback_to_closest_timestep_if_no_match: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map per-pixel labeled dates to input timestep indices.

    Dates are compared at day granularity, so a timestep on the same day as the label
    matches in both modes. If several timesteps qualify with the same day, the first
    one is used.

    Args:
        label_days: HW tensor of labeled days since 1970-01-01.
        valid: HW mask of pixels that have a label.
        timestep_days: T tensor with the days since 1970-01-01 of each timestep.
        mode: whether to pick the latest timestep on or before the labeled day, or
            the earliest one on or after it.
        fallback_to_closest_timestep_if_no_match: if no timestep is on the requested
            side of the labeled day, use the closest timestep instead of marking the
            pixel invalid.

    Returns:
        tuple (classes, valid) of the HW long timestep indices (0 at invalid pixels)
            and the HW float mask of pixels with a valid timestep target.
    """
    label_days = label_days.long()
    valid = valid.float()
    if len(timestep_days) == 0:
        return torch.zeros_like(label_days), torch.zeros_like(valid)

    days = timestep_days.to(device=label_days.device, dtype=torch.long)[:, None, None]
    if mode == DateToTimestepMode.BEFORE:
        candidates = days <= label_days[None]
        classes = torch.where(candidates, days, torch.iinfo(torch.long).min).argmax(
            dim=0
        )
    else:
        candidates = days >= label_days[None]
        classes = torch.where(candidates, days, torch.iinfo(torch.long).max).argmin(
            dim=0
        )

    matched = candidates.any(dim=0)
    if fallback_to_closest_timestep_if_no_match:
        closest = (days - label_days[None]).abs().argmin(dim=0)
        classes = torch.where(matched, classes, closest)
    else:
        valid = valid * matched

    classes = torch.where(valid > 0, classes, 0)
    return classes, valid


class PerPixelTimestepHead(SegmentationHead):
    """Head for PerPixelTimestepTask.

    It is meant to be applied over per-timestep logits at each pixel. During training
    and evaluation, it maps the labeled date at each pixel to an input timestep (see
    days_to_timesteps), and then behaves like SegmentationHead. It additionally
    attaches the date of each input timestep so that PerPixelTimestepTask can write
    dates instead of timestep indices.

    The logit channels must correspond to the timesteps of the input selected by the
    input_key in order. Each per-example output is a dict with "probs" (CHW softmax
    probabilities) and "timestamps" (int64 tensor with the number of days since
    1970-01-01 of the midpoint of each of the T input timesteps). When targets are
    given, it also has "targets", a dict with the timestep index "classes" and "valid"
    mask used for the loss, which PerPixelTimestepTask uses for metrics. Channels
    beyond T, e.g. from padding tokens, are ignored by PerPixelTimestepTask.
    """

    def __init__(
        self,
        input_key: str,
        mode: DateToTimestepMode,
        fallback_to_closest_timestep_if_no_match: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create a new PerPixelTimestepHead.

        Args:
            input_key: the key in the input dict of the RasterImage whose timestamps
                the logit channels correspond to.
            mode: whether the target of each labeled date is the latest timestep on or
                before it (BEFORE) or the earliest timestep on or after it (AFTER).
            fallback_to_closest_timestep_if_no_match: if no timestep is on the
                requested side of a labeled date, use the closest timestep as the
                target instead of excluding the pixel from the loss.
            kwargs: other arguments to pass to SegmentationHead.
        """
        super().__init__(**kwargs)
        self.input_key = input_key
        self.mode = DateToTimestepMode(mode)
        self.fallback_to_closest_timestep_if_no_match = (
            fallback_to_closest_timestep_if_no_match
        )

    def _get_timestep_days(
        self, inputs: dict[str, Any], num_channels: int, device: torch.device
    ) -> torch.Tensor:
        """Get the days since 1970-01-01 of each timestep of the input."""
        image = inputs.get(self.input_key)
        if not isinstance(image, RasterImage):
            raise ValueError(
                f"PerPixelTimestepHead expected a RasterImage at input key '{self.input_key}'"
            )
        if image.timestamps is None:
            raise ValueError(
                f"input '{self.input_key}' has no timestamps, which PerPixelTimestepHead requires"
            )
        if len(image.timestamps) > num_channels:
            raise ValueError(
                f"input '{self.input_key}' has {len(image.timestamps)} timesteps "
                f"but there are only {num_channels} logit channels"
            )
        return torch.tensor(
            [_midpoint_days(time_range) for time_range in image.timestamps],
            dtype=torch.int64,
            device=device,
        )

    def forward(
        self,
        intermediates: Any,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Compute timestep probabilities and attach the input timestamps.

        Args:
            intermediates: a FeatureMaps with a single feature map containing one
                logit per timestep.
            context: the model context.
            targets: optional targets from PerPixelTimestepTask, each containing
                "days" (labeled days since 1970-01-01) and "valid".

        Returns:
            ModelOutput whose outputs are per-example dicts with "probs" and
            "timestamps", and "targets" if targets were given.
        """
        if (
            not isinstance(intermediates, FeatureMaps)
            or len(intermediates.feature_maps) != 1
        ):
            raise ValueError(
                "input to PerPixelTimestepHead must be a FeatureMaps with one feature map"
            )
        logits = intermediates.feature_maps[0]
        timestep_days = [
            self._get_timestep_days(inputs, logits.shape[1], logits.device)
            for inputs in context.inputs
        ]

        timestep_targets: list[dict[str, Any]] | None = None
        if targets:
            timestep_targets = []
            for target, example_days in zip(targets, timestep_days):
                classes, valid = days_to_timesteps(
                    target["days"].get_hw_tensor(),
                    target["valid"].get_hw_tensor(),
                    example_days,
                    self.mode,
                    self.fallback_to_closest_timestep_if_no_match,
                )
                timestep_targets.append(
                    {
                        "classes": RasterImage(classes[None, None], timestamps=None),
                        "valid": RasterImage(valid[None, None], timestamps=None),
                    }
                )

        model_output = super().forward(intermediates, context, timestep_targets)
        probs = model_output.outputs
        assert isinstance(probs, torch.Tensor)

        outputs = []
        for i, (example_probs, example_days) in enumerate(zip(probs, timestep_days)):
            output = {"probs": example_probs, "timestamps": example_days}
            if timestep_targets is not None:
                output["targets"] = timestep_targets[i]
            outputs.append(output)

        return ModelOutput(outputs=outputs, loss_dict=model_output.loss_dict)


class PerPixelTimestepTask(SegmentationTask):
    """Predict an input timestep at each pixel, with dates as labels and outputs.

    The target raster must contain the labeled date at each pixel as the number of
    days since 1970-01-01 (UTC), with 65535 (2^16 - 1) at pixels without a label. It
    must be paired with PerPixelTimestepHead, which maps each labeled date to an input
    timestep. Training and metrics are then the same as SegmentationTask, with
    num_classes set to the number of input timesteps.

    At prediction time, the per-pixel argmax timestep is mapped to the number of days
    since 1970-01-01 (UTC) of the midpoint of that input image's time range, so the
    output layer's band set should use the uint16 dtype. 65535 is used as the nodata
    value; set nodata_value: 65535 on the output band set so that the written raster
    is tagged with it.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Create a new PerPixelTimestepTask.

        Args:
            kwargs: arguments to pass to SegmentationTask. output_probs, prob_scales,
                nodata_value, zero_is_invalid, and class_id_mapping are not supported.
        """
        super().__init__(**kwargs)
        if self.output_probs:
            raise ValueError("PerPixelTimestepTask does not support output_probs")
        if self.prob_scales is not None:
            raise ValueError("PerPixelTimestepTask does not support prob_scales")
        if self.nodata_value is not None:
            raise ValueError(
                "PerPixelTimestepTask does not support nodata_value or zero_is_invalid "
                f"(unlabeled pixels must be {TIMESTAMP_NODATA_VALUE})"
            )
        if self.class_id_mapping is not None:
            raise ValueError("PerPixelTimestepTask does not support class_id_mapping")

    def process_inputs(
        self,
        raw_inputs: Mapping[str, RasterImage | list[Feature]],
        metadata: SampleMetadata,
        load_targets: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read the labeled days since 1970-01-01 at each pixel.

        Args:
            raw_inputs: raster or vector data to process
            metadata: metadata about the patch being read
            load_targets: whether to load the targets or only inputs

        Returns:
            tuple (input_dict, target_dict) where target_dict contains "days" and
                "valid". PerPixelTimestepHead maps the days to timestep indices.
        """
        if not load_targets:
            return {}, {}

        assert isinstance(raw_inputs["targets"], RasterImage)
        days = raw_inputs["targets"].get_hw_tensor().long()
        valid = (days != TIMESTAMP_NODATA_VALUE).float() * self._get_window_valid_mask(
            days, metadata
        )
        return {}, {
            "days": RasterImage(days[None, None, :, :], timestamps=None),
            "valid": RasterImage(valid[None, None, :, :], timestamps=None),
        }

    def process_output(
        self, raw_output: Any, metadata: SampleMetadata
    ) -> npt.NDArray[Any]:
        """Convert the predicted timestep at each pixel to days since 1970-01-01.

        Args:
            raw_output: the per-example output from PerPixelTimestepHead.
            metadata: metadata about the patch being read.

        Returns:
            1xHxW uint16 array of days since 1970-01-01.
        """
        if not isinstance(raw_output, dict) or "timestamps" not in raw_output:
            raise ValueError(
                "the output for PerPixelTimestepTask must come from PerPixelTimestepHead"
            )
        timestamps = raw_output["timestamps"]
        probs = raw_output["probs"]
        if len(timestamps) == 0:
            return np.full(
                (1, probs.shape[1], probs.shape[2]),
                TIMESTAMP_NODATA_VALUE,
                dtype=np.uint16,
            )
        timestep_idx = super().process_output(probs[: len(timestamps)], metadata)
        return timestamps.cpu().numpy().astype(np.uint16)[timestep_idx]

    def visualize(
        self,
        input_dict: dict[str, Any],
        target_dict: dict[str, Any] | None,
        output: Any,
    ) -> dict[str, npt.NDArray[Any]]:
        """Visualize the predicted and target timesteps.

        Args:
            input_dict: the input dict from process_inputs
            target_dict: the target dict from process_inputs
            output: the prediction

        Returns:
            a dictionary mapping image name to visualization image
        """
        timestep_targets = output.get("targets") if target_dict is not None else None
        return super().visualize(input_dict, timestep_targets, output["probs"])

    def get_metrics(self) -> MetricCollection:
        """Get the SegmentationTask metrics, computed on the timestep probabilities."""
        return MetricCollection(
            {
                name: TimestepProbsMetricWrapper(metric)
                for name, metric in super().get_metrics().items()
            }
        )


class TimestepProbsMetricWrapper(Metric):
    """Pass PerPixelTimestepHead outputs to a SegmentationTask metric.

    The metric gets the "probs" of each output, along with the timestep index targets
    that PerPixelTimestepHead computed from the labeled dates.
    """

    def __init__(self, metric: Metric) -> None:
        """Create a new TimestepProbsMetricWrapper.

        Args:
            metric: the metric to wrap.
        """
        super().__init__()
        self.metric = metric

    def update(
        self, preds: list[dict[str, Any]], targets: list[dict[str, Any]]
    ) -> None:
        """Update metric.

        Args:
            preds: the per-example outputs from PerPixelTimestepHead.
            targets: the targets from PerPixelTimestepTask. They are unused since the
                preds contain the corresponding timestep index targets.
        """
        if any("targets" not in pred for pred in preds):
            raise ValueError(
                "PerPixelTimestepHead outputs must include targets to compute metrics"
            )
        self.metric.update(
            [pred["probs"] for pred in preds], [pred["targets"] for pred in preds]
        )

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
