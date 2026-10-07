"""Per-pixel timestep prediction, output as days since the Unix epoch.

The model classifies each pixel into one of the input timesteps (e.g. the image at
which a change occurred), like a SegmentationTask whose classes are the timesteps. At
prediction time, the predicted timestep is written as the uint16 number of days since
1970-01-01 (UTC) of the midpoint of that image's time range, so that the output is
meaningful without knowing which images the model was given.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torchmetrics import Metric, MetricCollection

from rslearn.train.model_context import (
    ModelContext,
    ModelOutput,
    RasterImage,
    SampleMetadata,
)

from .segmentation import SegmentationHead, SegmentationTask

UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

# Value written for pixels without a prediction.
TIMESTAMP_NODATA_VALUE = np.iinfo(np.uint16).max

# Largest day that can be written without colliding with nodata (2149-06-05).
MAX_DAYS = TIMESTAMP_NODATA_VALUE - 1


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


class PerPixelTimestepHead(SegmentationHead):
    """Head for PerPixelTimestepTask.

    It is meant to be applied over per-timestep logits at each pixel. It behaves like
    SegmentationHead, except after computing softmax outputs and cross entropy loss, it
    additionally attaches the date of each input timestep so that PerPixelTimestepTask
    can write dates instead of timestep indices.

    The logit channels must correspond to the timesteps of the input selected by the
    input_key in order. Each per-example output is a dict with "probs" (CHW softmax
    probabilities) and "timestamps" (int64 tensor with the number of days since
    1970-01-01 of the midpoint of each of the T input timesteps). Channels beyond T,
    e.g. from padding tokens, are ignored by PerPixelTimestepTask.
    """

    def __init__(self, input_key: str, **kwargs: Any) -> None:
        """Create a new PerPixelTimestepHead.

        Args:
            input_key: the key in the input dict of the RasterImage whose timestamps
                the logit channels correspond to.
            kwargs: other arguments to pass to SegmentationHead.
        """
        super().__init__(**kwargs)
        self.input_key = input_key

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
            targets: optional targets, same as for SegmentationHead.

        Returns:
            ModelOutput whose outputs are per-example dicts with "probs" and
            "timestamps".
        """
        model_output = super().forward(intermediates, context, targets)
        probs = model_output.outputs
        assert isinstance(probs, torch.Tensor)
        num_channels = probs.shape[1]

        outputs = []
        for example_probs, inputs in zip(probs, context.inputs):
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
            timestamps = torch.tensor(
                [_midpoint_days(time_range) for time_range in image.timestamps],
                dtype=torch.int64,
                device=probs.device,
            )
            outputs.append({"probs": example_probs, "timestamps": timestamps})

        return ModelOutput(outputs=outputs, loss_dict=model_output.loss_dict)


class PerPixelTimestepTask(SegmentationTask):
    """Predict an input timestep at each pixel and output it as days since 1970-01-01.

    Training is the same as SegmentationTask, with num_classes set to the number of
    input timesteps and targets containing timestep indices. It must be paired with
    PerPixelTimestepHead.

    At prediction time, the per-pixel argmax timestep is mapped to the number of days
    since 1970-01-01 (UTC) of the midpoint of that input image's time range, so the
    output layer's band set should use the uint16 dtype. 65535 (2^16 - 1) is used as
    the nodata value; set nodata_value: 65535 on the output band set so that the
    written raster is tagged with it.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Create a new PerPixelTimestepTask.

        Args:
            kwargs: arguments to pass to SegmentationTask. output_probs is not
                supported.
        """
        super().__init__(**kwargs)
        if self.output_probs:
            raise ValueError("PerPixelTimestepTask does not support output_probs")

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
        return super().visualize(input_dict, target_dict, output["probs"])

    def get_metrics(self) -> MetricCollection:
        """Get the SegmentationTask metrics, computed on the timestep probabilities."""
        return MetricCollection(
            {
                name: TimestepProbsMetricWrapper(metric)
                for name, metric in super().get_metrics().items()
            }
        )


class TimestepProbsMetricWrapper(Metric):
    """Pass the "probs" of PerPixelTimestepHead outputs to a SegmentationTask metric."""

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
            targets: the targets.
        """
        self.metric.update([pred["probs"] for pred in preds], targets)

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
