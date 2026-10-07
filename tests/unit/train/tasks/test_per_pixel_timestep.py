from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import torch

from rslearn.models.component import FeatureMaps
from rslearn.train.model_context import ModelContext, RasterImage, SampleMetadata
from rslearn.train.tasks.multi_task import MultiTask
from rslearn.train.tasks.per_pixel_timestep import (
    PerPixelTimestepHead,
    PerPixelTimestepTask,
)

INPUT_KEY = "image"

# Each timestep spans one day, so its midpoint is noon on that day.
DAYS = [datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 6, 15, tzinfo=UTC)]
TIME_RANGES = [(day, day + timedelta(days=1)) for day in DAYS]
MIDPOINT_DAYS = [(day - datetime(1970, 1, 1, tzinfo=UTC)).days for day in DAYS]


def _context(time_ranges: list[tuple[datetime, datetime]] | None) -> ModelContext:
    num_timesteps = len(time_ranges) if time_ranges is not None else 1
    image = RasterImage(torch.zeros((1, num_timesteps, 2, 2)), timestamps=time_ranges)
    return ModelContext(inputs=[{INPUT_KEY: image}], metadatas=[])


def _targets(classes: torch.Tensor) -> list[dict[str, RasterImage]]:
    return [
        {
            "classes": RasterImage(classes[None, None], timestamps=None),
            "valid": RasterImage(torch.ones((1, 1) + classes.shape), timestamps=None),
        }
    ]


def test_head_attaches_midpoint_timestamps() -> None:
    """The head should add each timestep's midpoint."""
    logits = torch.randn((1, 2, 2, 2))
    head = PerPixelTimestepHead(input_key=INPUT_KEY)
    output = head(FeatureMaps([logits]), _context(TIME_RANGES)).outputs[0]
    assert output["timestamps"].dtype == torch.int64
    assert output["timestamps"].tolist() == MIDPOINT_DAYS


def test_head_rejects_dates_outside_uint16_days() -> None:
    """Midpoints before 1970 or after 2149-06-06 cannot be written as uint16 days."""
    head = PerPixelTimestepHead(input_key=INPUT_KEY)
    feature_maps = FeatureMaps([torch.zeros((1, 1, 2, 2))])
    for day in [datetime(1969, 12, 31, tzinfo=UTC), datetime(2149, 6, 7, tzinfo=UTC)]:
        with pytest.raises(ValueError, match="uint16"):
            head(feature_maps, _context([(day, day + timedelta(hours=1))]))


def test_head_requires_timestamps() -> None:
    """The head should fail if the input is missing or has no timestamps."""
    feature_maps = FeatureMaps([torch.zeros((1, 2, 2, 2))])
    with pytest.raises(ValueError, match="RasterImage"):
        PerPixelTimestepHead(input_key="missing")(feature_maps, _context(TIME_RANGES))
    with pytest.raises(ValueError, match="no timestamps"):
        PerPixelTimestepHead(input_key=INPUT_KEY)(feature_maps, _context(None))


def test_head_requires_channel_per_timestep() -> None:
    """There must be at least as many logit channels as timesteps."""
    head = PerPixelTimestepHead(input_key=INPUT_KEY)
    with pytest.raises(ValueError, match="logit channels"):
        head(FeatureMaps([torch.zeros((1, 1, 2, 2))]), _context(TIME_RANGES))


def test_task_outputs_argmax_timestamp(
    empty_sample_metadata: SampleMetadata,
) -> None:
    """Each pixel should get the timestamp of its argmax timestep."""
    probs = torch.zeros((2, 2, 2))
    probs[0, 0, :] = 1
    probs[1, 1, :] = 1
    raw_output = {"probs": probs, "timestamps": torch.tensor(MIDPOINT_DAYS)}

    task = PerPixelTimestepTask(num_classes=2)
    result = task.process_output(raw_output, empty_sample_metadata)

    assert result.dtype == np.uint16
    expected = np.array([[MIDPOINT_DAYS[0]] * 2, [MIDPOINT_DAYS[1]] * 2])
    np.testing.assert_array_equal(result, expected[None])


def test_metrics_with_multi_task() -> None:
    """Metrics should be computed on the probabilities in the head output."""
    task = MultiTask(
        tasks={"ts": PerPixelTimestepTask(num_classes=2)},
        input_mapping={"ts": {}},
    )
    metrics = task.get_metrics()

    probs = torch.zeros((2, 2, 2))
    probs[0, 0, :] = 1
    probs[1, 1, :] = 1
    preds = [{"ts": {"probs": probs, "timestamps": torch.tensor(MIDPOINT_DAYS)}}]
    # Accuracy is macro-averaged: 2/3 pixels of class 0 and 1/1 of class 1 are correct.
    targets = [{"ts": _targets(torch.tensor([[0, 0], [1, 0]]))[0]}]
    metrics.update(preds, targets)
    assert metrics.compute()["ts/accuracy"] == pytest.approx((2 / 3 + 1) / 2)
