"""Unit tests for rslearn.change_alerts.sampler."""

import random
from datetime import UTC, datetime, timedelta
from typing import Any

import torch

from rslearn.change_alerts.sampler import (
    UNIX_EPOCH,
    ChangeTimeSeriesSampler,
    midpoint_day,
)
from rslearn.train.model_context import RasterImage
from rslearn.train.tasks.per_pixel_timestep import (
    TIMESTAMP_NODATA_VALUE,
    DateToTimestepMode,
    days_to_timesteps,
)

H = W = 4
CHANGE = datetime(2024, 5, 10, tzinfo=UTC)
CHANGE_DAY = (CHANGE - UNIX_EPOCH).days


def _series(
    end: datetime, period: timedelta, count: int, value: float = 0
) -> RasterImage:
    """Make a chronological series of count images ending at end."""
    timestamps = [
        (end - (count - i) * period, end - (count - i - 1) * period)
        for i in range(count)
    ]
    # Encode the timestep index in the pixel values so we can check the selection.
    image = torch.arange(count, dtype=torch.float32)[None, :, None, None] + value
    return RasterImage(image.expand(2, count, H, W).clone(), timestamps=timestamps)


def _targets(day: int = CHANGE_DAY) -> dict[str, Any]:
    """Make category and timestep targets with a change at pixel (1, 1) on day."""
    classes = torch.zeros((1, 1, H, W), dtype=torch.long)
    classes[0, 0, 1, 1] = 2
    days = torch.full((1, 1, H, W), TIMESTAMP_NODATA_VALUE, dtype=torch.long)
    days[0, 0, 1, 1] = day
    return {
        "category": {
            "classes": RasterImage(classes),
            "valid": RasterImage(torch.ones((1, 1, H, W))),
        },
        "timestep": {
            "days": RasterImage(days),
            "valid": RasterImage((days != TIMESTAMP_NODATA_VALUE).float()),
        },
    }


def _inputs(slot_ends_days: list[int]) -> dict:
    input_dict: dict = {}
    for idx, end_days in enumerate(slot_ends_days):
        end = CHANGE + timedelta(days=end_days)
        input_dict[f"freq_{idx}"] = _series(end, timedelta(days=7), 4)
    # One infrequent series shared by all slots. With slot ends [7, 35], the last
    # infrequent image ends at day 5, after the first frequent image of the first
    # slot starts (day -21) but before that of the second slot (day 7).
    input_dict["infreq"] = _series(
        CHANGE + timedelta(days=max(slot_ends_days) - 30),
        timedelta(days=90),
        10,
        value=100,
    )
    return input_dict


FREQUENT_KEYS = ["freq_0", "freq_1"]
MASKING_ARGS: dict[str, Any] = {
    "category_target": "category",
    "change_day_target": "timestep",
}


def test_history_series_and_targets() -> None:
    """Infrequent images precede the frequent images, and targets are unchanged."""
    sampler = ChangeTimeSeriesSampler(
        frequent_keys=FREQUENT_KEYS,
        infrequent_key="infreq",
        num_infrequent=8,
        output_key="sentinel2_l2a",
        frequent_index=0,
    )
    input_dict, target_dict = sampler(_inputs([7, 35]), _targets())

    image = input_dict["sentinel2_l2a"]
    assert image.shape == (2, 12, H, W)
    # The last infrequent image is dropped since it ends after the first frequent
    # image starts; then the 4 frequent images.
    assert image.image[0, :, 0, 0].tolist() == [101 + i for i in range(8)] + [
        0,
        1,
        2,
        3,
    ]
    assert image.timestamps is not None
    assert image.timestamps[-1][1] == CHANGE + timedelta(days=7)

    assert target_dict["category"]["valid"].image.sum() == H * W
    timestep = target_dict["timestep"]
    assert timestep["valid"].image.sum() == 1

    # Slot 0 ends one week after the change, so with mode AFTER the head maps the
    # change to the latest image.
    timestep_days = torch.tensor([midpoint_day(t) for t in image.timestamps])
    classes, valid = days_to_timesteps(
        timestep["days"].get_hw_tensor(),
        timestep["valid"].get_hw_tensor(),
        timestep_days,
        DateToTimestepMode.AFTER,
    )
    assert classes[1, 1] == 11
    assert valid.sum() == 1


def test_shared_infrequent_depends_on_slot() -> None:
    """A later slot uses more recent images from the shared infrequent input."""
    sampler = ChangeTimeSeriesSampler(
        frequent_keys=FREQUENT_KEYS,
        infrequent_key="infreq",
        num_infrequent=8,
        frequent_index=1,
    )
    input_dict, _ = sampler(_inputs([7, 35]), _targets())
    image = input_dict["image"]
    assert image.image[0, :, 0, 0].tolist() == [102 + i for i in range(8)] + [
        0,
        1,
        2,
        3,
    ]
    assert image.timestamps is not None
    assert image.timestamps[-1][1] == CHANGE + timedelta(days=35)


def test_change_inside_series_is_not_masked() -> None:
    """With masking enabled, a change within the time series stays valid."""
    sampler = ChangeTimeSeriesSampler(
        frequent_keys=FREQUENT_KEYS, frequent_index=0, **MASKING_ARGS
    )
    _, target_dict = sampler(_inputs([7, 35]), _targets())
    assert target_dict["category"]["valid"].image.sum() == H * W
    assert target_dict["timestep"]["valid"].image.sum() == 1


def test_change_outside_series_is_masked() -> None:
    """If the change is after the time series, the pixel is ignored by both tasks."""
    sampler = ChangeTimeSeriesSampler(
        frequent_keys=FREQUENT_KEYS, frequent_index=0, **MASKING_ARGS
    )
    _, target_dict = sampler(_inputs([-7, 35]), _targets())
    assert target_dict["timestep"]["valid"].image[0, 0, 1, 1] == 0
    assert target_dict["category"]["valid"].image[0, 0, 1, 1] == 0
    assert target_dict["category"]["valid"].image.sum() == H * W - 1


def test_change_before_series_is_masked() -> None:
    """If the change is before the first image, the pixel is ignored by both tasks."""
    sampler = ChangeTimeSeriesSampler(
        frequent_keys=["freq_0"], frequent_index=0, **MASKING_ARGS
    )
    # The frequent images start 21 days before the change.
    _, target_dict = sampler(_inputs([7]), _targets(CHANGE_DAY - 30))
    assert target_dict["timestep"]["valid"].image[0, 0, 1, 1] == 0
    assert target_dict["category"]["valid"].image[0, 0, 1, 1] == 0
    assert target_dict["category"]["valid"].image.sum() == H * W - 1


def test_no_masking_by_default() -> None:
    """Without the masking arguments, changes outside the time series stay valid."""
    sampler = ChangeTimeSeriesSampler(frequent_keys=FREQUENT_KEYS, frequent_index=0)
    _, target_dict = sampler(_inputs([-7, 35]), _targets())
    assert target_dict["category"]["valid"].image.sum() == H * W
    assert target_dict["timestep"]["valid"].image.sum() == 1


def test_prefers_most_frequent_images() -> None:
    """The frequent input with the most images is picked."""
    sampler = ChangeTimeSeriesSampler(frequent_keys=FREQUENT_KEYS)
    for _ in range(10):
        input_dict = _inputs([7, 35])
        input_dict["freq_0"] = _series(CHANGE + timedelta(days=7), timedelta(days=7), 3)
        input_dict, _ = sampler(input_dict, _targets())
        assert input_dict["image"].shape[1] == 4
        assert input_dict["image"].timestamps[-1][1] == CHANGE + timedelta(days=35)


def test_random_among_ties() -> None:
    """Frequent inputs tied for the most images are picked randomly."""
    random.seed(0)
    sampler = ChangeTimeSeriesSampler(frequent_keys=FREQUENT_KEYS)
    ends = set()
    for _ in range(50):
        input_dict, _ = sampler(_inputs([7, 35]), _targets())
        ends.add(input_dict["image"].timestamps[-1][1])
    assert ends == {CHANGE + timedelta(days=7), CHANGE + timedelta(days=35)}


def test_prediction_without_targets() -> None:
    """Without targets, only the time series is built, even with masking enabled."""
    sampler = ChangeTimeSeriesSampler(
        frequent_keys=["freq_0"], frequent_index=0, **MASKING_ARGS
    )
    input_dict, target_dict = sampler(_inputs([7]), {})
    assert input_dict["image"].shape[1] == 4
    assert target_dict == {}

    # MultiTask with skip_targets produces empty per-task target dicts.
    input_dict, target_dict = sampler(_inputs([7]), {"category": {}, "timestep": {}})
    assert input_dict["image"].shape[1] == 4
    assert target_dict == {"category": {}, "timestep": {}}
