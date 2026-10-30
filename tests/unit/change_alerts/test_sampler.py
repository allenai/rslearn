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
WEEK = timedelta(days=7)
QUARTER = timedelta(days=90)
# The training window starts 3 weeks before the change.
WINDOW_START = CHANGE - 3 * WEEK
INFREQUENT_VALUE = 100


def _series(
    start: datetime, period: timedelta, count: int, value: float = 0
) -> RasterImage:
    """Make a chronological series of count images starting at start.

    Each image covers one period, and every pixel in every band of image i is set to
    value + i, so the selected images can be identified from the output pixel values.
    For example, with value=100, the pixels of the third image are all 102.
    """
    timestamps = [(start + i * period, start + (i + 1) * period) for i in range(count)]
    image = torch.arange(count, dtype=torch.float32)[None, :, None, None] + value
    return RasterImage(image.expand(2, count, H, W).clone(), timestamps=timestamps)


def _inputs(num_frequent_images: int, start: datetime = WINDOW_START) -> dict:
    """Make frequent images from start, and 8 infrequent images ending at start."""
    return {
        "freq": _series(start, WEEK, num_frequent_images),
        "infreq": _series(start - 8 * QUARTER, QUARTER, 8, value=INFREQUENT_VALUE),
    }


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


def _values(input_dict: dict, key: str = "image") -> list[int]:
    """Get the value of each timestep in the sampled time series.

    This reverses the encoding in _series: each value is the index of the frequent
    image, or INFREQUENT_VALUE plus the index of the infrequent image, that was
    selected for that timestep.
    """
    return [int(v) for v in input_dict[key].image[0, :, 0, 0].tolist()]


def _frequent_values(input_dict: dict) -> list[int]:
    """Get the values of the frequent images (including the stand-in) in the series."""
    return [v for v in _values(input_dict) if v < INFREQUENT_VALUE]


SAMPLER_ARGS: dict[str, Any] = {
    "frequent_key": "freq",
    "num_frequent": 4,
    "infrequent_key": "infreq",
    "num_infrequent": 8,
}
MASKING_ARGS: dict[str, Any] = {
    "category_target": "category",
    "change_day_target": "timestep",
}
INFREQUENT_VALUES = [INFREQUENT_VALUE + i for i in range(8)]


def test_latest_prediction_window() -> None:
    """With N frequent images, latest uses them all after the infrequent images."""
    sampler = ChangeTimeSeriesSampler(
        **SAMPLER_ARGS, output_key="sentinel2_l2a", select="latest"
    )
    input_dict, target_dict = sampler(_inputs(4), _targets())

    image = input_dict["sentinel2_l2a"]
    assert image.shape == (2, 12, H, W)
    # The last infrequent image ends exactly at the first frequent image start.
    assert _values(input_dict, "sentinel2_l2a") == INFREQUENT_VALUES + [0, 1, 2, 3]
    assert image.timestamps is not None
    assert image.timestamps[-1][1] == CHANGE + WEEK

    assert target_dict["category"]["valid"].image.sum() == H * W
    timestep = target_dict["timestep"]
    assert timestep["valid"].image.sum() == 1

    # This is the earliest pick in a training window: the change appears in the latest
    # image.
    timestep_days = torch.tensor([midpoint_day(t) for t in image.timestamps])
    classes, valid = days_to_timesteps(
        timestep["days"].get_hw_tensor(),
        timestep["valid"].get_hw_tensor(),
        timestep_days,
        DateToTimestepMode.AFTER,
    )
    assert classes[1, 1] == 11
    assert valid.sum() == 1


def test_fewer_than_num_frequent() -> None:
    """If there are fewer than N frequent images, all of them are used."""
    sampler = ChangeTimeSeriesSampler(**SAMPLER_ARGS, select="latest")
    input_dict, _ = sampler(_inputs(2), _targets())
    assert _values(input_dict) == INFREQUENT_VALUES + [0, 1]


def test_stand_in_when_gap_is_large() -> None:
    """The frequent image before the pick replaces the oldest infrequent image."""
    sampler = ChangeTimeSeriesSampler(**SAMPLER_ARGS, select="latest")
    # The pick starts 7 weeks (49 days) after the infrequent images end.
    input_dict, _ = sampler(_inputs(11), _targets())
    assert _values(input_dict) == INFREQUENT_VALUES[1:] + [6, 7, 8, 9, 10]


def test_no_stand_in_when_gap_is_small() -> None:
    """No stand-in is used if the gap is less than half the infrequent period."""
    sampler = ChangeTimeSeriesSampler(**SAMPLER_ARGS, select="latest")
    # The pick starts 5 weeks (35 days) after the infrequent images end.
    input_dict, _ = sampler(_inputs(9), _targets())
    assert _values(input_dict) == INFREQUENT_VALUES + [5, 6, 7, 8]


def test_random_start_covers_window() -> None:
    """Random picks range from the earliest to the latest N consecutive images."""
    random.seed(0)
    sampler = ChangeTimeSeriesSampler(**SAMPLER_ARGS, full_frequent_probability=1)
    ends = set()
    for _ in range(300):
        input_dict, _ = sampler(_inputs(16), _targets())
        assert input_dict["image"].shape[1] == 12
        frequent = _frequent_values(input_dict)[-4:]
        assert frequent == list(range(frequent[0], frequent[0] + 4))
        ends.add(input_dict["image"].timestamps[-1][1])
    # The change appears anywhere from the latest image to 91 days before the end.
    assert ends == {CHANGE + WEEK * k for k in range(1, 14)}


def test_reduction_keeps_latest() -> None:
    """Reduced picks keep the latest n images, and the history is unchanged."""
    random.seed(0)
    sampler = ChangeTimeSeriesSampler(**SAMPLER_ARGS, full_frequent_probability=0)
    counts = set()
    for _ in range(100):
        input_dict, _ = sampler(_inputs(4), _targets())
        values = _values(input_dict)
        num_kept = len(values) - 8
        counts.add(num_kept)
        assert values == INFREQUENT_VALUES + list(range(4 - num_kept, 4))
    assert counts == {1, 2, 3}


def test_reduction_never_uses_dropped_image_as_stand_in() -> None:
    """The stand-in is the image just before the original pick."""
    random.seed(0)
    sampler = ChangeTimeSeriesSampler(**SAMPLER_ARGS, full_frequent_probability=0)
    num_stand_ins = 0
    for _ in range(200):
        input_dict, _ = sampler(_inputs(16), _targets())
        frequent = _frequent_values(input_dict)
        kept = frequent[-1] - 3, frequent[-1]
        if len(_values(input_dict)) - len(frequent) == 7:
            # The first frequent value is the stand-in, one before the pick start.
            assert frequent[0] == kept[0] - 1
            assert all(v > kept[0] for v in frequent[1:-1])
            num_stand_ins += 1
        else:
            assert all(kept[0] <= v for v in frequent)
    assert num_stand_ins > 0


def test_full_frequent_probability_one() -> None:
    """With full_frequent_probability=1, the picks are never reduced."""
    random.seed(0)
    sampler = ChangeTimeSeriesSampler(**SAMPLER_ARGS, full_frequent_probability=1)
    for _ in range(50):
        input_dict, _ = sampler(_inputs(4), _targets())
        assert _values(input_dict) == INFREQUENT_VALUES + [0, 1, 2, 3]


def test_change_inside_series_is_not_masked() -> None:
    """With masking enabled, a change within the time series stays valid."""
    sampler = ChangeTimeSeriesSampler(**SAMPLER_ARGS, select="latest", **MASKING_ARGS)
    _, target_dict = sampler(_inputs(4), _targets())
    assert target_dict["category"]["valid"].image.sum() == H * W
    assert target_dict["timestep"]["valid"].image.sum() == 1


def test_change_outside_series_is_masked() -> None:
    """If the change is after the time series, the pixel is ignored by both tasks."""
    sampler = ChangeTimeSeriesSampler(**SAMPLER_ARGS, select="latest", **MASKING_ARGS)
    _, target_dict = sampler(_inputs(4), _targets(CHANGE_DAY + 14))
    assert target_dict["timestep"]["valid"].image[0, 0, 1, 1] == 0
    assert target_dict["category"]["valid"].image[0, 0, 1, 1] == 0
    assert target_dict["category"]["valid"].image.sum() == H * W - 1


def test_change_before_series_is_masked() -> None:
    """If the change is before the first image, the pixel is ignored by both tasks."""
    sampler = ChangeTimeSeriesSampler(
        frequent_key="freq", num_frequent=4, select="latest", **MASKING_ARGS
    )
    # The frequent images start 21 days before the change.
    _, target_dict = sampler(_inputs(4), _targets(CHANGE_DAY - 30))
    assert target_dict["timestep"]["valid"].image[0, 0, 1, 1] == 0
    assert target_dict["category"]["valid"].image[0, 0, 1, 1] == 0
    assert target_dict["category"]["valid"].image.sum() == H * W - 1


def test_no_masking_by_default() -> None:
    """Without the masking arguments, changes outside the time series stay valid."""
    sampler = ChangeTimeSeriesSampler(**SAMPLER_ARGS, select="latest")
    _, target_dict = sampler(_inputs(4), _targets(CHANGE_DAY + 14))
    assert target_dict["category"]["valid"].image.sum() == H * W
    assert target_dict["timestep"]["valid"].image.sum() == 1


def test_prediction_without_targets() -> None:
    """Without targets, only the time series is built, even with masking enabled."""
    sampler = ChangeTimeSeriesSampler(**SAMPLER_ARGS, select="latest", **MASKING_ARGS)
    input_dict, target_dict = sampler(_inputs(4), {})
    assert input_dict["image"].shape[1] == 12
    assert target_dict == {}

    # MultiTask with skip_targets produces empty per-task target dicts.
    input_dict, target_dict = sampler(_inputs(4), {"category": {}, "timestep": {}})
    assert input_dict["image"].shape[1] == 12
    assert target_dict == {"category": {}, "timestep": {}}
