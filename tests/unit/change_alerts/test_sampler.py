"""Unit tests for rslearn.change_alerts.sampler."""

from datetime import UTC, datetime, timedelta

import pytest
import torch

from rslearn.change_alerts.sampler import (
    UNIX_EPOCH,
    ChangeTimeSeriesSampler,
    change_timestep,
    midpoint_day,
)
from rslearn.train.model_context import RasterImage
from rslearn.train.tasks.per_pixel_timestep import (
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


def _change_day_raster(day: int) -> RasterImage:
    days = torch.zeros((1, 1, H, W), dtype=torch.int32)
    days[0, 0, 1, 1] = day
    return RasterImage(days)


def _category_target() -> dict[str, RasterImage]:
    classes = torch.zeros((1, 1, H, W), dtype=torch.long)
    classes[0, 0, 1, 1] = 2
    return {
        "classes": RasterImage(classes),
        "valid": RasterImage(torch.ones((1, 1, H, W))),
    }


def _inputs(slot_ends_days: list[int]) -> dict:
    input_dict: dict = {"change_day": _change_day_raster(CHANGE_DAY)}
    for idx, end_days in enumerate(slot_ends_days):
        end = CHANGE + timedelta(days=end_days)
        input_dict[f"freq_{idx}"] = _series(end, timedelta(days=7), 8)
        # The last infrequent image overlaps the selected frequent images.
        input_dict[f"infreq_{idx}"] = _series(
            end - timedelta(days=14), timedelta(days=90), 10, value=100
        )
    return input_dict


OPTIONS: list[dict[str, str | None]] = [
    {"frequent": f"freq_{idx}", "infrequent": f"infreq_{idx}"} for idx in range(2)
]


def test_change_timestep() -> None:
    """The change maps to the image containing it, or the next one in a gap."""
    timestamps = [
        (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 8, tzinfo=UTC)),
        (datetime(2024, 1, 15, tzinfo=UTC), datetime(2024, 1, 15, tzinfo=UTC)),
    ]
    assert change_timestep(timestamps, datetime(2024, 1, 3, tzinfo=UTC)) == 0
    assert change_timestep(timestamps, datetime(2024, 1, 10, tzinfo=UTC)) == 1
    # An image ending exactly at the change time is before the change.
    assert change_timestep(timestamps, datetime(2024, 1, 8, tzinfo=UTC)) == 1
    assert change_timestep(timestamps, datetime(2023, 12, 1, tzinfo=UTC)) is None
    assert change_timestep(timestamps, datetime(2024, 2, 1, tzinfo=UTC)) is None


def test_history_series_and_targets() -> None:
    """Infrequent images precede the latest frequent images, with correct targets."""
    sampler = ChangeTimeSeriesSampler(
        options=OPTIONS,
        num_frequent=4,
        frequent_lookback_days=56,
        num_infrequent=8,
        output_key="sentinel2_l2a",
        option_index=0,
    )
    input_dict, target_dict = sampler(
        _inputs([7, 35]), {"category": _category_target()}
    )

    image = input_dict["sentinel2_l2a"]
    assert image.shape == (2, 12, H, W)
    # The last infrequent image is dropped since it ends after the first selected
    # frequent image starts; then the last 4 of the 8 frequent images.
    assert image.image[0, :, 0, 0].tolist() == [101 + i for i in range(8)] + [
        4,
        5,
        6,
        7,
    ]
    assert image.timestamps is not None
    assert image.timestamps[-1][1] == CHANGE + timedelta(days=7)
    for key in ["freq_0", "infreq_0", "freq_1", "infreq_1", "change_day"]:
        assert key not in input_dict

    timestep = target_dict["timestep"]
    # Slot 0 ends one week after the change, so it appears in the latest image. The
    # label is that image's midpoint day, which the head maps back to it.
    assert timestep["days"].image[0, 0, 1, 1] == midpoint_day(image.timestamps[11])
    assert timestep["valid"].image[0, 0].sum() == 1
    timestep_days = torch.tensor([midpoint_day(t) for t in image.timestamps])
    for mode in DateToTimestepMode:
        classes, valid = days_to_timesteps(
            timestep["days"].get_hw_tensor(),
            timestep["valid"].get_hw_tensor(),
            timestep_days,
            mode,
        )
        assert classes[1, 1] == 11
        assert valid.sum() == 1
    assert target_dict["category"]["valid"].image.sum() == H * W


def test_change_outside_series_is_invalid() -> None:
    """If the change is after the time series, the pixel is ignored."""
    sampler = ChangeTimeSeriesSampler(options=OPTIONS, num_frequent=4, option_index=0)
    input_dict = _inputs([-7, 35])
    _, target_dict = sampler(input_dict, {"category": _category_target()})
    assert target_dict["timestep"]["valid"].image.sum() == 0
    assert target_dict["category"]["valid"].image[0, 0, 1, 1] == 0
    assert target_dict["category"]["valid"].image.sum() == H * W - 1


def test_random_option_prefers_complete() -> None:
    """Options without enough images are skipped when another option is complete."""
    sampler = ChangeTimeSeriesSampler(options=OPTIONS, num_frequent=6)
    for _ in range(10):
        input_dict = _inputs([7, 35])
        input_dict["freq_0"] = _series(CHANGE + timedelta(days=7), timedelta(days=7), 3)
        input_dict, target_dict = sampler(input_dict, {"category": _category_target()})
        assert input_dict["image"].shape[1] == 6
        assert input_dict["image"].timestamps[-1][1] == CHANGE + timedelta(days=35)


def test_fallback_to_largest_option() -> None:
    """If no option is complete, the option with the most images is used."""
    sampler = ChangeTimeSeriesSampler(options=OPTIONS, num_frequent=12)
    input_dict, _ = sampler(_inputs([7, 35]), {"category": _category_target()})
    assert input_dict["image"].shape[1] == 8


def test_prediction_without_targets() -> None:
    """Without the change day raster, only the time series is built."""
    sampler = ChangeTimeSeriesSampler(
        options=[{"frequent": "freq_0"}], num_frequent=12, option_index=0
    )
    input_dict = _inputs([7])
    del input_dict["change_day"]
    input_dict, target_dict = sampler(input_dict, {})
    assert input_dict["image"].shape[1] == 8
    assert target_dict == {}


def test_drop_keys() -> None:
    """Inputs listed in drop_keys are removed even if not part of an option."""
    sampler = ChangeTimeSeriesSampler(
        options=[{"frequent": "freq_0"}],
        num_frequent=4,
        option_index=0,
        drop_keys=["freq_1", "infreq_1"],
    )
    input_dict, _ = sampler(_inputs([7, 35]), {"category": _category_target()})
    assert set(input_dict.keys()) == {"image", "infreq_0"}


def test_requires_infrequent_key() -> None:
    """Options must have an infrequent key when num_infrequent > 0."""
    with pytest.raises(ValueError):
        ChangeTimeSeriesSampler(
            options=[{"frequent": "freq_0"}], num_frequent=4, num_infrequent=8
        )
