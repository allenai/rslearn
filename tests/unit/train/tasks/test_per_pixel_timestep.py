from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import torch

from rslearn.models.component import FeatureMaps
from rslearn.train.model_context import ModelContext, RasterImage, SampleMetadata
from rslearn.train.tasks.multi_task import MultiTask
from rslearn.train.tasks.per_pixel_timestep import (
    DateToTimestepMode,
    PerPixelTimestepHead,
    PerPixelTimestepTask,
    days_to_timesteps,
)
from rslearn.utils.geometry import WGS84_PROJECTION

INPUT_KEY = "image"
NODATA = 65535

# Each timestep spans one day, so its midpoint is noon on that day.
DAYS = [datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 6, 15, tzinfo=UTC)]
TIME_RANGES = [(day, day + timedelta(days=1)) for day in DAYS]
MIDPOINT_DAYS = [(day - datetime(1970, 1, 1, tzinfo=UTC)).days for day in DAYS]


def _context(time_ranges: list[tuple[datetime, datetime]] | None) -> ModelContext:
    num_timesteps = len(time_ranges) if time_ranges is not None else 1
    image = RasterImage(torch.zeros((1, num_timesteps, 2, 2)), timestamps=time_ranges)
    return ModelContext(inputs=[{INPUT_KEY: image}], metadatas=[])


def _day_targets(days: torch.Tensor) -> dict[str, RasterImage]:
    return {
        "days": RasterImage(days[None, None], timestamps=None),
        "valid": RasterImage((days != NODATA).float()[None, None], timestamps=None),
    }


def _class_targets(classes: torch.Tensor) -> dict[str, RasterImage]:
    return {
        "classes": RasterImage(classes[None, None], timestamps=None),
        "valid": RasterImage(torch.ones((1, 1) + classes.shape), timestamps=None),
    }


def _convert(
    label_days: list[int],
    timestep_days: list[int],
    mode: DateToTimestepMode,
    fallback: bool = False,
) -> tuple[list[int], list[float]]:
    days = torch.tensor([label_days])
    classes, valid = days_to_timesteps(
        days,
        (days != NODATA).float(),
        torch.tensor(timestep_days),
        mode,
        fallback_to_closest_timestep_if_no_match=fallback,
    )
    return classes[0].tolist(), valid[0].tolist()


class TestDaysToTimesteps:
    """Tests for days_to_timesteps."""

    TIMESTEP_DAYS = [10, 20, 30]

    def test_before(self) -> None:
        """BEFORE picks the latest timestep on or before the labeled day."""
        classes, valid = _convert(
            [15, 25, 35], self.TIMESTEP_DAYS, DateToTimestepMode.BEFORE
        )
        assert classes == [0, 1, 2]
        assert valid == [1, 1, 1]

    def test_after(self) -> None:
        """AFTER picks the earliest timestep on or after the labeled day."""
        classes, valid = _convert(
            [5, 15, 25], self.TIMESTEP_DAYS, DateToTimestepMode.AFTER
        )
        assert classes == [0, 1, 2]
        assert valid == [1, 1, 1]

    def test_same_day_matches_both_modes(self) -> None:
        """A timestep on the labeled day matches in both modes."""
        for mode in DateToTimestepMode:
            classes, valid = _convert([20], self.TIMESTEP_DAYS, mode)
            assert classes == [1]
            assert valid == [1]

    def test_compares_days_not_index_order(self) -> None:
        """Timesteps need not be in chronological order."""
        timestep_days = [30, 10, 20]
        assert _convert([25], timestep_days, DateToTimestepMode.BEFORE)[0] == [2]
        assert _convert([15], timestep_days, DateToTimestepMode.AFTER)[0] == [2]

    def test_ties_pick_first_timestep(self) -> None:
        """Multiple timesteps on the same day deterministically pick the first."""
        timestep_days = [10, 20, 20, 30]
        for mode in DateToTimestepMode:
            assert _convert([20], timestep_days, mode)[0] == [1]
        assert _convert([25], timestep_days, DateToTimestepMode.BEFORE)[0] == [1]
        assert _convert([15], timestep_days, DateToTimestepMode.AFTER)[0] == [1]

    def test_no_match_is_invalid(self) -> None:
        """Without a qualifying timestep, the pixel is invalid by default."""
        classes, valid = _convert([5], self.TIMESTEP_DAYS, DateToTimestepMode.BEFORE)
        assert classes == [0]
        assert valid == [0]
        classes, valid = _convert([35], self.TIMESTEP_DAYS, DateToTimestepMode.AFTER)
        assert classes == [0]
        assert valid == [0]

    def test_no_match_falls_back_to_closest(self) -> None:
        """With the fallback, a pixel without a qualifying timestep uses the closest."""
        classes, valid = _convert(
            [5], self.TIMESTEP_DAYS, DateToTimestepMode.BEFORE, fallback=True
        )
        assert classes == [0]
        assert valid == [1]
        classes, valid = _convert(
            [35], self.TIMESTEP_DAYS, DateToTimestepMode.AFTER, fallback=True
        )
        assert classes == [2]
        assert valid == [1]

    def test_nodata_stays_invalid(self) -> None:
        """Unlabeled pixels stay invalid, even with the fallback."""
        for mode in DateToTimestepMode:
            classes, valid = _convert([NODATA], self.TIMESTEP_DAYS, mode, fallback=True)
            assert classes == [0]
            assert valid == [0]


def test_head_attaches_midpoint_timestamps() -> None:
    """The head should add each timestep's midpoint."""
    logits = torch.randn((1, 2, 2, 2))
    head = PerPixelTimestepHead(input_key=INPUT_KEY, mode=DateToTimestepMode.BEFORE)
    output = head(FeatureMaps([logits]), _context(TIME_RANGES)).outputs[0]
    assert output["timestamps"].dtype == torch.int64
    assert output["timestamps"].tolist() == MIDPOINT_DAYS
    assert "targets" not in output


def test_head_converts_day_targets() -> None:
    """The loss and output targets should use the timestep matching each labeled day."""
    # Label days: the first timestep's day, a day between the timesteps, the second
    # timestep's day, and nodata.
    label_days = torch.tensor(
        [[MIDPOINT_DAYS[0], MIDPOINT_DAYS[0] + 10], [MIDPOINT_DAYS[1], NODATA]]
    )
    # Logits strongly favor timestep 0 at the top row and timestep 1 at the bottom.
    logits = torch.zeros((1, 2, 2, 2))
    logits[0, 0, 0, :] = 10
    logits[0, 1, 1, :] = 10

    expected = {
        DateToTimestepMode.BEFORE: [[0, 0], [1, 0]],
        DateToTimestepMode.AFTER: [[0, 1], [1, 0]],
    }
    for mode, expected_classes in expected.items():
        head = PerPixelTimestepHead(input_key=INPUT_KEY, mode=mode)
        model_output = head(
            FeatureMaps([logits]), _context(TIME_RANGES), [_day_targets(label_days)]
        )
        targets = model_output.outputs[0]["targets"]
        assert targets["classes"].get_hw_tensor().tolist() == expected_classes
        assert targets["valid"].get_hw_tensor().tolist() == [[1, 1], [1, 0]]

        # With BEFORE every valid pixel agrees with the logits, so the loss is small,
        # while with AFTER the top right pixel is wrong.
        loss = model_output.loss_dict["cls"].item()
        if mode == DateToTimestepMode.BEFORE:
            assert loss < 0.01
        else:
            assert loss > 1


def test_head_rejects_dates_outside_uint16_days() -> None:
    """Midpoints before 1970 or after 2149-06-05 cannot be written as uint16 days."""
    head = PerPixelTimestepHead(input_key=INPUT_KEY, mode=DateToTimestepMode.BEFORE)
    feature_maps = FeatureMaps([torch.zeros((1, 1, 2, 2))])
    # 2149-06-06 is day 65535, which is reserved for nodata.
    for day in [datetime(1969, 12, 31, tzinfo=UTC), datetime(2149, 6, 6, tzinfo=UTC)]:
        with pytest.raises(ValueError, match="uint16"):
            head(feature_maps, _context([(day, day + timedelta(hours=1))]))


def test_head_requires_timestamps() -> None:
    """The head should fail if the input is missing or has no timestamps."""
    feature_maps = FeatureMaps([torch.zeros((1, 2, 2, 2))])
    with pytest.raises(ValueError, match="RasterImage"):
        PerPixelTimestepHead(input_key="missing", mode=DateToTimestepMode.BEFORE)(
            feature_maps, _context(TIME_RANGES)
        )
    with pytest.raises(ValueError, match="no timestamps"):
        PerPixelTimestepHead(input_key=INPUT_KEY, mode=DateToTimestepMode.BEFORE)(
            feature_maps, _context(None)
        )


def test_head_requires_channel_per_timestep() -> None:
    """There must be at least as many logit channels as timesteps."""
    head = PerPixelTimestepHead(input_key=INPUT_KEY, mode=DateToTimestepMode.BEFORE)
    with pytest.raises(ValueError, match="logit channels"):
        head(FeatureMaps([torch.zeros((1, 1, 2, 2))]), _context(TIME_RANGES))


def test_task_reads_day_targets() -> None:
    """process_inputs should keep the labeled days and treat 65535 as nodata."""
    task = PerPixelTimestepTask(num_classes=2)
    labels = torch.tensor([[[[MIDPOINT_DAYS[0], NODATA], [0, MIDPOINT_DAYS[1]]]]])
    metadata = SampleMetadata(
        window_group="",
        window_name="",
        window_bounds=(0, 0, 2, 2),
        crop_bounds=(0, 0, 2, 2),
        crop_idx=0,
        num_crops_in_window=1,
        time_range=None,
        projection=WGS84_PROJECTION,
        dataset_source=None,
    )
    _, target_dict = task.process_inputs(
        {"targets": RasterImage(labels, timestamps=None)}, metadata
    )
    assert target_dict["days"].get_hw_tensor().tolist() == labels[0, 0].tolist()
    assert target_dict["valid"].get_hw_tensor().tolist() == [[1, 0], [1, 1]]


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


def test_task_outputs_nodata_without_timesteps(
    empty_sample_metadata: SampleMetadata,
) -> None:
    """Without any input timesteps, every pixel should be nodata (65535)."""
    raw_output = {
        "probs": torch.zeros((2, 3, 4)),
        "timestamps": torch.tensor([], dtype=torch.int64),
    }
    task = PerPixelTimestepTask(num_classes=2)
    result = task.process_output(raw_output, empty_sample_metadata)

    assert result.dtype == np.uint16
    np.testing.assert_array_equal(result, np.full((1, 3, 4), 65535))


def test_metrics_with_multi_task() -> None:
    """Metrics should use the probabilities and timestep targets in the head output."""
    task = MultiTask(
        tasks={"ts": PerPixelTimestepTask(num_classes=2)},
        input_mapping={"ts": {}},
    )
    metrics = task.get_metrics()

    probs = torch.zeros((2, 2, 2))
    probs[0, 0, :] = 1
    probs[1, 1, :] = 1
    pred = {
        "probs": probs,
        "timestamps": torch.tensor(MIDPOINT_DAYS),
        "targets": _class_targets(torch.tensor([[0, 0], [1, 0]])),
    }
    # The dataset targets hold days, which the metrics should not use.
    day_targets = _day_targets(torch.full((2, 2), MIDPOINT_DAYS[0]))
    metrics.update([{"ts": pred}], [{"ts": day_targets}])
    # Accuracy is macro-averaged: 2/3 pixels of class 0 and 1/1 of class 1 are correct.
    assert metrics.compute()["ts/accuracy"] == pytest.approx((2 / 3 + 1) / 2)
