"""Test rslearn.models.olmoearth_pretrain.model_period_timestamps."""

from datetime import datetime, timedelta

import pytest
import torch
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskValue

from rslearn.const import WGS84_PROJECTION
from rslearn.models.olmoearth_pretrain.model_period_timestamps import (
    OlmoEarthPeriodTimestamps,
)
from rslearn.train.model_context import ModelContext, RasterImage, SampleMetadata


def _make_metadata(
    crop_bounds: tuple[int, int, int, int],
    time_range: tuple[datetime, datetime] | None = None,
) -> SampleMetadata:
    return SampleMetadata(
        window_group="test",
        window_name="test",
        window_bounds=crop_bounds,
        crop_bounds=crop_bounds,
        crop_idx=0,
        num_crops_in_window=1,
        time_range=time_range,
        projection=WGS84_PROJECTION,
        dataset_source=None,
    )


def _make_model(
    period_duration: timedelta = timedelta(days=30),
    max_matches: int = 4,
) -> OlmoEarthPeriodTimestamps:
    return OlmoEarthPeriodTimestamps(
        checkpoint_path="tests/unit/models/olmoearth_pretrain/",
        random_initialization=True,
        patch_size=4,
        embedding_size=128,
        period_duration=period_duration,
        max_matches=max_matches,
    )


def test_compute_periods_backwards_from_end() -> None:
    """Periods are built from the end of the time range backwards."""
    time_range = (datetime(2025, 1, 1), datetime(2025, 5, 1))
    periods = OlmoEarthPeriodTimestamps._compute_periods(
        time_range, timedelta(days=30), max_matches=4
    )
    # 120 days / 30 = 4 periods fitting. Constructed end-backwards, reversed.
    assert len(periods) == 4
    # Check chronological order and that they tile from the end.
    assert periods[3][1] == datetime(2025, 5, 1)
    assert periods[0][0] == periods[3][1] - 4 * timedelta(days=30)
    for i in range(3):
        assert periods[i][1] == periods[i + 1][0]


def test_compute_periods_truncated_by_max_matches() -> None:
    """max_matches limits the number of periods, keeping most recent ones."""
    time_range = (datetime(2025, 1, 1), datetime(2025, 5, 1))
    periods = OlmoEarthPeriodTimestamps._compute_periods(
        time_range, timedelta(days=30), max_matches=2
    )
    assert len(periods) == 2
    # Should be the 2 most recent periods.
    assert periods[1][1] == datetime(2025, 5, 1)
    assert periods[0][1] == periods[1][0]


def test_single_modality_period_alignment() -> None:
    """One modality with 4 images in a 4-period window should fill all 4 slots."""
    model = _make_model(period_duration=timedelta(days=30), max_matches=4)

    H, W = 4, 4
    time_range = (datetime(2025, 1, 1), datetime(2025, 5, 1))
    timestamps = [
        (datetime(2025, 1, 10), datetime(2025, 1, 10)),
        (datetime(2025, 2, 10), datetime(2025, 2, 10)),
        (datetime(2025, 3, 10), datetime(2025, 3, 10)),
        (datetime(2025, 4, 10), datetime(2025, 4, 10)),
    ]
    inputs = [
        {
            "sentinel2_l2a": RasterImage(
                image=torch.ones((12, 4, H, W), dtype=torch.float32),
                timestamps=timestamps,
            ),
        }
    ]
    context = ModelContext(
        inputs=inputs,
        metadatas=[_make_metadata((0, 0, H, W), time_range=time_range)],
    )
    sample, present_modalities, _ = model._prepare_modality_inputs(context)

    assert present_modalities == ["sentinel2_l2a"]
    assert sample.timestamps.shape == (1, 4, 3)

    mask = sample.sentinel2_l2a_mask  # (B, H, W, T, S)
    assert mask.shape[3] == 4
    for t_idx in range(4):
        assert (mask[0, :, :, t_idx, :] == MaskValue.ONLINE_ENCODER.value).all()


def test_missing_timesteps_in_periods() -> None:
    """One modality with 2 images in a 4-period window: 2 slots filled, 2 MISSING."""
    model = _make_model(period_duration=timedelta(days=30), max_matches=4)

    H, W = 4, 4
    time_range = (datetime(2025, 1, 1), datetime(2025, 5, 1))
    # Only 2 images: one in period 0 (Jan) and one in period 2 (Mar)
    timestamps = [
        (datetime(2025, 1, 15), datetime(2025, 1, 15)),
        (datetime(2025, 3, 15), datetime(2025, 3, 15)),
    ]
    inputs = [
        {
            "sentinel2_l2a": RasterImage(
                image=torch.ones((12, 2, H, W), dtype=torch.float32),
                timestamps=timestamps,
            ),
        }
    ]
    context = ModelContext(
        inputs=inputs,
        metadatas=[_make_metadata((0, 0, H, W), time_range=time_range)],
    )
    sample, _, _ = model._prepare_modality_inputs(context)

    mask = sample.sentinel2_l2a_mask  # (B, H, W, T, S)
    assert mask.shape[3] == 4

    # Period 0 (Jan) and period 2 (Mar) should be filled.
    assert (mask[0, :, :, 0, :] == MaskValue.ONLINE_ENCODER.value).all()
    assert (mask[0, :, :, 1, :] == MaskValue.MISSING.value).all()
    assert (mask[0, :, :, 2, :] == MaskValue.ONLINE_ENCODER.value).all()
    assert (mask[0, :, :, 3, :] == MaskValue.MISSING.value).all()


def test_two_modalities_period_alignment() -> None:
    """Two modalities with different image counts align to the same period grid."""
    model = _make_model(period_duration=timedelta(days=30), max_matches=3)

    H, W = 4, 4
    time_range = (datetime(2025, 1, 1), datetime(2025, 4, 1))
    # sentinel2: 3 images covering all 3 periods
    s2_timestamps = [
        (datetime(2025, 1, 15), datetime(2025, 1, 15)),
        (datetime(2025, 2, 15), datetime(2025, 2, 15)),
        (datetime(2025, 3, 15), datetime(2025, 3, 15)),
    ]
    # sentinel1: 1 image in period 1 (Feb) only
    s1_timestamps = [
        (datetime(2025, 2, 10), datetime(2025, 2, 10)),
    ]
    inputs = [
        {
            "sentinel2_l2a": RasterImage(
                image=torch.ones((12, 3, H, W), dtype=torch.float32),
                timestamps=s2_timestamps,
            ),
            "sentinel1": RasterImage(
                image=torch.ones((2, 1, H, W), dtype=torch.float32),
                timestamps=s1_timestamps,
            ),
        }
    ]
    context = ModelContext(
        inputs=inputs,
        metadatas=[_make_metadata((0, 0, H, W), time_range=time_range)],
    )
    sample, present_modalities, _ = model._prepare_modality_inputs(context)

    assert "sentinel2_l2a" in present_modalities
    assert "sentinel1" in present_modalities

    # sentinel2: all 3 periods filled
    s2_mask = sample.sentinel2_l2a_mask
    for t_idx in range(3):
        assert (s2_mask[0, :, :, t_idx, :] == MaskValue.ONLINE_ENCODER.value).all()

    # sentinel1: only period 1 filled, periods 0 and 2 missing
    s1_mask = sample.sentinel1_mask
    assert (s1_mask[0, :, :, 0, :] == MaskValue.MISSING.value).all()
    assert (s1_mask[0, :, :, 1, :] == MaskValue.ONLINE_ENCODER.value).all()
    assert (s1_mask[0, :, :, 2, :] == MaskValue.MISSING.value).all()


def test_max_matches_truncates_periods() -> None:
    """max_matches < available periods keeps only the most recent ones."""
    model = _make_model(period_duration=timedelta(days=30), max_matches=2)

    H, W = 4, 4
    # 4 periods possible, but max_matches=2 keeps only the last 2
    time_range = (datetime(2025, 1, 1), datetime(2025, 5, 1))
    # Images in all 4 months, but only the 2 most recent periods are used.
    timestamps = [
        (datetime(2025, 1, 15), datetime(2025, 1, 15)),
        (datetime(2025, 2, 15), datetime(2025, 2, 15)),
        (datetime(2025, 3, 15), datetime(2025, 3, 15)),
        (datetime(2025, 4, 15), datetime(2025, 4, 15)),
    ]
    inputs = [
        {
            "sentinel2_l2a": RasterImage(
                image=torch.ones((12, 4, H, W), dtype=torch.float32),
                timestamps=timestamps,
            ),
        }
    ]
    context = ModelContext(
        inputs=inputs,
        metadatas=[_make_metadata((0, 0, H, W), time_range=time_range)],
    )
    sample, _, _ = model._prepare_modality_inputs(context)

    # Only 2 period slots
    mask = sample.sentinel2_l2a_mask
    assert mask.shape[3] == 2
    # Both most-recent period slots should be filled (Mar and Apr images match)
    assert (mask[0, :, :, 0, :] == MaskValue.ONLINE_ENCODER.value).all()
    assert (mask[0, :, :, 1, :] == MaskValue.ONLINE_ENCODER.value).all()

    # Timestamps should be period midpoints for the 2 most recent periods
    assert sample.timestamps.shape == (1, 2, 3)


def test_time_range_none_raises() -> None:
    """ValueError should be raised when SampleMetadata.time_range is None."""
    model = _make_model()

    H, W = 4, 4
    inputs = [
        {
            "sentinel2_l2a": RasterImage(
                image=torch.zeros((12, 1, H, W), dtype=torch.float32),
                timestamps=[(datetime(2025, 1, 15), datetime(2025, 1, 15))],
            ),
        }
    ]
    context = ModelContext(
        inputs=inputs,
        metadatas=[_make_metadata((0, 0, H, W), time_range=None)],
    )
    with pytest.raises(ValueError, match="time_range"):
        model._prepare_modality_inputs(context)
