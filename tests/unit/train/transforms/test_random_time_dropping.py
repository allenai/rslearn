"""Unit tests for rslearn.train.transforms.random_time_dropping."""

from datetime import datetime, timedelta

import torch

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.random_time_dropping import RandomTimeDropping


def _make_timestamps(n: int) -> list[tuple[datetime, datetime]]:
    """Create n sequential daily timestamps."""
    base = datetime(2024, 6, 1)
    return [(base + timedelta(days=i), base + timedelta(days=i)) for i in range(n)]


def _make_input(
    C: int = 2,
    T: int = 5,
    H: int = 4,
    W: int = 4,
    with_timestamps: bool = False,
    with_expected_timestamps: bool = False,
) -> dict:
    timestamps = _make_timestamps(T) if with_timestamps else None
    expected_timestamps = _make_timestamps(T) if with_expected_timestamps else None
    return {
        "image": RasterImage(
            torch.randn(C, T, H, W),
            timestamps=timestamps,
            expected_timestamps=expected_timestamps,
        )
    }


def test_drops_some_timesteps() -> None:
    """With a high drop ratio, at least one timestep should be removed.

    Also verifies that timestamps shrink together with the tensor and that
    expected_timestamps are left unchanged.
    """
    torch.manual_seed(0)
    T = 10
    rand_drop = RandomTimeDropping(drop_ratio=0.8, min_keep=1)
    inp = _make_input(T=T, with_timestamps=True, with_expected_timestamps=True)
    expected_before = list(inp["image"].expected_timestamps)
    original_T = inp["image"].shape[1]
    inp, _ = rand_drop(inp, {})
    assert inp["image"].shape[1] < original_T
    # timestamps must shrink together with the tensor
    assert len(inp["image"].timestamps) == inp["image"].shape[1]
    # expected_timestamps must NOT be modified (model uses the mismatch)
    assert inp["image"].expected_timestamps == expected_before
    assert len(inp["image"].expected_timestamps) == T


def test_min_keep_respected() -> None:
    """Even with drop_ratio=1.0, min_keep timesteps survive."""
    torch.manual_seed(0)
    t = RandomTimeDropping(drop_ratio=1.0, min_keep=3)
    inp = _make_input(T=8)
    inp, _ = t(inp, {})
    assert inp["image"].shape[1] >= 3


def test_spatial_dims_preserved() -> None:
    """C, H, W should be unchanged after dropping."""
    torch.manual_seed(42)
    C, T, H, W = 3, 10, 8, 8
    t = RandomTimeDropping(drop_ratio=0.5)
    inp = _make_input(C=C, T=T, H=H, W=W)
    inp, _ = t(inp, {})
    assert inp["image"].shape[0] == C
    assert inp["image"].shape[2] == H
    assert inp["image"].shape[3] == W


def test_drop_ratio_zero_is_noop() -> None:
    """drop_ratio=0 should never drop anything."""
    t = RandomTimeDropping(drop_ratio=0.0)
    inp = _make_input(T=5)
    orig = inp["image"].image.clone()
    inp, _ = t(inp, {})
    assert torch.equal(inp["image"].image, orig)
