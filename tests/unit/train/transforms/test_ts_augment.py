"""Unit tests for rslearn.train.transforms.ts_augment."""

import torch

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.ts_augment import (
    GaussianNoise,
    RandomTimeMasking,
    TemporalShift,
)


def _make_input(C: int = 2, T: int = 5, H: int = 4, W: int = 4) -> dict:
    return {"image": RasterImage(torch.randn(C, T, H, W))}


def test_random_time_masking_zeros_some_timesteps() -> None:
    torch.manual_seed(0)
    t = RandomTimeMasking(mask_ratio=0.5)
    inp = _make_input()
    inp, _ = t(inp, {})
    # At least one timestep must survive.
    assert (inp["image"].image.abs().sum(dim=(0, 2, 3)) > 0).any()


def test_random_time_masking_single_timestep_noop() -> None:
    t = RandomTimeMasking(mask_ratio=0.9)
    inp = _make_input(T=1)
    orig = inp["image"].image.clone()
    inp, _ = t(inp, {})
    assert torch.equal(inp["image"].image, orig)


def test_random_time_masking_custom_value() -> None:
    torch.manual_seed(0)
    t = RandomTimeMasking(mask_ratio=0.5, mask_value=-9999.0)
    inp = _make_input()
    inp, _ = t(inp, {})
    img = inp["image"].image
    masked_ts = (img == -9999.0).all(dim=(0, 2, 3))
    assert masked_ts.any(), "At least one timestep should be masked with -9999"


def test_random_time_masking_append_mask_channel() -> None:
    torch.manual_seed(0)
    C, T = 2, 5
    t = RandomTimeMasking(mask_ratio=0.5, append_mask_channel=True)
    inp = _make_input(C=C, T=T)
    orig = inp["image"].image.clone()
    inp, _ = t(inp, {})
    img = inp["image"].image
    # Channel count should increase by 1 (the mask channel is prepended).
    assert img.shape[0] == C + 1
    mask_ch = img[0]  # [T, H, W]
    data_ch = img[1:]  # [C, T, H, W]
    # Mask channel values should be 0 or 1.
    assert set(mask_ch.unique().tolist()).issubset({0.0, 1.0})
    # Where mask_ch == 0 (masked), all data channels should equal mask_value (0).
    for ti in range(T):
        if mask_ch[ti, 0, 0].item() == 0.0:
            assert (data_ch[:, ti] == 0.0).all(), (
                f"Masked timestep {ti} should be zeroed"
            )
        else:
            assert torch.equal(data_ch[:, ti], orig[:, ti]), (
                f"Valid timestep {ti} should be unchanged"
            )


def test_random_time_masking_append_mask_channel_with_custom_value() -> None:
    """Mask channel should be consistent when using a custom mask_value."""
    torch.manual_seed(1)
    C, T = 3, 8
    mask_value = -9999.0
    t = RandomTimeMasking(
        mask_ratio=0.5, mask_value=mask_value, append_mask_channel=True
    )
    inp = _make_input(C=C, T=T)
    inp, _ = t(inp, {})
    img = inp["image"].image
    mask_ch = img[0]  # [T, H, W]
    data_ch = img[1:]  # [C, T, H, W]
    for ti in range(T):
        if mask_ch[ti, 0, 0].item() == 0.0:
            assert (data_ch[:, ti] == mask_value).all()
        else:
            assert (data_ch[:, ti] != mask_value).any()


def test_temporal_shift_preserves_shape_and_values() -> None:
    t = TemporalShift(max_shift=1, pad_mode="edge")
    inp = _make_input(T=4)
    shape_before = inp["image"].shape
    values_before = set(inp["image"].image.flatten().tolist())
    inp, _ = t(inp, {})
    # Shape is preserved (edge padding keeps T constant).
    assert inp["image"].shape == shape_before
    # All output values come from the original tensor (edge repeats existing values).
    values_after = set(inp["image"].image.flatten().tolist())
    assert values_after.issubset(values_before)


def test_gaussian_noise_changes_tensor() -> None:
    torch.manual_seed(42)
    t = GaussianNoise(std=0.1)
    inp = _make_input()
    orig = inp["image"].image.clone()
    inp, _ = t(inp, {})
    assert not torch.equal(inp["image"].image, orig)


def test_gaussian_noise_magnitude() -> None:
    torch.manual_seed(42)
    std = 0.05
    t = GaussianNoise(std=std)
    inp = _make_input(C=1, T=1, H=64, W=64)
    orig = inp["image"].image.clone()
    inp, _ = t(inp, {})
    diff_std = (inp["image"].image - orig).std().item()
    assert abs(diff_std - std) < 0.02
