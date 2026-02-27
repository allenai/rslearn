"""Unit tests for rslearn.train.transforms.gaussian_noise."""

import torch

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.gaussian_noise import GaussianNoise


def _make_input(C: int = 2, T: int = 5, H: int = 4, W: int = 4) -> dict:
    return {"image": RasterImage(torch.randn(C, T, H, W))}


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


def test_gaussian_noise_preserves_shape() -> None:
    t = GaussianNoise(std=0.01)
    C, T, H, W = 3, 4, 8, 8
    inp = _make_input(C=C, T=T, H=H, W=W)
    inp, _ = t(inp, {})
    assert inp["image"].shape == (C, T, H, W)
