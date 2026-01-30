"""Unit tests for rslearn.train.transforms.normalize."""

import pytest
import torch

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.normalize import Normalize


def test_normalize_specific_bands() -> None:
    """Verify normalization applies only to specified band indices."""
    # Normalize only band 0, leave band 1 unchanged.
    normalize = Normalize(
        mean=0,
        std=2,
        bands=[0],
    )
    # CTHW: 2 channels, 2 timesteps, 3x3 spatial
    input_dict = {
        "image": RasterImage(torch.ones((2, 2, 3, 3), dtype=torch.float32)),
    }
    input_dict, _ = normalize(input_dict, None)
    eps = 1e-6
    # Band 0 should be normalized: (1-0)/2 = 0.5
    assert torch.all(torch.abs(input_dict["image"].image[0, :, :, :] - 0.5) < eps)
    # Band 1 should be unchanged
    assert torch.all(torch.abs(input_dict["image"].image[1, :, :, :] - 1.0) < eps)


def test_scalar_mean_and_std() -> None:
    """Make sure scalar mean and std broadcast over all dimensions."""
    normalize = Normalize(
        mean=0,
        std=2,
    )
    # CTHW: 2 channels, 2 timesteps
    input_dict = {
        "image": RasterImage(torch.ones((2, 2, 1, 1), dtype=torch.float32)),
    }
    input_dict, _ = normalize(input_dict, None)
    result = input_dict["image"].image
    # All values should be (1-0)/2 = 0.5
    assert result[0, 0, 0, 0] == pytest.approx(0.5)
    assert result[0, 1, 0, 0] == pytest.approx(0.5)
    assert result[1, 0, 0, 0] == pytest.approx(0.5)
    assert result[1, 1, 0, 0] == pytest.approx(0.5)


def test_per_channel_mean_and_std() -> None:
    """Make sure per-channel mean/std broadcasts over T, H, W."""
    # Different normalization per channel
    normalize = Normalize(
        mean=[0, 1],
        std=[2, 1],
    )
    # CTHW: 2 channels, 3 timesteps
    input_dict = {
        "image": RasterImage(torch.ones((2, 3, 1, 1), dtype=torch.float32)),
    }
    input_dict, _ = normalize(input_dict, None)
    result = input_dict["image"].image
    # Channel 0: (1-0)/2 = 0.5 for all timesteps
    assert result[0, :, 0, 0] == pytest.approx(0.5)
    # Channel 1: (1-1)/1 = 0.0 for all timesteps
    assert result[1, :, 0, 0] == pytest.approx(0.0)


def test_per_channel_with_band_indices() -> None:
    """Make sure per-channel mean/std works with specific band indices."""
    # Normalize bands 0 and 2 with different params
    normalize = Normalize(
        mean=[0, 1],
        std=[2, 1],
        bands=[0, 2],
    )
    # CTHW: 3 channels, 2 timesteps
    input_dict = {
        "image": RasterImage(torch.ones((3, 2, 1, 1), dtype=torch.float32)),
    }
    input_dict, _ = normalize(input_dict, None)
    result = input_dict["image"].image
    # Band 0: (1-0)/2 = 0.5
    assert result[0, :, 0, 0] == pytest.approx(0.5)
    # Band 1: unchanged
    assert result[1, :, 0, 0] == pytest.approx(1.0)
    # Band 2: (1-1)/1 = 0.0
    assert result[2, :, 0, 0] == pytest.approx(0.0)
