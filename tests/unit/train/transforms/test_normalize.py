"""Unit tests for rslearn.train.transforms.normalize."""

import pytest
import torch

from rslearn.train.transforms.normalize import Normalize


def test_normalize_time_series() -> None:
    """Verify that the normalization is repeated on all images in the time series."""
    # We only apply normalization on band 0, not band 1.
    # So on each timestep the normalization should be applied to band 0.
    normalize = Normalize(
        mean=0,
        std=2,
        bands=[0],
        num_bands=2,
    )
    input_dict = {
        "image": torch.ones((4, 3, 3), dtype=torch.float32),
    }
    input_dict, _ = normalize(input_dict, None)
    eps = 1e-6
    assert torch.all(torch.abs(input_dict["image"][(0, 2), :, :] - 0.5) < eps)
    assert torch.all(torch.abs(input_dict["image"][(1, 3), :, :] - 1.0) < eps)


def test_scalar_mean_and_std_on_time_series() -> None:
    """Make sure scalar mean and std work on time series."""
    normalize = Normalize(
        mean=0,
        std=2,
        num_bands=2,
    )
    input_dict = {
        "image": torch.ones((4, 1, 1), dtype=torch.float32),
    }
    input_dict, _ = normalize(input_dict, None)
    result = input_dict["image"]
    assert result[0, 0, 0] == pytest.approx(0.5)
    assert result[1, 0, 0] == pytest.approx(0.5)
    assert result[2, 0, 0] == pytest.approx(0.5)
    assert result[3, 0, 0] == pytest.approx(0.5)


def test_list_mean_and_std_on_time_series() -> None:
    """Make sure list of means and stds works on time series."""
    normalize = Normalize(
        mean=[0, 1],
        std=[2, 1],
        num_bands=2,
    )
    input_dict = {
        "image": torch.ones((4, 1, 1), dtype=torch.float32),
    }
    input_dict, _ = normalize(input_dict, None)
    result = input_dict["image"]
    # First band in each image should be (1-0)/2 = 0.5.
    assert result[0, 0, 0] == pytest.approx(0.5)
    assert result[2, 0, 0] == pytest.approx(0.5)
    # Second band should be (1-1)/1 = 0.0.
    assert result[1, 0, 0] == pytest.approx(0.0)
    assert result[3, 0, 0] == pytest.approx(0.0)


def test_list_mean_and_std_with_band_indices() -> None:
    """Make sure list of means and stds works when passing band index list."""
    normalize = Normalize(
        mean=[0, 1],
        std=[2, 1],
        bands=[0, 2],
        num_bands=3,
    )
    input_dict = {
        "image": torch.ones((6, 1, 1), dtype=torch.float32),
    }
    input_dict, _ = normalize(input_dict, None)
    result = input_dict["image"]
    # First band should be (1-0)/2 = 0.5.
    assert result[0, 0, 0] == pytest.approx(0.5)
    assert result[3, 0, 0] == pytest.approx(0.5)
    # Second band should be unchanged.
    assert result[1, 0, 0] == pytest.approx(1.0)
    assert result[4, 0, 0] == pytest.approx(1.0)
    # Third band should be (1-1)/1 = 0.0.
    assert result[2, 0, 0] == pytest.approx(0.0)
    assert result[5, 0, 0] == pytest.approx(0.0)
