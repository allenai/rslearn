"""Unit tests for rslearn.train.transforms.normalize."""

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
