"""Unit tests for rslearn.train.transforms.select_bands."""

import pytest
import torch

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.select_bands import SelectBands


def test_select_bands() -> None:
    """Verify that SelectBands selects from channel dimension."""
    # CTHW: 3 channels, 2 timesteps, 1x1 spatial
    image = torch.zeros((3, 2, 1, 1), dtype=torch.float32)
    for channel_idx in range(image.shape[0]):
        image[channel_idx] = channel_idx
    select_bands = SelectBands(band_indices=[0, 2])
    input_dict = {"image": RasterImage(image)}
    input_dict, _ = select_bands(input_dict, None)
    result = input_dict["image"].image
    # Should have 2 channels now, same timesteps
    assert result.shape == (2, 2, 1, 1)
    # Channel 0 should have value 0, channel 1 (was channel 2) should have value 2
    assert result[0, :] == pytest.approx(0)
    assert result[1, :] == pytest.approx(2)
