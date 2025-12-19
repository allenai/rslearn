"""Unit tests for rslearn.train.transforms.select_bands."""

import pytest
import torch

from rslearn.train.transforms.select_bands import SelectBands
from rslearn.utils.raster_format import RasterImage


def test_select_bands_timeseries() -> None:
    """Verify that SelectBands works on a time series."""
    # We select two bands in a three-band image with two images.
    image = torch.zeros((6, 1, 1), dtype=torch.float32)
    for channel_idx in range(image.shape[0]):
        image[channel_idx] = channel_idx
    select_bands = SelectBands(
        band_indices=[0, 2],
        num_bands_per_timestep=3,
    )
    input_dict = {"image": image}
    input_dict, _ = select_bands(input_dict, None)
    result = input_dict["image"]
    assert result.shape == (4, 1, 1)
    assert result[0, 0, 0] == pytest.approx(0)
    assert result[1, 0, 0] == pytest.approx(2)
    assert result[2, 0, 0] == pytest.approx(3)
    assert result[3, 0, 0] == pytest.approx(5)


def test_select_bands_timeseries_raster_image() -> None:
    """Verify that SelectBands works on a time series."""
    # We select two bands in a three-band image with two images.
    image = torch.zeros((3, 2, 1, 1), dtype=torch.float32)
    for channel_idx in range(image.shape[0]):
        image[channel_idx] = channel_idx
    select_bands = SelectBands(
        band_indices=[0, 2],
        num_bands_per_timestep=3,
    )
    input_dict = {"image": RasterImage(image)}
    input_dict, _ = select_bands(input_dict, None)
    result = input_dict["image"].image
    assert result.shape == (2, 2, 1, 1)
    assert result[0, :] == pytest.approx(0)
    assert result[1, :] == pytest.approx(2)
