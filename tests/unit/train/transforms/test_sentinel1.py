"""Unit tests for rslearn.train.transforms.sentinel1."""

import pytest
import torch

from rslearn.train.transforms.sentinel1 import Sentinel1ToDecibels
from rslearn.utils.raster_format import RasterImage


def test_to_decibels() -> None:
    """Verify that converting to decibels works."""
    image = torch.zeros((2, 1, 1), dtype=torch.float32)
    image[0, 0, 0] = 0.1
    image[1, 0, 0] = 1.0
    to_decibels = Sentinel1ToDecibels()
    input_dict = {"image": image}
    input_dict, _ = to_decibels(input_dict, None)
    result = input_dict["image"]
    assert result[0, 0, 0] == pytest.approx(-10)
    assert result[1, 0, 0] == pytest.approx(0)


def test_to_decibels_rasterimage() -> None:
    """Verify that converting to decibels works."""
    image = torch.zeros((2, 1, 1), dtype=torch.float32)
    image[0, 0, 0] = 0.1
    image[1, 0, 0] = 1.0
    to_decibels = Sentinel1ToDecibels()
    input_dict = {"image": RasterImage(image)}
    input_dict, _ = to_decibels(input_dict, None)
    result = input_dict["image"].image
    assert result[0, 0, 0] == pytest.approx(-10)
    assert result[1, 0, 0] == pytest.approx(0)
