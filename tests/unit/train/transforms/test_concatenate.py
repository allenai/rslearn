"""Unit tests for rslearn.train.transforms.concatenate."""

import torch

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.concatenate import Concatenate, ConcatenateDim


def test_concatenate_time() -> None:
    """Test Mask with default arguments where image should be set 0."""
    concat = Concatenate(
        selections={"image1": [0], "image2": [0]},
        concatenate_dim=ConcatenateDim.TIME,
        output_selector="image3",
    )
    input_image = torch.ones((2, 4, 2, 2), dtype=torch.float32)
    input_image[0] = 0
    input_dict = {
        "image1": RasterImage(input_image.clone()),
        "image2": RasterImage(input_image.clone()),
    }
    input_dict, _ = concat(input_dict, {})
    assert input_dict["image3"].shape == (1, 8, 2, 2)
    assert (input_dict["image3"].image == 0).all()


def test_concatenate_channels() -> None:
    """Test Mask with default arguments where image should be set 0."""
    concat = Concatenate(
        selections={"image1": [0], "image2": [0]},
        concatenate_dim=ConcatenateDim.CHANNEL,
        output_selector="image3",
    )
    input_image = torch.ones((2, 4, 2, 2), dtype=torch.float32)
    input_image[0] = 0
    input_dict = {
        "image1": RasterImage(input_image.clone()),
        "image2": RasterImage(input_image.clone()),
    }
    input_dict, _ = concat(input_dict, {})
    assert input_dict["image3"].shape == (2, 4, 2, 2)
    assert (input_dict["image3"].image == 0).all()
