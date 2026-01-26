"""Unit tests for rslearn.train.transforms.mask."""

import torch

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.mask import Mask


def test_mask_to_zero() -> None:
    """Test Mask with default arguments where image should be set 0."""
    mask = Mask()
    input_dict = {
        "image": RasterImage(torch.ones((1, 1, 2, 2), dtype=torch.float32)),
        "mask": RasterImage(torch.tensor([[[[0, 1], [0, 2]]]], dtype=torch.int32)),
    }
    input_dict, _ = mask(input_dict, {})
    assert torch.all(input_dict["image"].image == torch.tensor([[[[0, 1], [0, 1]]]]))


def test_mask_multi_band() -> None:
    """Test masking when the mask is single-band but image has multiple bands."""
    mask = Mask()
    input_dict = {
        "image": RasterImage(torch.ones((2, 1, 2, 2), dtype=torch.float32)),
        "mask": RasterImage(torch.tensor([[[[0, 1], [0, 1]]]], dtype=torch.int32)),
    }
    input_dict, _ = mask(input_dict, {})
    assert torch.all(
        input_dict["image"].image
        == torch.tensor([[[[0, 1], [0, 1]]], [[[0, 1], [0, 1]]]])
    )


def test_mask_custom_value() -> None:
    """Test Mask with a non-zero target value and selector."""
    mask = Mask(selectors=["target/custom_image"], mask_value=2)
    input_dict = {
        "image": RasterImage(torch.ones((1, 1, 2, 2))),
        "mask": RasterImage(torch.tensor([[[[0, 1], [1, 1]]]])),
    }
    target_dict = {
        "custom_image": RasterImage(torch.ones((1, 1, 2, 2))),
    }
    input_dict, target_dict = mask(input_dict, target_dict)
    assert torch.all(input_dict["image"].image == torch.ones((1, 1, 2, 2)))
    assert torch.all(
        target_dict["custom_image"].image == torch.tensor([[[[2, 1], [1, 1]]]])
    )
