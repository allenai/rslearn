from typing import Any

import torch

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.resize import MaxPoolResize, Resize


def test_resize() -> None:
    """Verify that converting to decibels works."""
    target_class_4D = torch.tensor(
        [[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]], dtype=torch.float32
    ).unsqueeze(dim=1)
    resize_transform = Resize((2, 2), ["target/image"], "nearest")
    input_dict: dict[str, Any] = {}
    target_dict_4D: dict[str, Any] = {"image": RasterImage(target_class_4D)}
    tsf_input_4D, tsf_target_4D = resize_transform(input_dict, target_dict_4D)
    expected_target_4D = torch.tensor(
        [[[1, 2], [3, 4]]], dtype=torch.float32
    ).unsqueeze(dim=1)
    assert torch.all(tsf_target_4D["image"].image == expected_target_4D)
    assert tsf_input_4D == {}


def test_max_pool_resize_all_zeros() -> None:
    """MaxPoolResize on an all-zero label mask should produce all zeros."""
    # CTHW: 1 channel, 1 timestep, 10x10 spatial — all zeros
    label = torch.zeros(1, 1, 10, 10, dtype=torch.int32)
    transform = MaxPoolResize((2, 2), ["target/classes"])
    input_dict: dict[str, Any] = {}
    target_dict: dict[str, Any] = {"classes": RasterImage(label)}
    _, out_target = transform(input_dict, target_dict)
    assert out_target["classes"].image.shape == (1, 1, 2, 2)
    assert (out_target["classes"].image == 0).all()


def test_max_pool_resize_single_positive_pixel() -> None:
    """MaxPoolResize with one positive pixel should propagate a 1 to that pool region."""
    # CTHW: 1 channel, 1 timestep, 10x10 spatial — all zeros except one pixel
    label = torch.zeros(1, 1, 10, 10, dtype=torch.int32)
    label[0, 0, 3, 7] = 1  # falls in pool region (0, 1) for a 2x2 output
    transform = MaxPoolResize((2, 2), ["target/classes"])
    input_dict: dict[str, Any] = {}
    target_dict: dict[str, Any] = {"classes": RasterImage(label)}
    _, out_target = transform(input_dict, target_dict)
    assert out_target["classes"].image.shape == (1, 1, 2, 2)
    # The output should not be all zeros — the region containing the positive pixel is 1
    assert out_target["classes"].image.sum().item() == 1
    assert out_target["classes"].image[0, 0, 0, 1] == 1
