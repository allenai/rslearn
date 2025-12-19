from typing import Any

import pytest
import torch

from rslearn.train.transforms.resize import Resize
from rslearn.utils.raster_format import RasterImage


def test_resize() -> None:
    """Verify that converting to decibels works."""
    target_class_3D = torch.tensor(
        [[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]], dtype=torch.float32
    )
    target_class_2D = torch.tensor(
        [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=torch.float32
    )
    resize_transform = Resize((2, 2), ["target/image"], "nearest")
    input_dict: dict[str, Any] = {}
    target_dict_3D: dict[str, Any] = {"image": target_class_3D}
    target_dict_2D: dict[str, Any] = {"image": target_class_2D}
    tsf_input_3D, tsf_target_3D = resize_transform(input_dict, target_dict_3D)
    tsf_input_2D, tsf_target_2D = resize_transform(input_dict, target_dict_2D)
    expected_target_3D = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32)
    expected_target_2D = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    assert torch.all(tsf_target_3D["image"] == expected_target_3D)
    assert torch.all(tsf_target_2D["image"] == expected_target_2D)
    assert tsf_input_3D == {}
    assert tsf_input_2D == {}


def test_resize_rasterimage() -> None:
    """Verify that converting to decibels works."""
    target_class_4D = torch.tensor(
        [[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]], dtype=torch.float32
    ).unsqueeze(dim=1)
    target_class_2D = torch.tensor(
        [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=torch.float32
    )
    resize_transform = Resize((2, 2), ["target/image"], "nearest")
    input_dict: dict[str, Any] = {}
    target_dict_4D: dict[str, Any] = {"image": RasterImage(target_class_4D)}
    target_dict_2D: dict[str, Any] = {"image": RasterImage(target_class_2D)}
    tsf_input_4D, tsf_target_4D = resize_transform(input_dict, target_dict_4D)
    with pytest.raises(AssertionError):
        # we expect an error to be raised for 2D raster images since they
        # should always be 4D
        resize_transform(input_dict, target_dict_2D)
    expected_target_4D = torch.tensor(
        [[[1, 2], [3, 4]]], dtype=torch.float32
    ).unsqueeze(dim=1)
    assert torch.all(tsf_target_4D["image"].image == expected_target_4D)
    assert tsf_input_4D == {}
