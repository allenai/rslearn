from typing import Any

import torch

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.resize import Resize


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
