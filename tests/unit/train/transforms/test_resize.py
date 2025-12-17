from typing import Any

import torch

from rslearn.train.transforms.resize import Resize


def test_resize() -> None:
    """Verify that converting to decibels works."""
    target_class = torch.tensor(
        [[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]], dtype=torch.float32
    )

    resize_transform = Resize((2, 2), ["target/image"], "nearest")
    input_dict: dict[str, Any] = {}
    target_dict: dict[str, Any] = {"image": target_class}
    tsf_input, tsf_target = resize_transform(input_dict, target_dict)
    expected_target = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32)
    assert torch.all(tsf_target["image"] == expected_target)
    assert tsf_input == {}
