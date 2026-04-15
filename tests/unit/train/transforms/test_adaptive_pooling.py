"""Unit tests for rslearn.train.transforms.adaptive_pooling."""

from typing import Any

import torch

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.adaptive_pooling import AdaptivePooling


def test_adaptive_pooling_max_all_zeros() -> None:
    """Max adaptive pooling on an all-zero label mask should produce all zeros."""
    label = torch.zeros(1, 1, 10, 10, dtype=torch.int32)
    transform = AdaptivePooling((2, 2), ["target/classes"], pooling="max")
    input_dict: dict[str, Any] = {}
    target_dict: dict[str, Any] = {"classes": RasterImage(label)}
    _, out_target = transform(input_dict, target_dict)
    assert out_target["classes"].image.shape == (1, 1, 2, 2)
    assert (out_target["classes"].image == 0).all()


def test_adaptive_pooling_max_single_positive_pixel() -> None:
    """Max adaptive pooling should propagate a 1 to the matching pooled region."""
    label = torch.zeros(1, 1, 10, 10, dtype=torch.int32)
    label[0, 0, 3, 7] = 1
    transform = AdaptivePooling((2, 2), ["target/classes"], pooling="max")
    input_dict: dict[str, Any] = {}
    target_dict: dict[str, Any] = {"classes": RasterImage(label)}
    _, out_target = transform(input_dict, target_dict)
    assert out_target["classes"].image.shape == (1, 1, 2, 2)
    assert out_target["classes"].image.sum().item() == 1
    assert out_target["classes"].image[0, 0, 0, 1] == 1


def test_adaptive_pooling_max_bool_multi_timestep() -> None:
    """Max adaptive pooling should pool each boolean timestep independently."""
    label = torch.zeros(1, 2, 4, 4, dtype=torch.bool)
    label[0, 0, 0, 0] = True
    label[0, 1, 3, 2] = True

    transform = AdaptivePooling((2, 2), ["target/classes"], pooling="max")
    input_dict: dict[str, Any] = {}
    target_dict: dict[str, Any] = {"classes": RasterImage(label)}
    _, out_target = transform(input_dict, target_dict)

    expected = torch.tensor(
        [[[[True, False], [False, False]], [[False, False], [False, True]]]],
        dtype=torch.bool,
    )
    assert out_target["classes"].image.shape == (1, 2, 2, 2)
    assert out_target["classes"].image.dtype == torch.bool
    assert torch.equal(out_target["classes"].image, expected)


def test_adaptive_pooling_mean_float_multi_timestep() -> None:
    """Mean adaptive pooling should average each timestep independently."""
    image = torch.tensor(
        [
            [
                [[1, 1, 3, 3], [1, 1, 3, 3], [5, 5, 7, 7], [5, 5, 7, 7]],
                [[2, 2, 4, 4], [2, 2, 4, 4], [6, 6, 8, 8], [6, 6, 8, 8]],
            ]
        ],
        dtype=torch.float32,
    )

    transform = AdaptivePooling((2, 2), ["target/image"], pooling="mean")
    input_dict: dict[str, Any] = {}
    target_dict: dict[str, Any] = {"image": RasterImage(image)}
    _, out_target = transform(input_dict, target_dict)

    expected = torch.tensor(
        [[[[1, 3], [5, 7]], [[2, 4], [6, 8]]]],
        dtype=torch.float32,
    )
    assert out_target["image"].image.shape == (1, 2, 2, 2)
    assert out_target["image"].image.dtype == torch.float32
    assert torch.equal(out_target["image"].image, expected)
