"""Tests for the rslearn.models.resize_features module."""

import torch

from rslearn.models.resize_features import ResizeFeatures


def test_resize_two_feature_maps() -> None:
    """Resize two feature maps.

    - 6x6 -> 8x8
    - 3x3 -> 4x4
    """
    feature_maps = [
        torch.zeros((1, 2, 6, 6), dtype=torch.float32),
        torch.zeros((1, 4, 3, 3), dtype=torch.float32),
    ]
    resize_features = ResizeFeatures(out_sizes=[(8, 8), (4, 4)])
    result = resize_features(feature_maps, None)
    assert result[0].shape == (1, 2, 8, 8)
    assert result[1].shape == (1, 4, 4, 4)
