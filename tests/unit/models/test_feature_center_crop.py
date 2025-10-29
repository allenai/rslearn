"""Tests for the rslearn.models.feature_center_crop module."""

import pytest
import torch

from rslearn.models.feature_center_crop import FeatureCenterCrop


def test_crop_one_feature_map() -> None:
    """Test cropping a feature map to the center 1x1."""
    feat_map = torch.zeros((1, 2, 5, 5), dtype=torch.float32)
    feat_map[:, :, 2, 2] = 1
    feature_crop = FeatureCenterCrop(sizes=[(1, 1)])
    result = feature_crop([feat_map], None)
    assert len(result) == 1
    assert result[0].shape == (1, 2, 1, 1)
    assert result[0][0, 0, 0, 0] == pytest.approx(1)
    assert result[0][0, 1, 0, 0] == pytest.approx(1)
