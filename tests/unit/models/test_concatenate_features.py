"""Tests for the rslearn.models.concatenate_features module."""

import torch

from rslearn.models.concatenate_features import ConcatenateFeatures


def test_concatenate_features() -> None:
    """Test concatenating a feature map with additional features."""
    features = [torch.randn(2, 768, 8, 8)]
    inputs = [
        {"additional_features": torch.randn(2, 32, 32)},
        {"additional_features": torch.randn(2, 32, 32)},
    ]
    concatenate_features = ConcatenateFeatures(key="additional_features")
    result = concatenate_features(features, inputs)
    assert len(result) == 1
    assert result[0].shape == (2, 768 + 2, 8, 8)
