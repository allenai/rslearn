"""Tests for the rslearn.models.concatenate_features module."""

import torch

from rslearn.models.concatenate_features import ConcatenateFeatures


def test_concatenate_features_with_conv_layer() -> None:
    """Test concatenating a feature map with additional features (with conv layer)."""
    features = [
        torch.randn(2, 256, 32, 32),
        torch.randn(2, 512, 16, 16),
        torch.randn(2, 768, 8, 8),
        torch.randn(2, 1024, 4, 4),
    ]
    inputs = [
        {"input_key": torch.randn(2, 32, 32)},
        {"input_key": torch.randn(2, 32, 32)},
    ]
    concatenate_features = ConcatenateFeatures(
        key="input_key",
        in_channels=2,
        conv_channels=8,
        out_channels=4,
        num_conv_layers=2,
        kernel_size=3,
    )
    result = concatenate_features(features, inputs)
    assert len(result) == 4
    assert result[0].shape == (2, 256 + 4, 32, 32)
    assert result[1].shape == (2, 512 + 4, 16, 16)
    assert result[2].shape == (2, 768 + 4, 8, 8)
    assert result[3].shape == (2, 1024 + 4, 4, 4)


def test_concatenate_features_without_conv_layer() -> None:
    """Test concatenating a feature map with additional features (without conv layer)."""
    features = [
        torch.randn(2, 256, 32, 32),
        torch.randn(2, 512, 16, 16),
        torch.randn(2, 768, 8, 8),
        torch.randn(2, 1024, 4, 4),
    ]
    inputs = [
        {"input_key": torch.randn(2, 32, 32)},
        {"input_key": torch.randn(2, 32, 32)},
    ]
    concatenate_features = ConcatenateFeatures(
        key="input_key",
        num_conv_layers=0,
    )
    result = concatenate_features(features, inputs)
    assert len(result) == 4
    assert result[0].shape == (2, 256 + 2, 32, 32)
    assert result[1].shape == (2, 512 + 2, 16, 16)
    assert result[2].shape == (2, 768 + 2, 8, 8)
    assert result[3].shape == (2, 1024 + 2, 4, 4)
