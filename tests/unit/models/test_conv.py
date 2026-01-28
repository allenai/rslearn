"""Tests for the rslearn.models.conv module."""

import torch

from rslearn.models.component import FeatureMaps
from rslearn.models.conv import Conv
from rslearn.train.model_context import ModelContext


def test_conv_forward() -> None:
    """Test Conv forward pass with multiple feature maps."""
    batch_size = 1
    in_channels = 8
    out_channels = 16
    feature_h = 8
    feature_w = 8

    feature_maps = FeatureMaps(
        [
            # Each feature map is BCHW.
            torch.randn(
                (batch_size, in_channels, feature_h, feature_w), dtype=torch.float32
            ),
            torch.randn(
                (batch_size, in_channels, feature_h // 2, feature_w // 2),
                dtype=torch.float32,
            ),
        ]
    )

    conv = Conv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        padding="same",
        stride=1,
    )

    result = conv(feature_maps, ModelContext(inputs=[], metadatas=[]))

    assert isinstance(result, FeatureMaps)
    assert len(result.feature_maps) == 2
    assert result.feature_maps[0].shape == (
        batch_size,
        out_channels,
        feature_h,
        feature_w,
    )
    assert result.feature_maps[1].shape == (
        batch_size,
        out_channels,
        feature_h // 2,
        feature_w // 2,
    )


def test_conv_with_custom_activation() -> None:
    """Test Conv with custom activation function."""
    batch_size = 1
    in_channels = 8
    out_channels = 16
    feature_h = 8
    feature_w = 8

    feature_maps = FeatureMaps(
        [
            torch.randn(
                (batch_size, in_channels, feature_h, feature_w), dtype=torch.float32
            ),
        ]
    )

    conv = Conv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        activation=torch.nn.Tanh(),
    )

    result = conv(feature_maps, ModelContext(inputs=[], metadatas=[]))

    assert isinstance(result, FeatureMaps)
    assert len(result.feature_maps) == 1
    assert result.feature_maps[0].shape == (
        batch_size,
        out_channels,
        feature_h,
        feature_w,
    )
    # Verify activation is applied: Tanh outputs values in [-1, 1]
    assert torch.all(result.feature_maps[0] >= -1.0)
    assert torch.all(result.feature_maps[0] <= 1.0)


def test_conv_with_stride() -> None:
    """Test Conv with stride > 1."""
    batch_size = 1
    in_channels = 8
    out_channels = 16
    feature_h = 8
    feature_w = 8

    feature_maps = FeatureMaps(
        [
            torch.randn(
                (batch_size, in_channels, feature_h, feature_w), dtype=torch.float32
            ),
        ]
    )

    conv = Conv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        padding=1,
        stride=2,
    )

    result = conv(feature_maps, ModelContext(inputs=[], metadatas=[]))

    assert isinstance(result, FeatureMaps)
    assert len(result.feature_maps) == 1
    # With stride=2, output size should be halved
    assert result.feature_maps[0].shape == (
        batch_size,
        out_channels,
        feature_h // 2,
        feature_w // 2,
    )
