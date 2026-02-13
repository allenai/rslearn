"""Tests for the rslearn.models.fpn module."""

import torch

from rslearn.models.component import FeatureMaps
from rslearn.models.fpn import Fpn
from rslearn.train.model_context import ModelContext


def test_fpn_forward() -> None:
    """Test Fpn forward pass with multi-scale feature maps."""
    # We create a FeatureMaps with three maps that get progressively coarser resolution
    # but deeper.
    in_channels = [8, 16, 32]
    out_channels = 8
    batch_size = 2

    feature_maps = FeatureMaps(
        [
            # Each map is BCHW.
            torch.randn((batch_size, in_channels[0], 32, 32), dtype=torch.float32),
            torch.randn((batch_size, in_channels[1], 16, 16), dtype=torch.float32),
            torch.randn((batch_size, in_channels[2], 8, 8), dtype=torch.float32),
        ]
    )

    fpn = Fpn(in_channels=in_channels, out_channels=out_channels)
    result = fpn(feature_maps, ModelContext(inputs=[], metadatas=[]))

    assert isinstance(result, FeatureMaps)
    assert len(result.feature_maps) == len(feature_maps.feature_maps)
    assert result.feature_maps[0].shape == (batch_size, out_channels, 32, 32)
    assert result.feature_maps[1].shape == (batch_size, out_channels, 16, 16)
    assert result.feature_maps[2].shape == (batch_size, out_channels, 8, 8)


def test_fpn_with_prepend() -> None:
    """Test Fpn with prepend=True."""
    in_channels = [8, 16]
    out_channels = 8
    batch_size = 2

    feature_maps = FeatureMaps(
        [
            # Each map is BCHW.
            torch.randn((batch_size, in_channels[0], 32, 32), dtype=torch.float32),
            torch.randn((batch_size, in_channels[1], 16, 16), dtype=torch.float32),
        ]
    )

    fpn = Fpn(in_channels=in_channels, out_channels=out_channels, prepend=True)
    result = fpn(feature_maps, ModelContext(inputs=[], metadatas=[]))

    assert isinstance(result, FeatureMaps)
    # With prepend=True, output should have FPN outputs + original features
    assert len(result.feature_maps) == 2 * len(feature_maps.feature_maps)
    # First half should be FPN outputs
    assert result.feature_maps[0].shape == (batch_size, out_channels, 32, 32)
    assert result.feature_maps[1].shape == (batch_size, out_channels, 16, 16)
    # Second half should be original features
    assert result.feature_maps[2].shape == (batch_size, in_channels[0], 32, 32)
    assert result.feature_maps[3].shape == (batch_size, in_channels[1], 16, 16)
