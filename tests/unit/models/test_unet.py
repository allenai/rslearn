"""Tests for the rslearn.models.unet module."""

import pytest
import torch

from rslearn.models.component import FeatureMaps
from rslearn.models.unet import UNetDecoder
from rslearn.train.model_context import ModelContext


class TestUNetDecoder:
    """Test class for UNetDecoder."""

    # Share this batch size, image size, and feature map sizes across all tests here.
    BATCH_SIZE = 1
    IMAGE_SIZE = 64
    # Multi-scale features: (downsample_factor, channels)
    IN_CHANNELS = [(4, 16), (8, 32), (16, 64)]
    OUT_CHANNELS = 16

    @pytest.fixture
    def feature_maps(self) -> FeatureMaps:
        """Create multi-scale feature maps fixture."""
        return FeatureMaps(
            [
                torch.randn(
                    (self.BATCH_SIZE, 16, self.IMAGE_SIZE // 4, self.IMAGE_SIZE // 4),
                    dtype=torch.float32,
                ),  # 1/4 resolution
                torch.randn(
                    (self.BATCH_SIZE, 32, self.IMAGE_SIZE // 8, self.IMAGE_SIZE // 8),
                    dtype=torch.float32,
                ),  # 1/8 resolution
                torch.randn(
                    (self.BATCH_SIZE, 64, self.IMAGE_SIZE // 16, self.IMAGE_SIZE // 16),
                    dtype=torch.float32,
                ),  # 1/16 resolution
            ]
        )

    def test_unet_decoder_simple(self, feature_maps: FeatureMaps) -> None:
        """Test UNetDecoder forward pass with multi-scale features."""
        decoder = UNetDecoder(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            conv_layers_per_resolution=1,
            kernel_size=3,
        )
        result = decoder(feature_maps, ModelContext(inputs=[], metadatas=[]))

        assert isinstance(result, FeatureMaps)
        assert len(result.feature_maps) == 1
        # Output should be at the input resolution.
        assert result.feature_maps[0].shape == (
            self.BATCH_SIZE,
            self.OUT_CHANNELS,
            self.IMAGE_SIZE,
            self.IMAGE_SIZE,
        )

    def test_unet_decoder_with_target_resolution(
        self, feature_maps: FeatureMaps
    ) -> None:
        """Test UNetDecoder with target_resolution_factor > 1."""
        target_resolution_factor = 2  # Stop upsampling at 1/2 resolution

        decoder = UNetDecoder(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            target_resolution_factor=target_resolution_factor,
        )
        result = decoder(feature_maps, ModelContext(inputs=[], metadatas=[]))

        assert isinstance(result, FeatureMaps)
        assert len(result.feature_maps) == 1
        # Output should be at 1/2 the image resolution.
        assert result.feature_maps[0].shape == (
            self.BATCH_SIZE,
            self.OUT_CHANNELS,
            self.IMAGE_SIZE // 2,
            self.IMAGE_SIZE // 2,
        )

    def test_unet_decoder_with_interpolation(self, feature_maps: FeatureMaps) -> None:
        """Test UNetDecoder with original_size_to_interpolate."""
        original_size = (48, 48)

        decoder = UNetDecoder(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            original_size_to_interpolate=original_size,
        )
        result = decoder(feature_maps, ModelContext(inputs=[], metadatas=[]))

        assert isinstance(result, FeatureMaps)
        assert len(result.feature_maps) == 1
        # Output should be interpolated to original_size.
        assert result.feature_maps[0].shape == (
            self.BATCH_SIZE,
            self.OUT_CHANNELS,
            original_size[0],
            original_size[1],
        )
