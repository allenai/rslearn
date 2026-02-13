"""Tests for the rslearn.models.pooling_decoder module."""

import torch

from rslearn.models.component import FeatureMaps, FeatureVector
from rslearn.models.pooling_decoder import PoolingDecoder, SegmentationPoolingDecoder
from rslearn.train.model_context import ModelContext, RasterImage


def test_pooling_decoder_with_conv_and_fc_layers() -> None:
    """Test PoolingDecoder with conv and fc layers."""
    batch_size = 2
    embedding_size = 8
    out_channels = 3
    feature_h = 4
    feature_w = 4

    feature_maps = FeatureMaps(
        [
            # BCHW - with multiple feature maps, only last one should be used
            torch.randn(
                (batch_size, embedding_size, feature_h, feature_w), dtype=torch.float32
            ),
            torch.randn(
                (batch_size, embedding_size * 2, feature_h // 2, feature_w // 2),
                dtype=torch.float32,
            ),
        ]
    )

    decoder = PoolingDecoder(
        in_channels=embedding_size * 2,
        out_channels=out_channels,
        num_conv_layers=1,
        num_fc_layers=1,
        conv_channels=16,
        fc_channels=32,
    )

    result = decoder(feature_maps, ModelContext(inputs=[{}], metadatas=[]))

    # Should return FeatureVector
    assert isinstance(result, FeatureVector)
    # Shape should be (batch_size, out_channels)
    assert result.feature_vector.shape == (batch_size, out_channels)


def test_pooling_decoder_no_layers() -> None:
    """Test PoolingDecoder with no conv or fc layers."""
    batch_size = 2
    embedding_size = 8
    out_channels = 3
    feature_h = 4
    feature_w = 4

    feature_maps = FeatureMaps(
        [
            torch.randn(
                (batch_size, embedding_size, feature_h, feature_w), dtype=torch.float32
            ),
        ]
    )

    decoder = PoolingDecoder(
        in_channels=embedding_size,
        out_channels=out_channels,
        num_conv_layers=0,
        num_fc_layers=0,
    )

    result = decoder(feature_maps, ModelContext(inputs=[{}], metadatas=[]))

    assert isinstance(result, FeatureVector)
    assert result.feature_vector.shape == (batch_size, out_channels)


def test_segmentation_pooling_decoder() -> None:
    """Test with 8x2x2 features -> 2x4x4 output."""
    image_size = 4
    image_bands = 3
    patch_size = 2
    embedding_size = 8
    num_classes = 2
    feature_maps = FeatureMaps(
        [
            # BCHW.
            torch.zeros(
                (1, embedding_size, image_size // patch_size, image_size // patch_size),
                dtype=torch.float32,
            ),
        ]
    )
    input_dict = {
        "sentinel2": RasterImage(
            torch.zeros((image_bands, 1, image_size, image_size), dtype=torch.float32)
        ),
    }
    decoder = SegmentationPoolingDecoder(
        in_channels=embedding_size,
        out_channels=num_classes,
        num_fc_layers=1,
        fc_channels=embedding_size,
        image_key="sentinel2",
    )
    result = decoder(
        feature_maps, ModelContext(inputs=[input_dict], metadatas=[])
    ).feature_maps[0]
    assert result.shape == (1, num_classes, image_size, image_size)

    # Output should be the same at all pixels.
    assert torch.all(result[:, :, 0, 0] == result[:, :, 1, 1])


def test_segmentation_pooling_decoder_4d_nonsquare() -> None:
    """Test SegmentationPoolingDecoder with non-square H/W."""
    image_c = 3
    image_t = 1
    image_h = 6
    image_w = 8

    embedding_size = 8
    num_classes = 2
    patch_size = 2

    input_dict = {
        "sentinel2": RasterImage(
            torch.zeros((image_c, image_t, image_h, image_w), dtype=torch.float32)
        ),
    }
    feature_maps = FeatureMaps(
        [
            torch.zeros(
                (1, embedding_size, image_h // patch_size, image_w // patch_size),
                dtype=torch.float32,
            ),
        ]
    )
    decoder = SegmentationPoolingDecoder(
        in_channels=embedding_size,
        out_channels=num_classes,
        num_fc_layers=1,
        fc_channels=embedding_size,
        image_key="sentinel2",
    )
    result = decoder(feature_maps, ModelContext(inputs=[input_dict], metadatas=[]))
    assert len(result.feature_maps) == 1
    assert result.feature_maps[0].shape == (1, num_classes, image_h, image_w)
