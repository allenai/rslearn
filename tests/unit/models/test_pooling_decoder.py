"""Tests for the rslearn.models.pooling_decoder module."""

import torch

from rslearn.models.component import FeatureMaps
from rslearn.models.pooling_decoder import SegmentationPoolingDecoder
from rslearn.train.model_context import ModelContext, RasterImage


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
            torch.zeros((image_bands, image_size, image_size), dtype=torch.float32)
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
