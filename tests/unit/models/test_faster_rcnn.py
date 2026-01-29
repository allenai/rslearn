"""Tests for the rslearn.models.faster_rcnn module."""

import torch

from rslearn.models.component import FeatureMaps
from rslearn.models.faster_rcnn import FasterRCNN
from rslearn.train.model_context import ModelContext, RasterImage


def test_faster_rcnn_forward() -> None:
    """Test FasterRCNN Predictor forward pass."""
    batch_size = 1
    num_channels = 32
    num_classes = 3
    feature_h = 16
    feature_w = 16
    image_h = 64
    image_w = 64

    # Create the FeatureMaps input with two resolutions.
    feature_maps = FeatureMaps(
        [
            torch.randn(
                (batch_size, num_channels, feature_h, feature_w), dtype=torch.float32
            ),
            torch.randn(
                (batch_size, num_channels, feature_h // 2, feature_w // 2),
                dtype=torch.float32,
            ),
        ]
    )

    # Create input context with RasterImage, it is used to get original image size
    inputs = [
        {
            "image": RasterImage(
                torch.zeros((3, 1, image_h, image_w), dtype=torch.float32)
            ),
        }
        for _ in range(batch_size)
    ]

    # Initialize FasterRCNN
    faster_rcnn = FasterRCNN(
        # The first feature map is 1/4 and second is 1/8.
        downsample_factors=[4, 8],
        num_channels=num_channels,
        num_classes=num_classes,
        anchor_sizes=[[32], [64]],
    )

    # Forward pass without targets (inference mode)
    faster_rcnn.eval()
    context = ModelContext(inputs=inputs, metadatas=[])
    result = faster_rcnn(feature_maps, context, targets=None)

    # Check output structure
    assert result.outputs is not None
    assert len(result.outputs) == batch_size
    for output in result.outputs:
        assert "boxes" in output
        assert "labels" in output
        assert "scores" in output

    # Forward pass with targets (training mode)
    faster_rcnn.train()
    targets = [
        {
            "boxes": torch.tensor(
                [[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0]],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([1, 2], dtype=torch.int64),
        }
        for _ in range(batch_size)
    ]
    result_with_targets = faster_rcnn(feature_maps, context, targets=targets)

    # Check loss dict exists
    assert result_with_targets.loss_dict is not None
    assert "loss_box_reg" in result_with_targets.loss_dict
