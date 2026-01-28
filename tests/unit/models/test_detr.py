"""Tests for the rslearn.models.detr.detr module."""

import torch

from rslearn.models.component import FeatureMaps
from rslearn.models.detr.detr import Detr, DetrPredictor, SetCriterion
from rslearn.models.detr.transformer import Transformer
from rslearn.train.model_context import ModelContext, RasterImage


def test_detr_forward() -> None:
    """Test DETR Predictor forward pass."""
    batch_size = 1
    in_channels = 32
    num_classes = 3
    feature_h = 16
    feature_w = 16
    image_h = 64
    image_w = 64
    num_queries = 20

    # Create the FeatureMaps input.
    feature_maps = FeatureMaps(
        [
            torch.randn(
                (batch_size, in_channels, feature_h, feature_w), dtype=torch.float32
            ),
        ]
    )

    # Create input context with RasterImage, it is used to get original image size.
    inputs = [
        {
            "image": RasterImage(
                torch.zeros((3, 1, image_h, image_w), dtype=torch.float32)
            ),
        }
        for _ in range(batch_size)
    ]

    # Initialize DETR components
    transformer = Transformer(
        d_model=32,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
    )
    predictor = DetrPredictor(
        in_channels=in_channels,
        num_classes=num_classes,
        transformer=transformer,
        num_queries=num_queries,
    )
    criterion = SetCriterion(
        num_classes=num_classes,
    )
    detr = Detr(predictor=predictor, criterion=criterion)

    # Forward pass without targets (inference mode)
    detr.eval()
    context = ModelContext(inputs=inputs, metadatas=[])
    result = detr(feature_maps, context, targets=None)

    # Check output structure
    assert result.outputs is not None
    assert len(result.outputs) == batch_size
    for output in result.outputs:
        assert output["boxes"].shape[0] == num_queries
        assert output["labels"].shape[0] == num_queries
        assert output["scores"].shape[0] == num_queries

    # Forward pass with targets (training mode)
    detr.train()
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
    result_with_targets = detr(feature_maps, context, targets=targets)

    # Check loss dict exists
    assert result_with_targets.loss_dict is not None
    assert "loss_ce" in result_with_targets.loss_dict
