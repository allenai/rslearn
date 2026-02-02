"""Tests for the rslearn.models.pick_features module."""

import torch

from rslearn.models.component import FeatureMaps
from rslearn.models.pick_features import PickFeatures
from rslearn.train.model_context import ModelContext


def test_pick_features_single() -> None:
    """Test PickFeatures selecting a single feature map."""
    batch_size = 2
    feature_maps = FeatureMaps(
        [
            torch.randn((batch_size, 8, 16, 16), dtype=torch.float32),
            torch.randn((batch_size, 16, 8, 8), dtype=torch.float32),
            torch.randn((batch_size, 32, 4, 4), dtype=torch.float32),
        ]
    )

    pick_features = PickFeatures(indexes=[1])
    result = pick_features(feature_maps, ModelContext(inputs=[], metadatas=[]))
    assert isinstance(result, FeatureMaps)
    assert len(result.feature_maps) == 1
    assert result.feature_maps[0].shape == (batch_size, 16, 8, 8)
