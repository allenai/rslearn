"""Tests for the rslearn.models.global_pool module."""

import torch

from rslearn.models.component import FeatureMaps, FeatureVector
from rslearn.models.global_pool import GlobalPool
from rslearn.train.model_context import ModelContext


def test_global_pool_feature_vector() -> None:
    """Test GlobalPool returning FeatureVector (default)."""
    batch_size = 2
    channels = 16
    height = 8
    width = 8

    feature_maps = FeatureMaps(
        [
            torch.randn((batch_size, channels, height, width), dtype=torch.float32),
        ]
    )

    pool = GlobalPool(mode="mean", keep_spatial_dims=False)
    result = pool(feature_maps, ModelContext(inputs=[], metadatas=[]))

    assert isinstance(result, FeatureVector)
    assert result.feature_vector.shape == (batch_size, channels)


def test_global_pool_feature_maps() -> None:
    """Test GlobalPool returning FeatureMaps (keep_spatial_dims=True)."""
    batch_size = 2
    channels = 16
    height = 8
    width = 8

    feature_maps = FeatureMaps(
        [
            torch.randn((batch_size, channels, height, width), dtype=torch.float32),
        ]
    )

    pool = GlobalPool(mode="mean", keep_spatial_dims=True)
    result = pool(feature_maps, ModelContext(inputs=[], metadatas=[]))

    assert isinstance(result, FeatureMaps)
    assert len(result.feature_maps) == 1
    assert result.feature_maps[0].shape == (batch_size, channels, 1, 1)
