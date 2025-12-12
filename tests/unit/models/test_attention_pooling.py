import torch

from rslearn.models.attention_pooling import AttentionPool
from rslearn.models.component import FeatureMaps
from rslearn.train.model_context import ModelContext


def test_forward() -> None:
    """Test forward pass with FeatureMap and Attention Pooling."""
    pooling = AttentionPool(in_dim=768, num_heads=768 // 64)

    # see tests/unit/models/olmoearth_pretrain/test_model.py
    # for how we obtained this shape for the feature map
    feature_list = pooling(
        FeatureMaps(feature_maps=[torch.ones(1, 768, 1, 1, 6)]),
        ModelContext(inputs=[], metadatas=[]),
    ).feature_maps

    assert len(feature_list) == 1
    features = feature_list[0]
    # Feature shape should correspond to using patch_size=4.
    assert features.shape == (1, 768, 1, 1)


def test_forward_no_linear() -> None:
    """Test forward pass with FeatureMap and Attention Pooling."""
    pooling = AttentionPool(in_dim=768, num_heads=768 // 64, linear_on_kv=False)

    # see tests/unit/models/olmoearth_pretrain/test_model.py
    # for how we obtained this shape for the feature map
    feature_list = pooling(
        FeatureMaps(feature_maps=[torch.ones(1, 768, 1, 1, 6)]),
        ModelContext(inputs=[], metadatas=[]),
    ).feature_maps

    assert len(feature_list) == 1
    features = feature_list[0]
    # Feature shape should correspond to using patch_size=4.
    assert features.shape == (1, 768, 1, 1)
