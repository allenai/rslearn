import torch

from rslearn.models.attention_pooling import AttentionPool, SimpleAttentionPool
from rslearn.models.component import TokenFeatureMaps
from rslearn.train.model_context import ModelContext

MODEL_DIM = 16
NUM_HEADS = 2


def test_forward() -> None:
    """Test forward pass with FeatureMap and Attention Pooling."""
    pooling = AttentionPool(in_dim=MODEL_DIM, num_heads=NUM_HEADS)

    # see tests/unit/models/olmoearth_pretrain/test_model.py
    # for how we obtained this shape for the feature map
    feature_list = pooling(
        TokenFeatureMaps(feature_maps=[torch.ones(1, MODEL_DIM, 1, 1, 6)]),
        ModelContext(inputs=[], metadatas=[]),
    ).feature_maps

    assert len(feature_list) == 1
    features = feature_list[0]
    # Feature shape should correspond to using patch_size=4.
    assert features.shape == (1, MODEL_DIM, 1, 1)


def test_forward_no_linear() -> None:
    """Test forward pass with FeatureMap and Attention Pooling."""
    pooling = AttentionPool(in_dim=MODEL_DIM, num_heads=NUM_HEADS, linear_on_kv=False)

    # see tests/unit/models/olmoearth_pretrain/test_model.py
    # for how we obtained this shape for the feature map
    feature_list = pooling(
        TokenFeatureMaps(feature_maps=[torch.ones(1, MODEL_DIM, 1, 1, 6)]),
        ModelContext(inputs=[], metadatas=[]),
    ).feature_maps

    assert len(feature_list) == 1
    features = feature_list[0]
    # Feature shape should correspond to using patch_size=4.
    assert features.shape == (1, MODEL_DIM, 1, 1)


def test_forward_simple() -> None:
    """Test forward pass with FeatureMap and Attention Pooling."""
    pooling = SimpleAttentionPool(in_dim=MODEL_DIM)

    # see tests/unit/models/olmoearth_pretrain/test_model.py
    # for how we obtained this shape for the feature map
    feature_list = pooling(
        TokenFeatureMaps(feature_maps=[torch.ones(1, MODEL_DIM, 1, 1, 6)]),
        ModelContext(inputs=[], metadatas=[]),
    ).feature_maps

    assert len(feature_list) == 1
    features = feature_list[0]
    # Feature shape should correspond to using patch_size=4.
    assert features.shape == (1, MODEL_DIM, 1, 1)


def test_forward_simple_with_linear() -> None:
    """Test forward pass with FeatureMap and Attention Pooling."""
    pooling = SimpleAttentionPool(in_dim=MODEL_DIM, hidden_linear=True)

    # see tests/unit/models/olmoearth_pretrain/test_model.py
    # for how we obtained this shape for the feature map
    feature_list = pooling(
        TokenFeatureMaps(feature_maps=[torch.ones(1, MODEL_DIM, 1, 1, 6)]),
        ModelContext(inputs=[], metadatas=[]),
    ).feature_maps

    assert len(feature_list) == 1
    features = feature_list[0]
    # Feature shape should correspond to using patch_size=4.
    assert features.shape == (1, MODEL_DIM, 1, 1)
