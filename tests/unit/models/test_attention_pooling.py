import pytest
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


def test_attention_pool_spatial_preservation() -> None:
    """Test that AttentionPool preserves spatial structure correctly."""
    B, D, H, W, N = 1, MODEL_DIM, 2, 2, 6

    # Create input where each spatial position has uniform values across channels/tokens
    # but different values between positions: (0,0)->0, (0,1)->1, (1,0)->10, (1,1)->11
    feat_tokens = torch.zeros(B, D, H, W, N)
    for h in range(H):
        for w in range(W):
            feat_tokens[0, :, h, w, :] = h * 10 + w

    pooling = AttentionPool(in_dim=D, num_heads=NUM_HEADS)
    output = pooling(
        TokenFeatureMaps(feature_maps=[feat_tokens]),
        ModelContext(inputs=[], metadatas=[]),
    ).feature_maps[0]

    assert output.shape == (B, D, H, W)

    # Each spatial position should have distinct output values.
    vec_00 = output[0, :, 0, 0]
    vec_01 = output[0, :, 0, 1]
    vec_10 = output[0, :, 1, 0]
    vec_11 = output[0, :, 1, 1]

    assert not torch.allclose(vec_00, vec_01), "Positions (0,0) and (0,1) identical"
    assert not torch.allclose(vec_00, vec_10), "Positions (0,0) and (1,0) identical"
    assert not torch.allclose(vec_00, vec_11), "Positions (0,0) and (1,1) identical"
    assert not torch.allclose(vec_01, vec_10), "Positions (0,1) and (1,0) identical"
    assert not torch.allclose(vec_01, vec_11), "Positions (0,1) and (1,1) identical"
    assert not torch.allclose(vec_10, vec_11), "Positions (1,0) and (1,1) identical"


def test_simple_attention_pool_spatial_preservation() -> None:
    """Test that SimpleAttentionPool preserves spatial structure correctly."""
    B, D, H, W, N = 1, MODEL_DIM, 2, 2, 6

    feat_tokens = torch.zeros(B, D, H, W, N)
    for h in range(H):
        for w in range(W):
            feat_tokens[0, :, h, w, :] = h * 10 + w

    pooling = SimpleAttentionPool(in_dim=D)
    output = pooling(
        TokenFeatureMaps(feature_maps=[feat_tokens]),
        ModelContext(inputs=[], metadatas=[]),
    ).feature_maps[0]

    assert output.shape == (B, D, H, W)

    # Each spatial position should have distinct output values.
    vec_00 = output[0, :, 0, 0]
    vec_01 = output[0, :, 0, 1]
    vec_10 = output[0, :, 1, 0]
    vec_11 = output[0, :, 1, 1]

    assert not torch.allclose(vec_00, vec_01), "Positions (0,0) and (0,1) identical"
    assert not torch.allclose(vec_00, vec_10), "Positions (0,0) and (1,0) identical"
    assert not torch.allclose(vec_00, vec_11), "Positions (0,0) and (1,1) identical"
    assert not torch.allclose(vec_01, vec_10), "Positions (0,1) and (1,0) identical"
    assert not torch.allclose(vec_01, vec_11), "Positions (0,1) and (1,1) identical"
    assert not torch.allclose(vec_10, vec_11), "Positions (1,0) and (1,1) identical"


@pytest.mark.parametrize(
    "pooling",
    [
        AttentionPool(in_dim=MODEL_DIM, num_heads=NUM_HEADS),
        AttentionPool(in_dim=MODEL_DIM, num_heads=NUM_HEADS, linear_on_kv=False),
        SimpleAttentionPool(in_dim=MODEL_DIM),
        SimpleAttentionPool(in_dim=MODEL_DIM, hidden_linear=True),
    ],
)
def test_mask_matches_truncated_tokens(
    pooling: AttentionPool | SimpleAttentionPool,
) -> None:
    """Masking trailing tokens gives the same result as dropping them."""
    context = ModelContext(inputs=[], metadatas=[])
    B, D, H, W, N = 2, MODEL_DIM, 2, 2, 6
    num_valid = 4
    feat_tokens = torch.randn(B, D, H, W, N)
    mask = torch.ones(B, H, W, N, dtype=torch.bool)
    mask[..., num_valid:] = False

    masked = pooling(
        TokenFeatureMaps(feature_maps=[feat_tokens], masks=[mask]), context
    ).feature_maps[0]
    truncated = pooling(
        TokenFeatureMaps(feature_maps=[feat_tokens[..., :num_valid]]), context
    ).feature_maps[0]
    assert masked.shape == (B, D, H, W)
    assert torch.allclose(masked, truncated, atol=1e-5)

    # Perturbing the masked tokens does not change the output.
    feat_tokens2 = feat_tokens.clone()
    feat_tokens2[..., num_valid:] += 100.0
    masked2 = pooling(
        TokenFeatureMaps(feature_maps=[feat_tokens2], masks=[mask]), context
    ).feature_maps[0]
    assert torch.allclose(masked, masked2, atol=1e-5)


@pytest.mark.parametrize(
    "pooling",
    [
        AttentionPool(in_dim=MODEL_DIM, num_heads=NUM_HEADS),
        SimpleAttentionPool(in_dim=MODEL_DIM),
    ],
)
def test_mask_all_invalid_no_nan(
    pooling: AttentionPool | SimpleAttentionPool,
) -> None:
    """A location with no valid tokens still produces finite outputs."""
    feat_tokens = torch.randn(1, MODEL_DIM, 1, 2, 4)
    mask = torch.ones(1, 1, 2, 4, dtype=torch.bool)
    mask[:, 0, 1, :] = False
    out = pooling(
        TokenFeatureMaps(feature_maps=[feat_tokens], masks=[mask]),
        ModelContext(inputs=[], metadatas=[]),
    ).feature_maps[0]
    assert torch.isfinite(out).all()
