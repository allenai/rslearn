"""Test rslearn.models.olmoearth_pretrain."""

import pytest
import torch

from rslearn.models.attention_pooling import AttentionPool, SimpleAttentionPool
from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.train.model_context import ModelContext


def test_forward() -> None:
    """Test forward pass with randomly initialized model."""
    model = OlmoEarth(
        checkpoint_path="tests/unit/models/olmoearth_pretrain/",
        # With random initialization we only need config.json, not the weights.
        random_initialization=True,
        patch_size=4,
        embedding_size=128,
    )

    T = 2
    H = 4
    W = 4
    inputs = [
        {
            # 12 channels per timestep.
            "sentinel2_l2a": torch.zeros(
                (T * 12, H, W), dtype=torch.float32, device=torch.device("cpu")
            ),
        }
    ]
    feature_map = model(ModelContext(inputs=inputs, metadatas=[]))

    assert len(feature_map.feature_maps) == 1
    features = feature_map.feature_maps[0]
    # Feature shape should correspond to using patch_size=4.
    assert features.shape == (1, 128, 1, 1)

    # Backbone channels should match patch size and depth.
    assert model.get_backbone_channels() == [(4, 128)]


def test_forward_no_pooling() -> None:
    """Test forward pass with randomly initialized model."""
    model = OlmoEarth(
        checkpoint_path="tests/unit/models/olmoearth_pretrain/",
        # With random initialization we only need config.json, not the weights.
        random_initialization=True,
        patch_size=4,
        embedding_size=128,
        token_pooling=False,
    )

    T = 2
    H = 4
    W = 4
    inputs = [
        {
            # 12 channels per timestep.
            "sentinel2_l2a": torch.zeros(
                (T * 12, H, W), dtype=torch.float32, device=torch.device("cpu")
            ),
        }
    ]
    feature_map = model(ModelContext(inputs=inputs, metadatas=[]))

    assert len(feature_map.feature_maps) == 1
    features = feature_map.feature_maps[0]
    # Feature shape should correspond to using patch_size=4.
    # 6 = 3 band sets * 2 timesteps
    assert features.shape == (1, 128, 1, 1, 6)

    # Backbone channels should match patch size and depth.
    assert model.get_backbone_channels() == [(4, 128)]


def test_error_if_no_checkpoint() -> None:
    """Should raise error if there is no distributed checkpoint."""
    with pytest.raises(FileNotFoundError):
        OlmoEarth(
            checkpoint_path="tests/unit/models/olmoearth_pretrain/",
            patch_size=4,
            embedding_size=128,
        )


def test_with_attnpool() -> None:
    """Test forward pass with randomly initialized OlmoEarth model and Attention Pooling."""
    model = OlmoEarth(
        checkpoint_path="tests/unit/models/olmoearth_pretrain/",
        # With random initialization we only need config.json, not the weights.
        random_initialization=True,
        patch_size=4,
        embedding_size=128,
        # we now expect an extra N dimension on the back of this.
        token_pooling=False,
    )
    pooling = AttentionPool(in_dim=128, num_heads=2)

    T = 2
    H = 4
    W = 4
    inputs = [
        {
            # 12 channels per timestep.
            "sentinel2_l2a": torch.zeros((T * 12, H, W), dtype=torch.float32),
        }
    ]
    feature_maps = model(ModelContext(inputs=inputs, metadatas=[]))

    # check we have an N dimension
    interim_feature_list = feature_maps.feature_maps
    assert len(interim_feature_list) == 1
    interim_features = interim_feature_list[0]
    # Feature shape should correspond to using patch_size=4.
    # 3 band sets in s2, 2 timesteps
    assert interim_features.shape == (1, 128, 1, 1, 3 * 2)

    feature_list = pooling(
        feature_maps, ModelContext(inputs=[], metadatas=[])
    ).feature_maps

    assert len(feature_list) == 1
    features = feature_list[0]
    # Feature shape should correspond to using patch_size=4.
    assert features.shape == (1, 128, 1, 1)

    # Backbone channels should match patch size and depth.
    assert model.get_backbone_channels() == [(4, 128)]


def test_with_simple_attnpool() -> None:
    """Test forward pass with randomly initialized OlmoEarth model and Attention Pooling."""
    model = OlmoEarth(
        checkpoint_path="tests/unit/models/olmoearth_pretrain/",
        # With random initialization we only need config.json, not the weights.
        random_initialization=True,
        patch_size=4,
        embedding_size=128,
        # we now expect an extra N dimension on the back of this.
        token_pooling=False,
    )
    pooling = SimpleAttentionPool(in_dim=128)

    T = 2
    H = 4
    W = 4
    inputs = [
        {
            # 12 channels per timestep.
            "sentinel2_l2a": torch.zeros((T * 12, H, W), dtype=torch.float32),
        }
    ]
    feature_maps = model(ModelContext(inputs=inputs, metadatas=[]))

    # check we have an N dimension
    interim_feature_list = feature_maps.feature_maps
    assert len(interim_feature_list) == 1
    interim_features = interim_feature_list[0]
    # Feature shape should correspond to using patch_size=4.
    # 3 band sets in s2, 2 timesteps
    assert interim_features.shape == (1, 128, 1, 1, 3 * 2)

    feature_list = pooling(
        feature_maps, ModelContext(inputs=[], metadatas=[])
    ).feature_maps

    assert len(feature_list) == 1
    features = feature_list[0]
    # Feature shape should correspond to using patch_size=4.
    assert features.shape == (1, 128, 1, 1)

    # Backbone channels should match patch size and depth.
    assert model.get_backbone_channels() == [(4, 128)]
