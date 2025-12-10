import torch

from rslearn.models.attention_pooling import AttentionPool
from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.train.model_context import ModelContext


def test_forward() -> None:
    """Test forward pass with randomly initialized OlmoEarth model and Attention Pooling."""
    model = OlmoEarth(
        checkpoint_path="tests/unit/models/olmoearth_pretrain/",
        # With random initialization we only need config.json, not the weights.
        random_initialization=True,
        patch_size=4,
        embedding_size=768,
        # we now expect an extra N dimension on the back of this.
        token_pooling=False,
    )
    pooling = AttentionPool(in_dim=768, num_heads=768 // 64)

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
    assert interim_features.shape == (1, 768, 1, 1, 3 * 2)

    feature_list = pooling(
        feature_maps, ModelContext(inputs=[], metadatas=[])
    ).feature_maps

    assert len(feature_list) == 1
    features = feature_list[0]
    # Feature shape should correspond to using patch_size=4.
    assert features.shape == (1, 768, 1, 1)

    # Backbone channels should match patch size and depth.
    assert model.get_backbone_channels() == [(4, 768)]
