"""Test rslearn.models.olmoearth_pretrain."""

import torch

from rslearn.models.olmoearth_pretrain.model import OlmoEarth


def test_forward() -> None:
    """Test forward pass with randomly initialized model."""
    model = OlmoEarth(
        checkpoint_path="tests/unit/models/olmoearth_pretrain/",
        # With random initialization we only need config.json, not the weights.
        random_initialization=True,
        selector=["encoder"],
        patch_size=4,
        embedding_size=768,
    )

    T = 2
    H = 4
    W = 4
    inputs = [
        {
            # 12 channels per timestep.
            "sentinel2_l2a": torch.zeros((T * 12, H, W), dtype=torch.float32),
        }
    ]
    feature_list = model(inputs)

    assert len(feature_list) == 1
    features = feature_list[0]
    # Feature shape should correspond to using patch_size=4.
    assert features.shape == (1, 768, 1, 1)

    # Backbone channels should match patch size and depth.
    assert model.get_backbone_channels() == [(4, 768)]
