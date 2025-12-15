"""Test rslearn.models.olmoearth_pretrain."""

import pytest
import torch
from olmoearth_pretrain.train.masking import MaskValue

from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.train.model_context import ModelContext


def test_forward() -> None:
    """Test forward pass with randomly initialized model."""
    model = OlmoEarth(
        checkpoint_path="tests/unit/models/olmoearth_pretrain/",
        # With random initialization we only need config.json, not the weights.
        random_initialization=True,
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
    context = ModelContext(inputs=inputs, metadatas=[])
    feature_list = model(context)

    assert len(feature_list.feature_maps) == 1
    features = feature_list.feature_maps[0]
    # Feature shape should correspond to using patch_size=4.
    assert features.shape == (1, 768, 1, 1)

    # Backbone channels should match patch size and depth.
    assert model.get_backbone_channels() == [(4, 768)]


def test_forward_with_different_timesteps() -> None:
    """Test forward pass with different time steps per modality and per batch element"""
    model = OlmoEarth(
        checkpoint_path="tests/unit/models/olmoearth_pretrain/",
        # With random initialization we only need config.json, not the weights.
        random_initialization=True,
        patch_size=4,
        embedding_size=768,
    )

    max_timesteps = 8
    H = 4
    W = 4
    inputs = [
        {
            # 12 channels per timestep.
            "sentinel2_l2a": torch.zeros(
                (max_timesteps * 12, H, W), dtype=torch.float32
            ),
            # 2 channels per timestep.
            "sentinel1": torch.zeros((5 * 2, H, W), dtype=torch.float32),
        },
        {
            # 12 channels per timestep.
            "sentinel2_l2a": torch.zeros((7 * 12, H, W), dtype=torch.float32),
            # 2 channels per timestep.
            "sentinel1": torch.zeros((4 * 2, H, W), dtype=torch.float32),
        },
    ]
    context = ModelContext(inputs=inputs, metadatas=[])
    sample, present_modalities, _ = model._prepare_modality_inputs(context)
    # tensors must follow shape: [b h w t c]
    assert present_modalities == ["sentinel2_l2a", "sentinel1"]
    assert sample.sentinel2_l2a.shape == (2, H, W, max_timesteps, 12)
    assert sample.sentinel1.shape == (2, H, W, max_timesteps, 2)

    assert (sample.sentinel2_l2a_mask[0] == MaskValue.ONLINE_ENCODER.value).all()
    assert (
        sample.sentinel2_l2a_mask[1, :, :, 0:7, :] == MaskValue.ONLINE_ENCODER.value
    ).all()
    assert (sample.sentinel2_l2a_mask[1, :, :, 7:, :] == MaskValue.MISSING.value).all()
    assert (
        sample.sentinel1_mask[0, :, :, 0:5, :] == MaskValue.ONLINE_ENCODER.value
    ).all()
    assert (sample.sentinel1_mask[0, :, :, 5:, :] == MaskValue.MISSING.value).all()
    assert (
        sample.sentinel1_mask[1, :, :, 0:4, :] == MaskValue.ONLINE_ENCODER.value
    ).all()
    assert (sample.sentinel1_mask[1, :, :, 4:, :] == MaskValue.MISSING.value).all()

    assert sample.timestamps.shape == (2, max_timesteps, 3)
    assert (sample.timestamps[:, :, 1] == torch.arange(max_timesteps)).all()

    feature_list = model(context)

    assert len(feature_list.feature_maps) == 1
    features = feature_list.feature_maps[0]
    assert features.shape == (2, 768, 1, 1)


def test_error_if_no_checkpoint() -> None:
    """Should raise error if there is no distributed checkpoint."""
    with pytest.raises(FileNotFoundError):
        OlmoEarth(
            checkpoint_path="tests/unit/models/olmoearth_pretrain/",
            patch_size=4,
            embedding_size=768,
        )
