"""Test rslearn.models.olmoearth_pretrain."""

from datetime import datetime

import pytest
import torch
from olmoearth_pretrain.datatypes import MaskValue

from rslearn.models.attention_pooling import AttentionPool, SimpleAttentionPool
from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.train.model_context import ModelContext, RasterImage


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
            "sentinel2_l2a": RasterImage(
                image=torch.zeros(
                    (12, T, H, W), dtype=torch.float32, device=torch.device("cpu")
                ),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1))
                    for x in range(1, T + 1)
                ],
            )
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
            "sentinel2_l2a": RasterImage(
                image=torch.zeros(
                    (12, T, H, W), dtype=torch.float32, device=torch.device("cpu")
                ),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1))
                    for x in range(1, T + 1)
                ],
            )
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
            random_initialization=True,
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
            "sentinel2_l2a": RasterImage(
                image=torch.zeros(
                    (12, T, H, W), dtype=torch.float32, device=torch.device("cpu")
                ),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1))
                    for x in range(1, T + 1)
                ],
            )
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
            "sentinel2_l2a": RasterImage(
                image=torch.zeros(
                    (12, T, H, W), dtype=torch.float32, device=torch.device("cpu")
                ),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1))
                    for x in range(1, T + 1)
                ],
            )
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


def test_forward_with_different_timesteps() -> None:
    """Test forward pass with different time steps per modality and per batch element"""
    model = OlmoEarth(
        checkpoint_path="tests/unit/models/olmoearth_pretrain/",
        # With random initialization we only need config.json, not the weights.
        random_initialization=True,
        patch_size=4,
        embedding_size=128,
        # Use non-legacy timestamps to properly test variable-length padding behavior
        use_legacy_timestamps=False,
    )

    max_timesteps = 8
    H = 4
    W = 4
    inputs = [
        {
            # 12 channels per timestep.
            "sentinel2_l2a": RasterImage(
                image=torch.zeros((12, max_timesteps, H, W), dtype=torch.float32),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1))
                    for x in range(1, max_timesteps + 1)
                ],
            ),
            # 2 channels per timestep.
            "sentinel1": RasterImage(
                image=torch.zeros((2, 5, H, W), dtype=torch.float32),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1))
                    for x in range(1, 5 + 1)
                ],
            ),
        },
        {
            # 12 channels per timestep.
            "sentinel2_l2a": RasterImage(
                image=torch.zeros((12, 7, H, W), dtype=torch.float32),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1))
                    for x in range(1, 7 + 1)
                ],
            ),
            # 2 channels per timestep.
            "sentinel1": RasterImage(
                image=torch.zeros((2, 4, H, W), dtype=torch.float32),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1))
                    for x in range(1, 4 + 1)
                ],
            ),
        },
    ]
    context = ModelContext(inputs=inputs, metadatas=[])
    sample, present_modalities, _ = model._prepare_modality_inputs(context)

    def assert_modality_shapes() -> None:
        """Verify modality tensors have correct shape [batch, H, W, time, channels]."""
        assert present_modalities == ["sentinel2_l2a", "sentinel1"]
        assert sample.sentinel2_l2a.shape == (2, H, W, max_timesteps, 12)
        assert sample.sentinel1.shape == (2, H, W, max_timesteps, 2)

    def assert_mask_valid_timesteps(
        mask: torch.Tensor, batch_idx: int, valid_timesteps: int
    ) -> None:
        """Verify mask marks valid timesteps as ONLINE_ENCODER and padding as MISSING."""
        assert (
            mask[batch_idx, :, :, :valid_timesteps, :] == MaskValue.ONLINE_ENCODER.value
        ).all(), f"Expected valid timesteps 0:{valid_timesteps} to be ONLINE_ENCODER"
        if valid_timesteps < max_timesteps:
            assert (
                mask[batch_idx, :, :, valid_timesteps:, :] == MaskValue.MISSING.value
            ).all(), f"Expected padded timesteps {valid_timesteps}: to be MISSING"

    def assert_sentinel2_masks() -> None:
        """Verify Sentinel-2 masks: batch 0 has 8 valid, batch 1 has 7 valid."""
        assert_mask_valid_timesteps(
            sample.sentinel2_l2a_mask, batch_idx=0, valid_timesteps=8
        )
        assert_mask_valid_timesteps(
            sample.sentinel2_l2a_mask, batch_idx=1, valid_timesteps=7
        )

    def assert_sentinel1_masks() -> None:
        """Verify Sentinel-1 masks: batch 0 has 5 valid, batch 1 has 4 valid."""
        assert_mask_valid_timesteps(
            sample.sentinel1_mask, batch_idx=0, valid_timesteps=5
        )
        assert_mask_valid_timesteps(
            sample.sentinel1_mask, batch_idx=1, valid_timesteps=4
        )

    def assert_timestamps() -> None:
        """Verify timestamp tensor shape and values."""
        assert sample.timestamps.shape == (2, max_timesteps, 3)
        # First batch element: all 8 timestamps populated
        assert (sample.timestamps[0, :, 1] == torch.arange(max_timesteps)).all()
        # Second batch element: only 7 timestamps (max across its modalities)
        assert (sample.timestamps[1, :-1, 1] == torch.arange(max_timesteps)[:7]).all()
        padded_timestep = sample.timestamps[1, -1, :]
        is_padded_zero = (padded_timestep == 0).all()
        assert is_padded_zero, (
            f"Padded timestamp for sample 1 at last timestep is not all zeros: {padded_timestep.tolist()}"
        )

    def assert_output_features() -> None:
        """Verify model output has expected shape."""
        feature_list = model(context)
        assert len(feature_list.feature_maps) == 1
        features = feature_list.feature_maps[0]
        assert features.shape == (2, 128, 1, 1)

    assert_modality_shapes()
    assert_sentinel2_masks()
    assert_sentinel1_masks()
    assert_timestamps()
    assert_output_features()
