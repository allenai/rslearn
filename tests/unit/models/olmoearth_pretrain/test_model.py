"""Test rslearn.models.olmoearth_pretrain."""

from datetime import datetime

import torch
from olmoearth_pretrain.train.masking import MaskValue

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


def test_batch_with_missing_modality_in_later_samples() -> None:
    """Test that model handles batches where some samples are missing a modality.

    This test catches the bug where the model only checks context.inputs[0] to
    determine which modalities are present, causing a KeyError when sample 0 has
    a modality but later samples don't.

    Scenario: sample 0 has sentinel1 + sentinel2, sample 1 has only sentinel2.
    Expected: Model should handle this gracefully (either skip the modality for
    samples that don't have it, or raise a clear error).
    """
    model = OlmoEarth(
        checkpoint_path="tests/unit/models/olmoearth_pretrain/",
        random_initialization=True,
        patch_size=4,
        embedding_size=128,
        use_legacy_timestamps=False,
    )

    T = 2
    H = 4
    W = 4
    inputs = [
        # Sample 0: has BOTH sentinel1 and sentinel2_l2a
        {
            "sentinel2_l2a": RasterImage(
                image=torch.zeros((12, T, H, W), dtype=torch.float32),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1))
                    for x in range(1, T + 1)
                ],
            ),
            "sentinel1": RasterImage(
                image=torch.zeros((2, T, H, W), dtype=torch.float32),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1))
                    for x in range(1, T + 1)
                ],
            ),
        },
        # Sample 1: has ONLY sentinel2_l2a (missing sentinel1)
        {
            "sentinel2_l2a": RasterImage(
                image=torch.zeros((12, T, H, W), dtype=torch.float32),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1))
                    for x in range(1, T + 1)
                ],
            ),
            # sentinel1 is missing!
        },
    ]
    context = ModelContext(inputs=inputs, metadatas=[])

    # Model should handle missing modalities gracefully - no KeyError should be raised.
    # The model detects modalities from ANY sample and masks missing data appropriately.
    feature_map = model(context)

    # Verify we get valid output for both samples
    assert len(feature_map.feature_maps) == 1
    features = feature_map.feature_maps[0]
    assert features.shape[0] == 2  # Both samples processed


def test_batch_with_missing_modality_in_first_sample() -> None:
    """Test that model uses modalities even when sample 0 doesn't have them.

    This test catches the bug where modalities present in later samples but not
    in sample 0 are completely ignored (wasted data).

    Scenario: sample 0 has only sentinel2, sample 1 has sentinel1 + sentinel2.
    Expected: Model should use sentinel1 from sample 1 (not ignore it).
    """
    model = OlmoEarth(
        checkpoint_path="tests/unit/models/olmoearth_pretrain/",
        random_initialization=True,
        patch_size=4,
        embedding_size=128,
        use_legacy_timestamps=False,
    )

    T = 2
    H = 4
    W = 4
    inputs = [
        # Sample 0: has ONLY sentinel2_l2a (missing sentinel1)
        {
            "sentinel2_l2a": RasterImage(
                image=torch.zeros((12, T, H, W), dtype=torch.float32),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1))
                    for x in range(1, T + 1)
                ],
            ),
            # sentinel1 is missing!
        },
        # Sample 1: has BOTH sentinel1 and sentinel2_l2a
        {
            "sentinel2_l2a": RasterImage(
                image=torch.zeros((12, T, H, W), dtype=torch.float32),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1))
                    for x in range(1, T + 1)
                ],
            ),
            "sentinel1": RasterImage(
                image=torch.zeros((2, T, H, W), dtype=torch.float32),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1))
                    for x in range(1, T + 1)
                ],
            ),
        },
    ]
    context = ModelContext(inputs=inputs, metadatas=[])

    # Check which modalities are detected - should check ALL samples, not just inputs[0]
    _, present_modalities, _ = model._prepare_modality_inputs(context)

    assert "sentinel1" in present_modalities
    assert "sentinel2_l2a" in present_modalities

    # Verify the model runs successfully
    feature_map = model(context)
    assert len(feature_map.feature_maps) == 1
    features = feature_map.feature_maps[0]
    assert features.shape[0] == 2  # Both samples processed


def test_temporal_alignment_with_expected_timestamps() -> None:
    """Test that timestamps are properly aligned to expected positions.

    When expected_timestamps is provided, missing timesteps should be inserted
    at their correct temporal positions (not just padded at the end).
    """
    model = OlmoEarth(
        checkpoint_path="tests/unit/models/olmoearth_pretrain/",
        random_initialization=True,
        patch_size=4,
        embedding_size=128,
        use_legacy_timestamps=False,
    )

    H = 4
    W = 4

    # Expected timestamps: Jan, Feb, Mar, Apr (4 months)
    expected_ts = [(datetime(2025, m, 1), datetime(2025, m, 28)) for m in range(1, 5)]

    # Sample 1: Has data for Jan, Mar (missing Feb, Apr)
    # The tensor values are non-zero so we can verify placement
    sample1_tensor = torch.ones((12, 2, H, W), dtype=torch.float32)
    sample1_tensor[:, 0, :, :] = 3.0  # March
    sample1_tensor[:, 1, :, :] = 1.0  # Jan
    sample1_timestamps = [
        (datetime(2025, 3, 1), datetime(2025, 3, 28)),  # Mar
        (datetime(2025, 1, 1), datetime(2025, 1, 28)),  # Jan
    ]

    # Sample 2: Has data for all 4 months
    sample2_tensor = torch.ones((12, 4, H, W), dtype=torch.float32)
    for i in range(4):
        sample2_tensor[:, i, :, :] = float(i + 1)
    sample2_timestamps = expected_ts.copy()

    inputs = [
        {
            "sentinel2_l2a": RasterImage(
                image=sample1_tensor,
                timestamps=sample1_timestamps,
                expected_timestamps=expected_ts,
            )
        },
        {
            "sentinel2_l2a": RasterImage(
                image=sample2_tensor,
                timestamps=sample2_timestamps,
                expected_timestamps=expected_ts,
            )
        },
    ]

    context = ModelContext(inputs=inputs, metadatas=[])
    sample, present_modalities, _ = model._prepare_modality_inputs(context)

    # Should have 4 timesteps (based on expected_timestamps)
    assert sample.sentinel2_l2a.shape == (2, H, W, 4, 12)

    # Sample 1: Check that Jan is at position 0, Mar is at position 2
    # Positions 1 (Feb) and 3 (Apr) should be zeros
    s1_data = sample.sentinel2_l2a[0]  # H, W, T, C
    assert torch.allclose(s1_data[0, 0, 0, :], torch.ones(12))  # Jan at pos 0
    assert torch.allclose(s1_data[0, 0, 1, :], torch.zeros(12))  # Feb missing
    assert torch.allclose(s1_data[0, 0, 2, :], torch.ones(12) * 3)  # Mar at pos 2
    assert torch.allclose(s1_data[0, 0, 3, :], torch.zeros(12))  # Apr missing

    # Sample 1 mask: positions 0 and 2 should be ONLINE_ENCODER, 1 and 3 should be MISSING
    s1_mask = sample.sentinel2_l2a_mask[0]  # H, W, T, S
    assert (s1_mask[:, :, 0, :] == MaskValue.ONLINE_ENCODER.value).all()  # Jan
    assert (s1_mask[:, :, 1, :] == MaskValue.MISSING.value).all()  # Feb
    assert (s1_mask[:, :, 2, :] == MaskValue.ONLINE_ENCODER.value).all()  # Mar
    assert (s1_mask[:, :, 3, :] == MaskValue.MISSING.value).all()  # Apr

    # Sample 2: All positions should have data and be marked ONLINE_ENCODER
    s2_mask = sample.sentinel2_l2a_mask[1]
    assert (s2_mask == MaskValue.ONLINE_ENCODER.value).all()


def test_missing_modality_handling() -> None:
    """Test that completely missing modalities are handled correctly.

    When a sample is missing a modality entirely, it should have an all-zero
    tensor with all timestamps marked as MISSING.
    """
    model = OlmoEarth(
        checkpoint_path="tests/unit/models/olmoearth_pretrain/",
        random_initialization=True,
        patch_size=4,
        embedding_size=128,
        use_legacy_timestamps=False,
    )

    H = 4
    W = 4
    T = 3

    timestamps = [
        (datetime(2025, m, 1), datetime(2025, m, 28)) for m in range(1, T + 1)
    ]

    # Sample 1: Has both sentinel2 and sentinel1
    inputs = [
        {
            "sentinel2_l2a": RasterImage(
                image=torch.ones((12, T, H, W), dtype=torch.float32),
                timestamps=timestamps,
            ),
            "sentinel1": RasterImage(
                image=torch.ones((2, T, H, W), dtype=torch.float32) * 2,
                timestamps=timestamps,
            ),
        },
        # Sample 2: Only has sentinel2, missing sentinel1 entirely
        {
            "sentinel2_l2a": RasterImage(
                image=torch.ones((12, T, H, W), dtype=torch.float32) * 3,
                timestamps=timestamps,
            ),
            # sentinel1 is completely missing
        },
    ]

    context = ModelContext(inputs=inputs, metadatas=[])
    sample, present_modalities, _ = model._prepare_modality_inputs(context)

    # Both modalities should be present (sentinel1 detected from sample 1)
    assert "sentinel2_l2a" in present_modalities
    assert "sentinel1" in present_modalities

    # Shapes should be consistent
    assert sample.sentinel2_l2a.shape == (2, H, W, T, 12)
    assert sample.sentinel1.shape == (2, H, W, T, 2)

    # Sample 1: sentinel1 should have data (value 2)
    s1_s1_data = sample.sentinel1[0]  # H, W, T, C
    assert torch.allclose(s1_s1_data, torch.ones(H, W, T, 2) * 2)
    s1_s1_mask = sample.sentinel1_mask[0]
    assert (s1_s1_mask == MaskValue.ONLINE_ENCODER.value).all()

    # Sample 2: sentinel1 should be all zeros with all MISSING mask
    s2_s1_data = sample.sentinel1[1]
    assert torch.allclose(s2_s1_data, torch.zeros(H, W, T, 2))
    s2_s1_mask = sample.sentinel1_mask[1]
    assert (s2_s1_mask == MaskValue.MISSING.value).all()


def test_chronological_sorting_in_alignment() -> None:
    """Test that data is sorted chronologically during alignment.

    Even if actual timestamps come in non-chronological order, the aligned
    output should be in chronological order.
    """
    model = OlmoEarth(
        checkpoint_path="tests/unit/models/olmoearth_pretrain/",
        random_initialization=True,
        patch_size=4,
        embedding_size=128,
        use_legacy_timestamps=False,
    )

    H = 4
    W = 4

    # Expected timestamps in chronological order
    expected_ts = [
        (datetime(2025, 1, 1), datetime(2025, 1, 28)),
        (datetime(2025, 2, 1), datetime(2025, 2, 28)),
        (datetime(2025, 3, 1), datetime(2025, 3, 28)),
    ]

    # Input data in NON-chronological order: Mar, Jan, Feb
    tensor = torch.ones((12, 3, H, W), dtype=torch.float32)
    tensor[:, 0, :, :] = 3.0  # This is March data
    tensor[:, 1, :, :] = 1.0  # This is January data
    tensor[:, 2, :, :] = 2.0  # This is February data

    # Timestamps match the non-chronological order
    actual_timestamps = [
        (datetime(2025, 3, 1), datetime(2025, 3, 28)),  # Mar
        (datetime(2025, 1, 1), datetime(2025, 1, 28)),  # Jan
        (datetime(2025, 2, 1), datetime(2025, 2, 28)),  # Feb
    ]

    inputs = [
        {
            "sentinel2_l2a": RasterImage(
                image=tensor,
                timestamps=actual_timestamps,
                expected_timestamps=expected_ts,
            )
        }
    ]

    context = ModelContext(inputs=inputs, metadatas=[])
    sample, _, _ = model._prepare_modality_inputs(context)

    # After alignment, data should be in chronological order: Jan(1), Feb(2), Mar(3)
    data = sample.sentinel2_l2a[0]  # H, W, T, C
    assert torch.allclose(data[0, 0, 0, :], torch.ones(12) * 1.0)  # Jan at pos 0
    assert torch.allclose(data[0, 0, 1, :], torch.ones(12) * 2.0)  # Feb at pos 1
    assert torch.allclose(data[0, 0, 2, :], torch.ones(12) * 3.0)  # Mar at pos 2
