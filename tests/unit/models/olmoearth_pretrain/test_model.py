"""Test rslearn.models.olmoearth_pretrain."""

from datetime import datetime, timedelta

import torch
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskValue

from rslearn.const import WGS84_PROJECTION
from rslearn.models.attention_pooling import AttentionPool, SimpleAttentionPool
from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.train.model_context import ModelContext, RasterImage, SampleMetadata


def _make_metadata(crop_bounds: tuple[int, int, int, int]) -> SampleMetadata:
    return SampleMetadata(
        window_group="test",
        window_name="test",
        window_bounds=crop_bounds,
        crop_bounds=crop_bounds,
        crop_idx=0,
        num_crops_in_window=1,
        time_range=None,
        projection=WGS84_PROJECTION,
        dataset_source=None,
    )


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
    feature_map = model(
        ModelContext(inputs=inputs, metadatas=[_make_metadata((0, 0, H, W))])
    )

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
    feature_map = model(
        ModelContext(inputs=inputs, metadatas=[_make_metadata((0, 0, H, W))])
    )

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
    feature_maps = model(
        ModelContext(inputs=inputs, metadatas=[_make_metadata((0, 0, H, W))])
    )

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
    feature_maps = model(
        ModelContext(inputs=inputs, metadatas=[_make_metadata((0, 0, H, W))])
    )

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
    context = ModelContext(
        inputs=inputs,
        metadatas=[_make_metadata((0, 0, H, W)), _make_metadata((0, 0, H, W))],
    )
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
            mask[batch_idx, :, :, :valid_timesteps] == MaskValue.ONLINE_ENCODER.value
        ).all(), f"Expected valid timesteps 0:{valid_timesteps} to be ONLINE_ENCODER"
        if valid_timesteps < max_timesteps:
            assert (
                mask[batch_idx, :, :, valid_timesteps:] == MaskValue.MISSING.value
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
    context = ModelContext(
        inputs=inputs,
        metadatas=[_make_metadata((0, 0, H, W)), _make_metadata((0, 0, H, W))],
    )

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
    context = ModelContext(
        inputs=inputs,
        metadatas=[_make_metadata((0, 0, H, W)), _make_metadata((0, 0, H, W))],
    )

    # Check which modalities are detected - should check ALL samples, not just inputs[0]
    _, present_modalities, _ = model._prepare_modality_inputs(context)

    assert "sentinel1" in present_modalities
    assert "sentinel2_l2a" in present_modalities

    # Verify the model runs successfully
    feature_map = model(context)
    assert len(feature_map.feature_maps) == 1
    features = feature_map.feature_maps[0]
    assert features.shape[0] == 2  # Both samples processed


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

    context = ModelContext(
        inputs=inputs,
        metadatas=[_make_metadata((0, 0, H, W)), _make_metadata((0, 0, H, W))],
    )
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


def test_legacy_timestamps_one_modality_three_timesteps() -> None:
    """Legacy timestamps with one modality should produce [1 Jan 2024, 1 Feb 2024, 1 Mar 2024]."""
    model = OlmoEarth(
        checkpoint_path="tests/unit/models/olmoearth_pretrain/",
        random_initialization=True,
        patch_size=4,
        embedding_size=128,
        use_legacy_timestamps=True,
    )

    H = 4
    W = 4
    T = 3
    inputs = [
        {
            "sentinel2_l2a": RasterImage(
                image=torch.zeros((12, T, H, W), dtype=torch.float32),
                timestamps=[
                    (datetime(2025, 6, 15), datetime(2025, 6, 15)),
                    (datetime(2025, 9, 1), datetime(2025, 9, 1)),
                    (datetime(2025, 12, 25), datetime(2025, 12, 25)),
                ],
            ),
        }
    ]
    context = ModelContext(inputs=inputs, metadatas=[_make_metadata((0, 0, H, W))])
    sample, present_modalities, _ = model._prepare_modality_inputs_legacy(context)

    assert present_modalities == ["sentinel2_l2a"]
    assert sample.timestamps.shape == (1, T, 3)

    # Legacy timestamps: day=1, month=0-indexed, year=2024
    expected = torch.tensor(
        [
            [1, 0, 2024],  # 1 January 2024
            [1, 1, 2024],  # 1 February 2024
            [1, 2, 2024],  # 1 March 2024
        ],
        dtype=torch.int32,
    )
    assert (sample.timestamps[0] == expected).all()


def test_legacy_timestamps_two_modalities_different_timesteps() -> None:
    """Legacy timestamps with two modalities: max_timesteps from the larger, shorter one padded."""
    model = OlmoEarth(
        checkpoint_path="tests/unit/models/olmoearth_pretrain/",
        random_initialization=True,
        patch_size=4,
        embedding_size=128,
        use_legacy_timestamps=True,
    )

    H = 4
    W = 4
    inputs = [
        {
            "sentinel2_l2a": RasterImage(
                image=torch.ones((12, 3, H, W), dtype=torch.float32),
                timestamps=[
                    (datetime(2025, m, 1), datetime(2025, m, 1)) for m in range(1, 4)
                ],
            ),
            "sentinel1": RasterImage(
                image=torch.ones((2, 2, H, W), dtype=torch.float32) * 2,
                timestamps=[
                    (datetime(2025, m, 1), datetime(2025, m, 1)) for m in range(1, 3)
                ],
            ),
        }
    ]
    context = ModelContext(inputs=inputs, metadatas=[_make_metadata((0, 0, H, W))])
    sample, present_modalities, _ = model._prepare_modality_inputs_legacy(context)

    assert "sentinel2_l2a" in present_modalities
    assert "sentinel1" in present_modalities

    # max_timesteps = 3 (from sentinel2_l2a)
    expected_ts = torch.tensor(
        [
            [1, 0, 2024],
            [1, 1, 2024],
            [1, 2, 2024],
        ],
        dtype=torch.int32,
    )
    assert sample.timestamps.shape == (1, 3, 3)
    assert (sample.timestamps[0] == expected_ts).all()

    # sentinel2_l2a: all 3 timesteps valid
    s2_mask = sample.sentinel2_l2a_mask[0]  # H, W, T, S
    assert (s2_mask == MaskValue.ONLINE_ENCODER.value).all()

    # sentinel1: first 2 timesteps valid, third is MISSING
    s1_mask = sample.sentinel1_mask[0]  # H, W, T, S
    assert (s1_mask[:, :, :2] == MaskValue.ONLINE_ENCODER.value).all()
    assert (s1_mask[:, :, 2] == MaskValue.MISSING.value).all()


def test_normal_timestamps_one_modality() -> None:
    """Normal timestamps with one modality should exactly match the input timestamps."""
    model = OlmoEarth(
        checkpoint_path="tests/unit/models/olmoearth_pretrain/",
        random_initialization=True,
        patch_size=4,
        embedding_size=128,
        use_legacy_timestamps=False,
    )

    H = 4
    W = 4
    inputs = [
        {
            "sentinel2_l2a": RasterImage(
                image=torch.zeros((12, 3, H, W), dtype=torch.float32),
                timestamps=[
                    (datetime(2025, 1, 5), datetime(2025, 1, 5)),
                    (datetime(2025, 3, 15), datetime(2025, 3, 15)),
                    (datetime(2025, 7, 20), datetime(2025, 7, 20)),
                ],
            ),
        }
    ]
    context = ModelContext(inputs=inputs, metadatas=[_make_metadata((0, 0, H, W))])
    sample, _, _ = model._prepare_modality_inputs(context)

    assert sample.timestamps.shape == (1, 3, 3)
    expected = torch.tensor(
        [
            [5, 0, 2025],  # Jan 5 (month 0-indexed)
            [15, 2, 2025],  # Mar 15
            [20, 6, 2025],  # Jul 20
        ],
        dtype=torch.int32,
    )
    assert (sample.timestamps[0] == expected).all()

    # All timesteps should be valid for sentinel2_l2a
    s2_mask = sample.sentinel2_l2a_mask[0]  # H, W, T, S
    assert (s2_mask == MaskValue.ONLINE_ENCODER.value).all()


def test_normal_timestamps_two_modalities_15d_tolerance() -> None:
    """Greedy timestamp alignment with 15-day tolerance merges close timestamps across modalities."""
    model = OlmoEarth(
        checkpoint_path="tests/unit/models/olmoearth_pretrain/",
        random_initialization=True,
        patch_size=4,
        embedding_size=128,
        use_legacy_timestamps=False,
        timestamp_error_tolerance=timedelta(days=15),
    )

    H = 4
    W = 4
    # sentinel2_l2a is first in MODALITY_NAMES, sentinel1 is second.
    s2_dates = [
        datetime(2025, 1, 1),
        datetime(2025, 2, 1),
        datetime(2025, 5, 1),
        datetime(2025, 6, 1),
    ]
    s1_dates = [
        datetime(2025, 1, 10),
        datetime(2025, 2, 2),
        datetime(2025, 2, 3),
        datetime(2025, 3, 1),
        datetime(2025, 4, 10),
        datetime(2025, 4, 25),
    ]

    s2_tensor = torch.zeros((12, len(s2_dates), H, W), dtype=torch.float32)
    for i in range(len(s2_dates)):
        s2_tensor[:, i, :, :] = float(i + 1)

    s1_tensor = torch.zeros((2, len(s1_dates), H, W), dtype=torch.float32)
    for i in range(len(s1_dates)):
        s1_tensor[:, i, :, :] = float(i + 10)

    inputs = [
        {
            "sentinel2_l2a": RasterImage(
                image=s2_tensor,
                timestamps=[(d, d) for d in s2_dates],
            ),
            "sentinel1": RasterImage(
                image=s1_tensor,
                timestamps=[(d, d) for d in s1_dates],
            ),
        }
    ]
    context = ModelContext(inputs=inputs, metadatas=[_make_metadata((0, 0, H, W))])
    sample, present_modalities, _ = model._prepare_modality_inputs(context)

    # Expected merged timestamps (sorted):
    # Jan 1, Feb 1, Feb 3, Mar 1, Apr 10, May 1, Jun 1
    # - Jan 10 -> Jan 1 (9d < 15d)
    # - Feb 2 -> Feb 1 (1d < 15d)
    # - Feb 3 -> new (Feb 1 already used by Feb 2 in sentinel1)
    # - Mar 1 -> new
    # - Apr 10 -> new (21d from May 1 >= 15d)
    # - Apr 25 -> May 1 (6d < 15d)
    assert sample.timestamps.shape == (1, 7, 3)

    expected_months = [0, 1, 1, 2, 3, 4, 5]  # 0-indexed months
    expected_days = [1, 1, 3, 1, 10, 1, 1]
    assert (
        sample.timestamps[0, :, 0] == torch.tensor(expected_days, dtype=torch.int32)
    ).all()
    assert (
        sample.timestamps[0, :, 1] == torch.tensor(expected_months, dtype=torch.int32)
    ).all()
    assert (sample.timestamps[0, :, 2] == 2025).all()

    # sentinel2_l2a has data at indices 0, 1, 5, 6; MISSING at 2, 3, 4
    s2_mask = sample.sentinel2_l2a_mask[0]  # H, W, T, S
    for t in [0, 1, 5, 6]:
        assert (s2_mask[:, :, t] == MaskValue.ONLINE_ENCODER.value).all(), (
            f"s2 t={t} should be valid"
        )
    for t in [2, 3, 4]:
        assert (s2_mask[:, :, t] == MaskValue.MISSING.value).all(), (
            f"s2 t={t} should be missing"
        )

    # sentinel1 has data at indices 0, 1, 2, 3, 4, 5; MISSING at 6
    s1_mask = sample.sentinel1_mask[0]  # H, W, T, S
    for t in [0, 1, 2, 3, 4, 5]:
        assert (s1_mask[:, :, t] == MaskValue.ONLINE_ENCODER.value).all(), (
            f"s1 t={t} should be valid"
        )
    assert (s1_mask[:, :, 6] == MaskValue.MISSING.value).all()

    # Verify data placement for sentinel2_l2a (values 1-4 at indices 0,1,5,6)
    s2_data = sample.sentinel2_l2a[0]  # H, W, T, C
    assert torch.allclose(s2_data[0, 0, 0, :], torch.ones(12) * 1.0)  # Jan 1
    assert torch.allclose(s2_data[0, 0, 1, :], torch.ones(12) * 2.0)  # Feb 1
    assert torch.allclose(s2_data[0, 0, 5, :], torch.ones(12) * 3.0)  # May 1
    assert torch.allclose(s2_data[0, 0, 6, :], torch.ones(12) * 4.0)  # Jun 1

    # Verify data placement for sentinel1 (values 10-15 at indices 0,1,2,3,4,5)
    s1_data = sample.sentinel1[0]  # H, W, T, C
    assert torch.allclose(s1_data[0, 0, 0, :], torch.ones(2) * 10.0)  # Jan 10 -> idx 0
    assert torch.allclose(s1_data[0, 0, 1, :], torch.ones(2) * 11.0)  # Feb 2 -> idx 1
    assert torch.allclose(s1_data[0, 0, 2, :], torch.ones(2) * 12.0)  # Feb 3 -> idx 2
    assert torch.allclose(s1_data[0, 0, 3, :], torch.ones(2) * 13.0)  # Mar 1 -> idx 3
    assert torch.allclose(s1_data[0, 0, 4, :], torch.ones(2) * 14.0)  # Apr 10 -> idx 4
    assert torch.allclose(s1_data[0, 0, 5, :], torch.ones(2) * 15.0)  # Apr 25 -> idx 5


def test_normal_timestamps_two_modalities_1hr_tolerance() -> None:
    """With 1-hour tolerance, no cross-modality merging occurs; all timestamps are separate."""
    model = OlmoEarth(
        checkpoint_path="tests/unit/models/olmoearth_pretrain/",
        random_initialization=True,
        patch_size=4,
        embedding_size=128,
        use_legacy_timestamps=False,
        timestamp_error_tolerance=timedelta(hours=1),
    )

    H = 4
    W = 4
    s2_dates = [
        datetime(2025, 1, 1),
        datetime(2025, 2, 1),
        datetime(2025, 5, 1),
        datetime(2025, 6, 1),
    ]
    s1_dates = [
        datetime(2025, 1, 10),
        datetime(2025, 2, 2),
        datetime(2025, 2, 3),
        datetime(2025, 3, 1),
        datetime(2025, 4, 10),
        datetime(2025, 4, 25),
    ]

    s2_tensor = torch.zeros((12, len(s2_dates), H, W), dtype=torch.float32)
    for i in range(len(s2_dates)):
        s2_tensor[:, i, :, :] = float(i + 1)

    s1_tensor = torch.zeros((2, len(s1_dates), H, W), dtype=torch.float32)
    for i in range(len(s1_dates)):
        s1_tensor[:, i, :, :] = float(i + 10)

    inputs = [
        {
            "sentinel2_l2a": RasterImage(
                image=s2_tensor,
                timestamps=[(d, d) for d in s2_dates],
            ),
            "sentinel1": RasterImage(
                image=s1_tensor,
                timestamps=[(d, d) for d in s1_dates],
            ),
        }
    ]
    context = ModelContext(inputs=inputs, metadatas=[_make_metadata((0, 0, H, W))])
    sample, _, _ = model._prepare_modality_inputs(context)

    # With 1hr tolerance, no merging. All 10 timestamps are distinct (sorted):
    # Jan 1, Jan 10, Feb 1, Feb 2, Feb 3, Mar 1, Apr 10, Apr 25, May 1, Jun 1
    assert sample.timestamps.shape == (1, 10, 3)

    expected_days = [1, 10, 1, 2, 3, 1, 10, 25, 1, 1]
    expected_months = [0, 0, 1, 1, 1, 2, 3, 3, 4, 5]
    assert (
        sample.timestamps[0, :, 0] == torch.tensor(expected_days, dtype=torch.int32)
    ).all()
    assert (
        sample.timestamps[0, :, 1] == torch.tensor(expected_months, dtype=torch.int32)
    ).all()

    # sentinel2_l2a at indices 0, 2, 8, 9; MISSING elsewhere
    s2_mask = sample.sentinel2_l2a_mask[0]  # H, W, T, S
    for t in [0, 2, 8, 9]:
        assert (s2_mask[:, :, t] == MaskValue.ONLINE_ENCODER.value).all(), (
            f"s2 t={t} should be valid"
        )
    for t in [1, 3, 4, 5, 6, 7]:
        assert (s2_mask[:, :, t] == MaskValue.MISSING.value).all(), (
            f"s2 t={t} should be missing"
        )

    # sentinel1 at indices 1, 3, 4, 5, 6, 7; MISSING elsewhere
    s1_mask = sample.sentinel1_mask[0]  # H, W, T, S
    for t in [1, 3, 4, 5, 6, 7]:
        assert (s1_mask[:, :, t] == MaskValue.ONLINE_ENCODER.value).all(), (
            f"s1 t={t} should be valid"
        )
    for t in [0, 2, 8, 9]:
        assert (s1_mask[:, :, t] == MaskValue.MISSING.value).all(), (
            f"s1 t={t} should be missing"
        )
