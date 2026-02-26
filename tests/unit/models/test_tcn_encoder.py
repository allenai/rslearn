"""Unit tests for rslearn.models.tcn_encoder mask-channel handling."""

import torch

from rslearn.models.tcn_encoder import (
    AttentionPooling,
    SimpleTCNEncoder,
    TCNEncoder,
    prepare_ts_modality,
)
from rslearn.train.model_context import ModelContext, RasterImage


def _make_context(
    C: int, T: int, B: int = 2, H: int = 1, W: int = 1, mod_key: str = "era5"
) -> ModelContext:
    """Build a minimal ModelContext with ``B`` batch items of shape [C, T, H, W]."""
    inputs = [{mod_key: RasterImage(torch.randn(C, T, H, W))} for _ in range(B)]
    return ModelContext(inputs=inputs, metadatas=[])


def _make_masked_context(
    C: int,
    T: int,
    B: int = 2,
    H: int = 1,
    W: int = 1,
    mod_key: str = "era5",
    mask_ratio: float = 0.4,
) -> ModelContext:
    """Build a ModelContext where channel-0 is a binary mask (1=valid, 0=masked).

    This simulates the output of :class:`RandomTimeMasking` with
    ``append_mask_channel=True``.
    """
    inputs = []
    for _ in range(B):
        data = torch.randn(C, T, H, W)
        # Build a per-timestep mask and broadcast to spatial dims.
        ts_mask = (torch.rand(T) >= mask_ratio).float()
        # Ensure at least one valid timestep.
        if ts_mask.sum() == 0:
            ts_mask[0] = 1.0
        mask_ch = ts_mask.view(1, T, 1, 1).expand(1, T, H, W)
        # Zero out masked timesteps in data channels.
        data[:, ts_mask == 0] = 0.0
        # Prepend mask channel → [C+1, T, H, W]
        combined = torch.cat([mask_ch, data], dim=0)
        inputs.append({mod_key: RasterImage(combined)})
    return ModelContext(inputs=inputs, metadatas=[])


# ---------------------------------------------------------------------------
# prepare_ts_modality
# ---------------------------------------------------------------------------


class TestPrepareModality:
    """Tests for prepare_ts_modality with and without mask channel."""

    def test_basic_shape_no_mask(self) -> None:
        C, T, B = 14, 30, 3
        ctx = _make_context(C, T, B, mod_key="era5")
        data, mask = prepare_ts_modality(ctx, "era5")
        assert data.shape == (B, T, C)
        assert mask.shape == (B, T)
        assert mask.dtype == torch.bool
        # No masking → all True.
        assert mask.all()

    def test_mask_channel_stripped_and_merged(self) -> None:
        """When has_mask_channel=True the mask channel is extracted and data has C cols."""
        C, T, B = 14, 30, 2
        ctx = _make_masked_context(C, T, B, mod_key="era5", mask_ratio=0.5)
        data, mask = prepare_ts_modality(ctx, "era5", has_mask_channel=True)
        # Data should have exactly C channels (mask channel stripped).
        assert data.shape == (B, T, C)
        assert mask.shape == (B, T)
        # At least some timesteps should be masked.
        assert not mask.all(), "Expected some masked timesteps"
        # All mask values should be bool.
        assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# AttentionPooling
# ---------------------------------------------------------------------------


class TestAttentionPooling:
    """Tests for mask-aware AttentionPooling."""

    def test_no_mask_backward_compatible(self) -> None:
        pool = AttentionPooling(d_model=16)
        x = torch.randn(2, 16, 10)
        out = pool(x)
        assert out.shape == (2, 16)

    def test_mask_suppresses_masked_positions(self) -> None:
        """Masked positions should not contribute to the pooled output."""
        torch.manual_seed(42)
        pool = AttentionPooling(d_model=8)
        B, D, T = 1, 8, 6
        x = torch.randn(B, D, T)
        # Mask: only the first 3 timesteps are valid.
        mask = torch.tensor([[True, True, True, False, False, False]])

        out_masked = pool(x, mask=mask)
        # Compare with manually zeroing out and pooling only valid positions.
        x_valid = x[:, :, :3]
        out_valid_only = pool(x_valid)
        # They should be close (same attention weights over the same data).
        assert torch.allclose(out_masked, out_valid_only, atol=1e-5)

    def test_all_valid_matches_no_mask(self) -> None:
        torch.manual_seed(0)
        pool = AttentionPooling(d_model=8)
        x = torch.randn(2, 8, 5)
        all_valid = torch.ones(2, 5, dtype=torch.bool)
        out_mask = pool(x, mask=all_valid)
        out_none = pool(x, mask=None)
        assert torch.allclose(out_mask, out_none, atol=1e-6)


# ---------------------------------------------------------------------------
# SimpleTCNEncoder
# ---------------------------------------------------------------------------


class TestSimpleTCNEncoder:
    """Tests for SimpleTCNEncoder with mask support."""

    def test_forward_no_mask(self) -> None:
        """Basic forward pass without mask channel (backward compat)."""
        C, T = 14, 64
        enc = SimpleTCNEncoder(
            num_conv_layers=2,
            in_channels=C,
            base_dim=16,
            output_dim=32,
            mod_key="era5",
        )
        enc.eval()
        ctx = _make_context(C, T, B=2, mod_key="era5")
        out = enc(ctx)
        assert out.feature_vector.shape == (2, 32)

    def test_forward_with_mask_channel(self) -> None:
        """Forward pass with mask channel should not crash and strip the channel."""
        C, T = 14, 64
        enc = SimpleTCNEncoder(
            num_conv_layers=2,
            in_channels=C,
            base_dim=16,
            output_dim=32,
            mod_key="era5",
            has_mask_channel=True,
        )
        enc.eval()
        ctx = _make_masked_context(C, T, B=2, mod_key="era5", mask_ratio=0.3)
        out = enc(ctx)
        assert out.feature_vector.shape == (2, 32)

    def test_spatial_output_with_mask(self) -> None:
        C, T = 14, 64
        enc = SimpleTCNEncoder(
            num_conv_layers=2,
            in_channels=C,
            base_dim=16,
            output_dim=32,
            mod_key="era5",
            output_spatial_size=5,
            has_mask_channel=True,
        )
        enc.eval()
        ctx = _make_masked_context(C, T, B=2, mod_key="era5")
        out = enc(ctx)
        assert out.feature_maps[0].shape == (2, 32, 5, 5)


# ---------------------------------------------------------------------------
# TCNEncoder
# ---------------------------------------------------------------------------


class TestTCNEncoder:
    """Tests for TCNEncoder with mask support."""

    def test_forward_no_mask(self) -> None:
        """Basic forward pass without mask channel (backward compat)."""
        C, T = 14, 64
        enc = TCNEncoder(
            in_channels=C,
            d_model=32,
            d_output=16,
            dilations=[1, 2, 4],
            pooling_windows=[1, 2],
            mod_key="era5",
            num_groups=8,
        )
        enc.eval()
        ctx = _make_context(C, T, B=2, mod_key="era5")
        out = enc(ctx)
        assert out.feature_vector.shape == (2, 16)

    def test_forward_with_mask_channel(self) -> None:
        """Forward pass with mask channel should not crash and strip the channel."""
        C, T = 14, 64
        enc = TCNEncoder(
            in_channels=C,
            d_model=32,
            d_output=16,
            dilations=[1, 2, 4],
            pooling_windows=[1, 2],
            mod_key="era5",
            num_groups=8,
            has_mask_channel=True,
        )
        enc.eval()
        ctx = _make_masked_context(C, T, B=2, mod_key="era5", mask_ratio=0.3)
        out = enc(ctx)
        assert out.feature_vector.shape == (2, 16)

    def test_spatial_output_with_mask(self) -> None:
        C, T = 14, 64
        enc = TCNEncoder(
            in_channels=C,
            d_model=32,
            d_output=16,
            dilations=[1, 2, 4],
            pooling_windows=[1, 2],
            mod_key="era5",
            num_groups=8,
            output_spatial_size=3,
            has_mask_channel=True,
        )
        enc.eval()
        ctx = _make_masked_context(C, T, B=2, mod_key="era5")
        out = enc(ctx)
        assert out.feature_maps[0].shape == (2, 16, 3, 3)

    def test_masked_timesteps_suppressed_in_attention(self) -> None:
        """Verify that fully masked chunks produce different output than unmasked."""
        torch.manual_seed(123)
        C, T = 14, 64
        enc = TCNEncoder(
            in_channels=C,
            d_model=32,
            d_output=16,
            dilations=[1, 2],
            pooling_windows=[1],
            mod_key="era5",
            num_groups=8,
            has_mask_channel=True,
        )
        enc.eval()

        # All-valid context.
        ctx_valid = _make_masked_context(C, T, B=1, mod_key="era5", mask_ratio=0.0)
        out_valid = enc(ctx_valid)

        # Half-masked context (same underlying data but different mask).
        ctx_masked = _make_masked_context(C, T, B=1, mod_key="era5", mask_ratio=0.5)
        out_masked = enc(ctx_masked)

        # Outputs should differ because the mask changes what attention sees.
        assert not torch.allclose(
            out_valid.feature_vector, out_masked.feature_vector, atol=1e-4
        )
