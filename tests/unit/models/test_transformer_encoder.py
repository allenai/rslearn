"""Unit tests for rslearn.models.transformer_encoder."""

import torch

from rslearn.models.transformer_encoder import PatchTransformerEncoder
from rslearn.train.model_context import ModelContext, RasterImage


def _make_context(
    c: int, t: int, b: int = 2, h: int = 1, w: int = 1, mod_key: str = "era5"
) -> ModelContext:
    inputs = [{mod_key: RasterImage(torch.randn(c, t, h, w))} for _ in range(b)]
    return ModelContext(inputs=inputs, metadatas=[])


def _make_masked_context(
    c: int,
    t: int,
    b: int = 2,
    h: int = 1,
    w: int = 1,
    mod_key: str = "era5",
    mask_ratio: float = 0.4,
) -> ModelContext:
    inputs = []
    for _ in range(b):
        data = torch.randn(c, t, h, w)
        ts_mask = (torch.rand(t) >= mask_ratio).float()
        if ts_mask.sum() == 0:
            ts_mask[0] = 1.0
        mask_ch = ts_mask.view(1, t, 1, 1).expand(1, t, h, w)
        data[:, ts_mask == 0] = 0.0
        combined = torch.cat([mask_ch, data], dim=0)
        inputs.append({mod_key: RasterImage(combined)})
    return ModelContext(inputs=inputs, metadatas=[])


class TestPatchTransformerEncoder:
    """Tests for patch-based Transformer time-series encoder."""

    def test_forward_attention_pooling_no_mask_channel(self) -> None:
        c, t = 14, 365
        enc = PatchTransformerEncoder(
            in_channels=c,
            d_model=96,
            d_output=32,
            num_layers=2,
            num_heads=4,
            patch_kernel_size=14,
            patch_stride=7,
            pooling="attention",
            position_encoding="sinusoidal",
            mod_key="era5",
        )
        enc.eval()
        ctx = _make_context(c, t, b=2, mod_key="era5")
        out = enc(ctx)
        assert out.feature_vector.shape == (2, 32)

    def test_forward_with_mask_channel(self) -> None:
        c, t = 14, 365
        enc = PatchTransformerEncoder(
            in_channels=c,
            d_model=96,
            d_output=32,
            num_layers=2,
            num_heads=4,
            patch_kernel_size=14,
            patch_stride=7,
            pooling="gated",
            has_mask_channel=True,
            mod_key="era5",
        )
        enc.eval()
        ctx = _make_masked_context(c, t, b=2, mod_key="era5", mask_ratio=0.3)
        out = enc(ctx)
        assert out.feature_vector.shape == (2, 32)

    def test_cls_mean_concat_pooling_and_patch_padding(self) -> None:
        c, t = 14, 30  # not divisible by patch size
        enc = PatchTransformerEncoder(
            in_channels=c,
            d_model=64,
            d_output=24,
            num_layers=2,
            num_heads=4,
            patch_kernel_size=7,
            patch_stride=7,
            pooling="cls_mean_concat",
            position_encoding="learned",
            max_position_embeddings=64,
            mod_key="era5",
        )
        enc.eval()
        ctx = _make_context(c, t, b=3, mod_key="era5")
        out = enc(ctx)
        assert out.feature_vector.shape == (3, 24)

    def test_spatial_output_mode(self) -> None:
        c, t = 14, 365
        enc = PatchTransformerEncoder(
            in_channels=c,
            d_model=96,
            d_output=16,
            num_layers=2,
            num_heads=4,
            patch_kernel_size=5,
            patch_stride=5,
            pooling="mean",
            output_spatial_size=4,
            mod_key="era5",
        )
        enc.eval()
        ctx = _make_context(c, t, b=2, mod_key="era5")
        out = enc(ctx)
        assert out.feature_maps[0].shape == (2, 16, 4, 4)

    def test_sinusoidal_position_encoding_odd_dimension(self) -> None:
        c, t = 14, 60
        enc = PatchTransformerEncoder(
            in_channels=c,
            d_model=95,
            d_output=12,
            num_layers=2,
            num_heads=5,
            patch_kernel_size=14,
            patch_stride=7,
            pooling="mean",
            position_encoding="sinusoidal",
            mod_key="era5",
        )
        enc.eval()
        ctx = _make_context(c, t, b=2, mod_key="era5")
        out = enc(ctx)
        assert out.feature_vector.shape == (2, 12)
