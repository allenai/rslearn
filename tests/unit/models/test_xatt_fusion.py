"""Tests for the rslearn.models.xatt_fusion module."""

import pytest
import torch

from rslearn.models.component import FeatureExtractor, FeatureMaps, FeatureVector
from rslearn.models.xatt_fusion import CrossAttentionFusionExtractor
from rslearn.train.model_context import ModelContext


class _FixedVectorExtractor(FeatureExtractor):
    """Feature extractor returning a fixed FeatureVector."""

    def __init__(self, vector: torch.Tensor):
        super().__init__()
        self._vector = vector

    def forward(self, context: ModelContext) -> FeatureVector:
        return FeatureVector(feature_vector=self._vector.clone())


class _FixedMapsExtractor(FeatureExtractor):
    """Feature extractor returning fixed FeatureMaps."""

    def __init__(self, feature_maps: list[torch.Tensor]):
        super().__init__()
        self._feature_maps = feature_maps

    def forward(self, context: ModelContext) -> FeatureMaps:
        return FeatureMaps(feature_maps=[fm.clone() for fm in self._feature_maps])


def _dummy_context(batch_size: int) -> ModelContext:
    return ModelContext(inputs=[{} for _ in range(batch_size)], metadatas=[])


def test_cross_attention_feature_vectors_shape() -> None:
    """Cross-attention vector mode should keep primary channel dim."""
    primary = torch.randn(2, 6)
    context_a = torch.randn(2, 4)
    context_b = torch.randn(2, 5)

    model = CrossAttentionFusionExtractor(
        primary_path=[_FixedVectorExtractor(primary)],
        context_paths=[
            [_FixedVectorExtractor(context_a)],
            [_FixedVectorExtractor(context_b)],
        ],
        primary_output_channels=6,
        context_output_channels=[4, 5],
        attention_dim=8,
        num_memory_tokens=4,
        num_heads=4,
    )

    out = model(_dummy_context(batch_size=2))
    assert isinstance(out, FeatureVector)
    assert out.feature_vector.shape == (2, 6)


def test_cross_attention_feature_vectors_with_ffn_block_shape() -> None:
    """Post-fusion FFN block should preserve primary vector shape."""
    primary = torch.randn(2, 6)
    context_a = torch.randn(2, 4)
    context_b = torch.randn(2, 5)

    model = CrossAttentionFusionExtractor(
        primary_path=[_FixedVectorExtractor(primary)],
        context_paths=[
            [_FixedVectorExtractor(context_a)],
            [_FixedVectorExtractor(context_b)],
        ],
        primary_output_channels=6,
        context_output_channels=[4, 5],
        attention_dim=8,
        num_memory_tokens=4,
        num_heads=4,
        post_fusion_mode="ffn",
        ffn_expansion=3.0,
        ffn_activation="gelu",
    )

    out = model(_dummy_context(batch_size=2))
    assert isinstance(out, FeatureVector)
    assert out.feature_vector.shape == (2, 6)


def test_cross_attention_feature_maps_shape() -> None:
    """Cross-attention maps mode should keep primary map shapes at each scale."""
    primary = [torch.randn(2, 6, 8, 8), torch.randn(2, 6, 4, 4)]
    context = [torch.randn(2, 3, 8, 8), torch.randn(2, 3, 4, 4)]

    model = CrossAttentionFusionExtractor(
        primary_path=[_FixedMapsExtractor(primary)],
        context_paths=[[_FixedMapsExtractor(context)]],
        primary_output_channels=6,
        context_output_channels=[3],
        attention_dim=12,
        num_memory_tokens=4,
        num_heads=4,
    )

    out = model(_dummy_context(batch_size=2))
    assert isinstance(out, FeatureMaps)
    assert len(out.feature_maps) == 2
    assert out.feature_maps[0].shape == primary[0].shape
    assert out.feature_maps[1].shape == primary[1].shape


def test_cross_attention_feature_maps_with_self_attn_ffn_shape() -> None:
    """Post-fusion self-attn+FFN block should preserve map shapes."""
    primary = [torch.randn(2, 6, 8, 8), torch.randn(2, 6, 4, 4)]
    context = [torch.randn(2, 3, 8, 8), torch.randn(2, 3, 4, 4)]

    model = CrossAttentionFusionExtractor(
        primary_path=[_FixedMapsExtractor(primary)],
        context_paths=[[_FixedMapsExtractor(context)]],
        primary_output_channels=6,
        context_output_channels=[3],
        attention_dim=12,
        num_memory_tokens=8,
        num_heads=4,
        post_fusion_mode="self_attn_ffn",
        ffn_expansion=2.0,
        ffn_activation="swiglu",
    )

    out = model(_dummy_context(batch_size=2))
    assert isinstance(out, FeatureMaps)
    assert len(out.feature_maps) == 2
    assert out.feature_maps[0].shape == primary[0].shape
    assert out.feature_maps[1].shape == primary[1].shape


def test_cross_attention_requires_context_path() -> None:
    """Cross-attention requires at least one context path."""
    primary = torch.randn(2, 6)
    with pytest.raises(ValueError, match="at least one context path"):
        CrossAttentionFusionExtractor(
            primary_path=[_FixedVectorExtractor(primary)],
            context_paths=[],
            primary_output_channels=6,
            context_output_channels=[],
        )


def test_cross_attention_validates_runtime_channels() -> None:
    """Configured channels must match runtime outputs."""
    primary = torch.randn(2, 6)
    context = torch.randn(2, 4)
    model = CrossAttentionFusionExtractor(
        primary_path=[_FixedVectorExtractor(primary)],
        context_paths=[[_FixedVectorExtractor(context)]],
        primary_output_channels=7,
        context_output_channels=[4],
        attention_dim=8,
        num_memory_tokens=4,
        num_heads=4,
        pre_fusion_norm=False,
    )

    with pytest.raises(ValueError, match="produced FeatureVector with 6 channels"):
        model(_dummy_context(batch_size=2))


def test_cross_attention_alpha_defaults_to_scalar_zero() -> None:
    """Default alpha should be a scalar initialized to zero."""
    primary = torch.randn(2, 6)
    context = torch.randn(2, 4)
    model = CrossAttentionFusionExtractor(
        primary_path=[_FixedVectorExtractor(primary)],
        context_paths=[[_FixedVectorExtractor(context)]],
        primary_output_channels=6,
        context_output_channels=[4],
        attention_dim=8,
        num_heads=4,
    )
    assert model.cross_attn_alpha.ndim == 0
    assert float(model.cross_attn_alpha.detach().cpu().item()) == 0.0


def test_context_dropout_returns_missing_context_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Context dropout should return per-sample missing-context flags."""
    primary = torch.randn(2, 6)
    context = torch.randn(2, 4)
    model = CrossAttentionFusionExtractor(
        primary_path=[_FixedVectorExtractor(primary)],
        context_paths=[[_FixedVectorExtractor(context)]],
        primary_output_channels=6,
        context_output_channels=[4],
        attention_dim=8,
        num_heads=4,
        context_dropout_prob=0.5,
    )
    model.train()
    monkeypatch.setattr(
        torch,
        "rand",
        lambda size, device=None: torch.zeros(size, device=device),
    )

    missing_context = model._apply_context_dropout(
        [
            FeatureVector(feature_vector=primary.clone()),
            FeatureVector(feature_vector=context.clone()),
        ]
    )
    assert missing_context is not None
    assert torch.equal(missing_context, torch.ones(2, dtype=torch.bool))


def test_forward_stores_path0_intermediate_in_context() -> None:
    """Forward should store the path0 intermediate in context.context_dict."""
    primary = torch.randn(2, 6)
    context_vec = torch.randn(2, 4)
    model = CrossAttentionFusionExtractor(
        primary_path=[_FixedVectorExtractor(primary)],
        context_paths=[[_FixedVectorExtractor(context_vec)]],
        primary_output_channels=6,
        context_output_channels=[4],
        attention_dim=8,
        num_heads=4,
        pre_fusion_norm=False,
    )

    context = _dummy_context(batch_size=2)
    _ = model(context)
    stored = context.context_dict["path0_intermediate"]
    assert isinstance(stored, FeatureVector)
    torch.testing.assert_close(stored.feature_vector, primary)


def test_cross_attend_skips_attention_residual_when_context_missing() -> None:
    """Missing-context mask should zero the cross-attention residual branch."""
    primary = torch.randn(2, 6)
    context = torch.randn(2, 4)
    model = CrossAttentionFusionExtractor(
        primary_path=[_FixedVectorExtractor(primary)],
        context_paths=[[_FixedVectorExtractor(context)]],
        primary_output_channels=6,
        context_output_channels=[4],
        attention_dim=8,
        num_heads=4,
    )
    model.eval()
    with torch.no_grad():
        model.cross_attn_alpha.fill_(1.0)

    primary_tokens = torch.randn(2, 3, 6)
    context_vec = torch.randn(2, 4)
    missing_context = torch.ones(2, dtype=torch.bool)

    out = model._cross_attend(
        primary_tokens, context_vec, missing_context=missing_context
    )
    expected = model.query_out_proj(model.query_in_proj(primary_tokens))
    torch.testing.assert_close(out, expected)
