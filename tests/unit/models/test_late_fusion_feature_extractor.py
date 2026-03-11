"""Tests for the rslearn.models.LateFusionFeatureExtractor module."""

import pytest
import torch

from rslearn.models.component import FeatureExtractor, FeatureMaps, FeatureVector
from rslearn.models.LateFusionFeatureExtractor import LateFusionFeatureExtractor
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


def test_concat_feature_vectors() -> None:
    """Concat mode should concatenate vectors along channel dim."""
    vec_a = torch.randn(2, 3)
    vec_b = torch.randn(2, 5)
    model = LateFusionFeatureExtractor(
        paths=[[_FixedVectorExtractor(vec_a)], [_FixedVectorExtractor(vec_b)]],
        fusion_mode="concat",
    )

    out = model(_dummy_context(batch_size=2))
    assert isinstance(out, FeatureVector)
    assert out.feature_vector.shape == (2, 8)
    torch.testing.assert_close(out.feature_vector, torch.cat([vec_a, vec_b], dim=1))


def test_concat_feature_maps() -> None:
    """Concat mode should concatenate maps at each scale along channel dim."""
    maps_a = [torch.randn(2, 3, 8, 8), torch.randn(2, 3, 4, 4)]
    maps_b = [torch.randn(2, 5, 8, 8), torch.randn(2, 5, 4, 4)]
    model = LateFusionFeatureExtractor(
        paths=[[_FixedMapsExtractor(maps_a)], [_FixedMapsExtractor(maps_b)]],
        fusion_mode="concat",
    )

    out = model(_dummy_context(batch_size=2))
    assert isinstance(out, FeatureMaps)
    assert len(out.feature_maps) == 2
    assert out.feature_maps[0].shape == (2, 8, 8, 8)
    assert out.feature_maps[1].shape == (2, 8, 4, 4)


def test_gated_feature_vectors_shape() -> None:
    """Gated mode should produce configured output vector channels."""
    vec_a = torch.randn(2, 6)
    vec_b = torch.randn(2, 4)
    model = LateFusionFeatureExtractor(
        paths=[[_FixedVectorExtractor(vec_a)], [_FixedVectorExtractor(vec_b)]],
        fusion_mode="gated",
        gated_output_channels=7,
        path_output_channels=[6, 4],
    )

    out = model(_dummy_context(batch_size=2))
    assert isinstance(out, FeatureVector)
    assert out.feature_vector.shape == (2, 7)


def test_mixing_feature_vectors_uses_bottleneck_gate_and_outputs_expected_shape() -> (
    None
):
    """Mixing vector mode should use a bottleneck gate and keep expected output shape."""
    vec_a = torch.randn(2, 6)
    vec_b = torch.randn(2, 4)
    out_ch = 8
    hidden = 5

    model = LateFusionFeatureExtractor(
        paths=[[_FixedVectorExtractor(vec_a)], [_FixedVectorExtractor(vec_b)]],
        fusion_mode="mixing",
        gated_output_channels=out_ch,
        path_output_channels=[6, 4],
        mixing_gate_hidden_dim=hidden,
    )

    assert isinstance(model.mixing_gate, torch.nn.Sequential)
    assert isinstance(model.mixing_gate[0], torch.nn.Linear)
    assert isinstance(model.mixing_gate[2], torch.nn.Linear)
    assert model.mixing_gate[0].out_features == hidden
    assert model.mixing_gate[2].in_features == hidden

    out = model(_dummy_context(batch_size=2))
    assert isinstance(out, FeatureVector)
    assert out.feature_vector.shape == (2, out_ch)


def test_mixing_feature_maps_shape() -> None:
    """Mixing map mode should produce configured output channels at each scale."""
    maps_a = [torch.randn(2, 3, 8, 8), torch.randn(2, 3, 4, 4)]
    maps_b = [torch.randn(2, 5, 8, 8), torch.randn(2, 5, 4, 4)]
    model = LateFusionFeatureExtractor(
        paths=[[_FixedMapsExtractor(maps_a)], [_FixedMapsExtractor(maps_b)]],
        fusion_mode="mixing",
        gated_output_channels=[4, 6],
        path_output_channels=[3, 5],
        mixing_gate_hidden_dim=7,
    )

    out = model(_dummy_context(batch_size=2))
    assert isinstance(out, FeatureMaps)
    assert len(out.feature_maps) == 2
    assert out.feature_maps[0].shape == (2, 4, 8, 8)
    assert out.feature_maps[1].shape == (2, 6, 4, 4)


def test_film_feature_vectors_identity_at_init() -> None:
    """FiLM should start at identity: output equals primary vector at init."""
    primary = torch.randn(2, 6)
    context = torch.randn(2, 4)
    model = LateFusionFeatureExtractor(
        paths=[[_FixedVectorExtractor(primary)], [_FixedVectorExtractor(context)]],
        fusion_mode="film",
        gated_output_channels=1,
        path_output_channels=[6, 4],
    )

    out = model(_dummy_context(batch_size=2))
    assert isinstance(out, FeatureVector)
    torch.testing.assert_close(out.feature_vector, primary)


def test_film_feature_maps_identity_at_init() -> None:
    """FiLM map mode should start at identity for primary maps."""
    primary = [torch.randn(2, 6, 8, 8)]
    context = [torch.randn(2, 4, 8, 8)]
    model = LateFusionFeatureExtractor(
        paths=[[_FixedMapsExtractor(primary)], [_FixedMapsExtractor(context)]],
        fusion_mode="film",
        gated_output_channels=[1],
        path_output_channels=[6, 4],
    )

    out = model(_dummy_context(batch_size=2))
    assert isinstance(out, FeatureMaps)
    assert len(out.feature_maps) == 1
    torch.testing.assert_close(out.feature_maps[0], primary[0])


def test_context_path_dropout_masks_context_vectors_and_preserves_path0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Context path dropout should mask path 1+ and keep path 0 unchanged."""
    vec_a = torch.randn(2, 3)
    vec_b = torch.randn(2, 4)
    model = LateFusionFeatureExtractor(
        paths=[[_FixedVectorExtractor(vec_a)], [_FixedVectorExtractor(vec_b)]],
        fusion_mode="concat",
        path_dropout_prob=0.5,
    )
    model.train()
    monkeypatch.setattr(
        torch,
        "rand",
        lambda size, device=None: torch.zeros(size, device=device),
    )

    context = _dummy_context(batch_size=2)
    out = model(context)
    assert isinstance(out, FeatureVector)
    torch.testing.assert_close(out.feature_vector[:, :3], vec_a)
    torch.testing.assert_close(out.feature_vector[:, 3:], torch.zeros_like(vec_b))

    stored = context.context_dict["path0_intermediate"]
    assert isinstance(stored, FeatureVector)
    torch.testing.assert_close(stored.feature_vector, vec_a)
    masks = context.context_dict["path_dropout_masks"]
    assert masks is not None
    assert len(masks) == 1
    assert torch.equal(masks[0], torch.zeros(2, dtype=torch.bool))


def test_context_path_dropout_disabled_in_eval_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Context path dropout should be disabled when the module is in eval mode."""
    vec_a = torch.randn(2, 3)
    vec_b = torch.randn(2, 4)
    model = LateFusionFeatureExtractor(
        paths=[[_FixedVectorExtractor(vec_a)], [_FixedVectorExtractor(vec_b)]],
        fusion_mode="concat",
        path_dropout_prob=0.5,
    )
    model.eval()
    monkeypatch.setattr(
        torch,
        "rand",
        lambda size, device=None: torch.zeros(size, device=device),
    )

    out = model(_dummy_context(batch_size=2))
    assert isinstance(out, FeatureVector)
    torch.testing.assert_close(out.feature_vector, torch.cat([vec_a, vec_b], dim=1))
