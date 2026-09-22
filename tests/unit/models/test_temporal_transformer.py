"""Tests for rslearn.models.temporal_transformer."""

import torch

from rslearn.models.component import TokenFeatureMaps
from rslearn.models.temporal_transformer import TemporalTransformer
from rslearn.train.model_context import ModelContext

DIM = 16
NUM_TOKENS = 5


def test_shape_preserved() -> None:
    """The output should have the same shape as the input for each map."""
    model = TemporalTransformer(dim=DIM, depth=1, num_heads=2, dim_feedforward=32)
    model.eval()
    maps = [torch.randn(2, DIM, 3, 4, NUM_TOKENS), torch.randn(2, DIM, 1, 2, 3)]
    out = model(TokenFeatureMaps(maps), ModelContext(inputs=[], metadatas=[]))
    assert isinstance(out, TokenFeatureMaps)
    assert len(out.feature_maps) == 2
    for inp, res in zip(maps, out.feature_maps):
        assert res.shape == inp.shape


def test_locations_independent() -> None:
    """Tokens at one spatial location must not influence another location."""
    model = TemporalTransformer(dim=DIM, depth=1, num_heads=2, dim_feedforward=32)
    model.eval()
    x = torch.randn(1, DIM, 1, 2, NUM_TOKENS)
    out = model(TokenFeatureMaps([x]), ModelContext(inputs=[], metadatas=[]))
    # Perturb the second location only; the first location's output is unchanged.
    x2 = x.clone()
    x2[:, :, 0, 1, :] += 1.0
    out2 = model(TokenFeatureMaps([x2]), ModelContext(inputs=[], metadatas=[]))
    assert torch.allclose(
        out.feature_maps[0][:, :, 0, 0, :], out2.feature_maps[0][:, :, 0, 0, :]
    )
    assert not torch.allclose(
        out.feature_maps[0][:, :, 0, 1, :], out2.feature_maps[0][:, :, 0, 1, :]
    )


def test_positional_embedding() -> None:
    """With positional_embedding_num_tokens set, the embedding is used and enforced."""
    model = TemporalTransformer(
        dim=DIM,
        depth=1,
        num_heads=2,
        dim_feedforward=32,
        positional_embedding_num_tokens=NUM_TOKENS,
    )
    model.eval()
    assert model.temporal_pos is not None
    assert model.temporal_pos.shape == (1, NUM_TOKENS, DIM)

    # Identical tokens at every slot produce different outputs only because of the
    # positional embedding.
    x = torch.ones(1, DIM, 1, 1, NUM_TOKENS)
    out = model(TokenFeatureMaps([x]), ModelContext(inputs=[], metadatas=[]))
    tokens = out.feature_maps[0][0, :, 0, 0, :]  # (C, N)
    assert not torch.allclose(tokens[:, 0], tokens[:, 1])


def test_mask_excludes_tokens_and_passes_through() -> None:
    """Masked tokens do not influence valid tokens, and masks are passed through."""
    model = TemporalTransformer(dim=DIM, depth=1, num_heads=2, dim_feedforward=32)
    model.eval()
    context = ModelContext(inputs=[], metadatas=[])
    num_valid = 3
    x = torch.randn(1, DIM, 1, 2, NUM_TOKENS)
    mask = torch.ones(1, 1, 2, NUM_TOKENS, dtype=torch.bool)
    mask[..., num_valid:] = False

    out = model(TokenFeatureMaps([x], masks=[mask]), context)
    assert isinstance(out, TokenFeatureMaps)
    assert out.masks is not None
    assert torch.equal(out.masks[0], mask)
    assert out.feature_maps[0].shape == x.shape
    assert not torch.isnan(out.feature_maps[0]).any()

    # Perturbing the masked tokens leaves the valid token outputs unchanged.
    x2 = x.clone()
    x2[..., num_valid:] += 5.0
    out2 = model(TokenFeatureMaps([x2], masks=[mask]), context)
    assert torch.allclose(
        out.feature_maps[0][..., :num_valid],
        out2.feature_maps[0][..., :num_valid],
        atol=1e-5,
    )

    # The valid token outputs match running the transformer on the truncated
    # (unmasked) sequence.
    out_trunc = model(TokenFeatureMaps([x[..., :num_valid]]), context)
    assert torch.allclose(
        out.feature_maps[0][..., :num_valid], out_trunc.feature_maps[0], atol=1e-5
    )


def test_all_masked_location_no_nan() -> None:
    """A location where every token is masked still produces finite outputs."""
    model = TemporalTransformer(dim=DIM, depth=1, num_heads=2, dim_feedforward=32)
    model.eval()
    x = torch.randn(1, DIM, 1, 2, NUM_TOKENS)
    mask = torch.ones(1, 1, 2, NUM_TOKENS, dtype=torch.bool)
    mask[:, 0, 1, :] = False
    out = model(
        TokenFeatureMaps([x], masks=[mask]), ModelContext(inputs=[], metadatas=[])
    )
    assert torch.isfinite(out.feature_maps[0]).all()
