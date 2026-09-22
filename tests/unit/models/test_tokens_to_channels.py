"""Tests for rslearn.models.tokens_to_channels."""

import torch

from rslearn.models.component import FeatureMaps, TokenFeatureMaps
from rslearn.models.tokens_to_channels import TokensToChannels
from rslearn.train.model_context import ModelContext

DIM = 8
NUM_TOKENS = 5
CONTEXT = ModelContext(inputs=[], metadatas=[])


def test_shape_and_values() -> None:
    """Each token is projected independently and laid out in token order."""
    model = TokensToChannels(in_dim=DIM, out_dim=1)
    x = torch.randn(2, DIM, 3, 4, NUM_TOKENS)
    out = model(TokenFeatureMaps([x]), CONTEXT)
    assert isinstance(out, FeatureMaps)
    assert len(out.feature_maps) == 1
    feat = out.feature_maps[0]
    assert feat.shape == (2, NUM_TOKENS, 3, 4)

    # Check one token against a manual application of the linear layer.
    b, h, w, n = 1, 2, 3, 4
    expected = model.linear(x[b, :, h, w, n])
    assert torch.allclose(feat[b, n, h, w], expected[0])


def test_out_dim() -> None:
    """With out_dim > 1, the per-token outputs are contiguous in the channel dim."""
    model = TokensToChannels(in_dim=DIM, out_dim=3)
    x = torch.randn(1, DIM, 2, 2, NUM_TOKENS)
    feat = model(TokenFeatureMaps([x]), CONTEXT).feature_maps[0]
    assert feat.shape == (1, NUM_TOKENS * 3, 2, 2)
    n = 2
    expected = model.linear(x[0, :, 1, 0, n])
    assert torch.allclose(feat[0, n * 3 : (n + 1) * 3, 1, 0], expected)


def test_mask_fills_invalid_tokens() -> None:
    """Outputs of masked tokens are replaced by mask_fill_value, others unchanged."""
    fill = -7.0
    model = TokensToChannels(in_dim=DIM, out_dim=2, mask_fill_value=fill)
    x = torch.randn(2, DIM, 2, 3, NUM_TOKENS)
    mask = torch.ones(2, 2, 3, NUM_TOKENS, dtype=torch.bool)
    # Sample 1 only has the first three tokens valid.
    mask[1, ..., 3:] = False

    unmasked = model(TokenFeatureMaps([x]), CONTEXT).feature_maps[0]
    masked = model(TokenFeatureMaps([x], masks=[mask]), CONTEXT).feature_maps[0]
    assert masked.shape == unmasked.shape == (2, NUM_TOKENS * 2, 2, 3)

    # Sample 0 is fully valid so it is unchanged.
    assert torch.allclose(masked[0], unmasked[0])
    # Sample 1: valid token channels unchanged, invalid ones filled.
    assert torch.allclose(masked[1, : 3 * 2], unmasked[1, : 3 * 2])
    assert (masked[1, 3 * 2 :] == fill).all()

    # Perturbing the masked tokens does not change the output.
    x2 = x.clone()
    x2[1, ..., 3:] += 100.0
    masked2 = model(TokenFeatureMaps([x2], masks=[mask]), CONTEXT).feature_maps[0]
    assert torch.allclose(masked, masked2)
