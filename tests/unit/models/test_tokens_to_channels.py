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
