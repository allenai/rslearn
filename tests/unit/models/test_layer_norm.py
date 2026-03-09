"""Tests for the rslearn.models.layer_norm module."""

import pytest
import torch

from rslearn.models.component import FeatureMaps, FeatureVector, TokenFeatureMaps
from rslearn.models.layer_norm import LayerNorm
from rslearn.models.pooling_decoder import PoolingDecoder
from rslearn.train.model_context import ModelContext


def test_layer_norm_feature_maps() -> None:
    """Test LayerNorm with FeatureMaps input."""
    features = FeatureMaps(
        [torch.randn(2, 8, 4, 4, dtype=torch.float32), torch.randn(2, 8, 2, 2)]
    )
    layer_norm = LayerNorm(num_channels=8)

    output = layer_norm(features, ModelContext(inputs=[], metadatas=[]))

    assert isinstance(output, FeatureMaps)
    assert len(output.feature_maps) == 2
    assert output.feature_maps[0].shape == (2, 8, 4, 4)
    assert output.feature_maps[1].shape == (2, 8, 2, 2)


def test_layer_norm_after_pooling_decoder() -> None:
    """Test that LayerNorm can run after PoolingDecoder output."""
    batch_size = 2
    in_channels = 8
    out_channels = 6

    feature_maps = FeatureMaps([torch.randn(batch_size, in_channels, 4, 4)])
    decoder = PoolingDecoder(
        in_channels=in_channels,
        out_channels=out_channels,
        num_conv_layers=0,
        num_fc_layers=0,
    )
    layer_norm = LayerNorm(num_channels=out_channels)

    decoded = decoder(feature_maps, ModelContext(inputs=[{}], metadatas=[]))
    output = layer_norm(decoded, ModelContext(inputs=[{}], metadatas=[]))

    assert isinstance(decoded, FeatureVector)
    assert isinstance(output, FeatureVector)
    assert output.feature_vector.shape == (batch_size, out_channels)


def test_layer_norm_token_feature_maps() -> None:
    """Test TokenFeatureMaps support."""
    token_features = TokenFeatureMaps([torch.randn(1, 5, 2, 3, 4)])
    layer_norm = LayerNorm(num_channels=5)

    output = layer_norm(token_features, ModelContext(inputs=[], metadatas=[]))

    assert isinstance(output, TokenFeatureMaps)
    assert output.feature_maps[0].shape == (1, 5, 2, 3, 4)


def test_layer_norm_rejects_unsupported_input() -> None:
    """Test that unsupported inputs raise a helpful error."""
    layer_norm = LayerNorm(num_channels=8)

    with pytest.raises(ValueError, match="input to LayerNorm must be"):
        layer_norm("not-an-intermediate", ModelContext(inputs=[], metadatas=[]))


def test_layer_norm_rejects_channel_mismatch() -> None:
    """Test that channel mismatch raises a helpful error."""
    layer_norm = LayerNorm(num_channels=8)
    features = FeatureMaps([torch.randn(2, 7, 4, 4)])

    with pytest.raises(ValueError, match="input size at normalize_dim does not match"):
        layer_norm(features, ModelContext(inputs=[], metadatas=[]))


def test_layer_norm_custom_normalize_dim() -> None:
    """Test configuring normalization over a non-channel dimension."""
    tensor = torch.randn(2, 4, 3)
    layer_norm = LayerNorm(num_channels=3, normalize_dim=2)

    output = layer_norm(tensor, ModelContext(inputs=[], metadatas=[]))

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 4, 3)
