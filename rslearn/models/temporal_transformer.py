"""Transformer over the token dimension of a TokenFeatureMaps."""

from typing import Any

import torch
from einops import rearrange
from torch import nn

from rslearn.models.component import IntermediateComponent, TokenFeatureMaps
from rslearn.train.model_context import ModelContext


class TemporalTransformer(IntermediateComponent):
    """Apply a per-spatial-patch transformer on the tokens at each patch.

    Given a BCHWN token feature map, apply a transformer independently at each (b, h, w)
    location over the N tokens. Usually these are per-timestep tokens. This component is
    most useful if the underlying encoder does not perform any temporal reasoning, e.g.
    if the per-timestep tokens are derived by concatenating the outputs of a
    single-timestep model across timesteps in an image time series.

    The output has the same shape as the input. If the input TokenFeatureMaps has
    masks, invalid tokens are excluded from attention (as key padding) and the masks
    are passed through unchanged to the output.
    """

    def __init__(
        self,
        dim: int,
        depth: int = 1,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        positional_embedding_num_tokens: int | None = None,
    ) -> None:
        """Create a new TemporalTransformer.

        Args:
            dim: the token embedding dimension C.
            depth: number of transformer encoder layers.
            num_heads: number of attention heads.
            dim_feedforward: hidden size of the feed-forward network in each layer.
            dropout: dropout rate in the transformer layers.
            positional_embedding_num_tokens: if set, add a learned positional
                embedding over this many token slots. Inputs must then have exactly
                this many tokens.
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.positional_embedding_num_tokens = positional_embedding_num_tokens
        if positional_embedding_num_tokens is not None:
            self.temporal_pos: nn.Parameter | None = nn.Parameter(
                torch.randn(1, positional_embedding_num_tokens, dim) * 0.02
            )
        else:
            self.temporal_pos = None

    def forward(self, intermediates: Any, context: ModelContext) -> TokenFeatureMaps:
        """Contextualize the tokens in each feature map.

        Args:
            intermediates: the output from the previous component, which must be a
                TokenFeatureMaps.
            context: the model context.

        Returns:
            a TokenFeatureMaps with the same shapes (and masks) as the input.
        """
        if not isinstance(intermediates, TokenFeatureMaps):
            raise ValueError("input to TemporalTransformer must be a TokenFeatureMaps")

        outputs = []
        for feat_tokens, mask in zip(
            intermediates.feature_maps, intermediates.get_masks()
        ):
            b, _, h, w, n = feat_tokens.shape
            if (
                self.positional_embedding_num_tokens is not None
                and n != self.positional_embedding_num_tokens
            ):
                raise ValueError(
                    "TemporalTransformer expected "
                    f"{self.positional_embedding_num_tokens} tokens, got {n}"
                )
            x = rearrange(feat_tokens, "b c h w n -> (b h w) n c")
            if self.temporal_pos is not None:
                x = x + self.temporal_pos

            src_key_padding_mask = None
            if mask is not None:
                # True in src_key_padding_mask means the token is ignored.
                padding_mask = ~rearrange(mask, "b h w n -> (b h w) n")
                # If every token at a location is masked, attention would be NaN.
                # Un-mask those rows; their outputs are meaningless anyway and
                # downstream components should ignore them via the mask.
                all_masked = padding_mask.all(dim=1, keepdim=True)
                src_key_padding_mask = padding_mask & ~all_masked

            x = self.temporal_encoder(x, src_key_padding_mask=src_key_padding_mask)
            outputs.append(rearrange(x, "(b h w) n c -> b c h w n", b=b, h=h, w=w))
        return TokenFeatureMaps(outputs, masks=intermediates.masks)
