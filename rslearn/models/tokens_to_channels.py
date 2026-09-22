"""Project each token of a TokenFeatureMaps and stack the results as channels."""

from typing import Any

import torch
from einops import rearrange
from torch import nn

from rslearn.train.model_context import ModelContext

from .component import FeatureMaps, IntermediateComponent, TokenFeatureMaps


class TokensToChannels(IntermediateComponent):
    """Apply a shared linear projection per token and lay the tokens out as channels.

    Given a BCHWN token feature map, a linear layer C -> out_dim is applied to every
    token independently and the results are concatenated along the channel dimension
    in token order, producing a B x (N * out_dim) x H x W feature map.

    With out_dim=1 this turns per-timestep tokens into one logit per timestep at each
    location, e.g. for predicting the timestep at which an event occurred as an
    N-way classification per pixel.

    If the input TokenFeatureMaps has masks, the output channels corresponding to
    invalid tokens are set to mask_fill_value (the output shape does not change).
    """

    def __init__(
        self, in_dim: int, out_dim: int = 1, mask_fill_value: float = 0.0
    ) -> None:
        """Create a new TokensToChannels.

        Args:
            in_dim: the token embedding dimension C.
            out_dim: the number of output values per token.
            mask_fill_value: the value to write into the output channels of tokens
                that are marked invalid by the input mask (if any).
        """
        super().__init__()
        self.out_dim = out_dim
        self.mask_fill_value = mask_fill_value
        self.linear = nn.Linear(in_dim, out_dim)

    def forward_for_map(
        self, feat_tokens: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Project a single BCHWN token feature map to B(N*out_dim)HW.

        Args:
            feat_tokens: the BCHWN token feature map.
            mask: optional BHWN bool mask of valid tokens. Outputs for invalid tokens
                are set to mask_fill_value.
        """
        x = rearrange(feat_tokens, "b c h w n -> b h w n c")
        x = self.linear(x)  # (B, H, W, N, out_dim)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), self.mask_fill_value)
        return rearrange(x, "b h w n d -> b (n d) h w")

    def forward(self, intermediates: Any, context: ModelContext) -> FeatureMaps:
        """Project the tokens in each feature map.

        Args:
            intermediates: the output from the previous component, which must be a
                TokenFeatureMaps.
            context: the model context.

        Returns:
            a FeatureMaps where each map is B x (N * out_dim) x H x W.
        """
        if not isinstance(intermediates, TokenFeatureMaps):
            raise ValueError("input to TokensToChannels must be a TokenFeatureMaps")
        return FeatureMaps(
            [
                self.forward_for_map(feat, mask)
                for feat, mask in zip(
                    intermediates.feature_maps, intermediates.get_masks()
                )
            ]
        )
