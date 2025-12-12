"""An attention pooling layer."""

import math
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from rslearn.models.component import FeatureMaps, IntermediateComponent
from rslearn.train.model_context import ModelContext


class SimpleAttentionPool(IntermediateComponent):
    """Simple Attention Pooling.

    Given a feature map of shape BCHWN,
    learn an attention layer which aggregates over
    the N dimension. Do this simply
    """

    def __init__(self, in_dim: int, hidden_linear: bool = False) -> None:
        """Initialize the simple attention pooling layer."""
        super().__init__()
        if hidden_linear:
            self.hidden_linear = nn.Linear(in_features=in_dim, out_features=in_dim)
        else:
            self.hidden_linear = None
        self.linear = nn.Linear(in_features=in_dim, out_features=1)

    def forward_for_map(self, feat_tokens: torch.Tensor) -> torch.Tensor:
        """Attention pooling for a single feature map."""
        B, D, H, W, N = feat_tokens.shape
        feat_tokens = rearrange(feat_tokens, "b d h w n -> (b h w) n d")
        if self.hidden_linear is not None:
            feat_tokens = torch.nn.functional.relu(self.hidden_linear(feat_tokens))
        attention_scores = torch.nn.functional.softmax(self.linear(feat_tokens), dim=1)
        feat_tokens = (attention_scores * feat_tokens).sum(dim=1)
        return rearrange(feat_tokens, "(b h w) d -> b d h w", b=B, h=H, w=W)

    def forward(self, intermediates: Any, context: ModelContext) -> FeatureMaps:
        """Forward pass for attention pooling linear probe.

        Args:
            intermediates: the output from the previous component, which must be a FeatureMaps.
                This FeatureMaps must contain one feature map with an extra dimension at the
                end, which will be pooled over.
            context: the model context.
            feat_tokens (torch.Tensor): Input feature tokens of shape (B, C, H, W, N).

        Returns:
            torch.Tensor:
                - output, attentioned pool over the last dimension (B, C, H, W)
        """
        if not isinstance(intermediates, FeatureMaps):
            raise ValueError("input to Attention Pool must be a FeatureMaps")

        features = []
        for feat in intermediates.feature_maps:
            features.append(self.forward_for_map(feat))
        return FeatureMaps(features)


class AttentionPool(IntermediateComponent):
    """Attention Pooling.

    Given a feature map of shape BCHWN,
    learn an attention layer which aggregates over
    the N dimension.
    """

    def __init__(self, in_dim: int, num_heads: int, linear_on_kv: bool = False) -> None:
        """Initialize the attention pooling layer."""
        super().__init__()
        assert in_dim % 64 == 0, "in_dim must be divisible by 64"
        self.query_token: nn.Parameter = nn.Parameter(torch.empty(in_dim))
        if linear_on_kv:
            self.kv: nn.Linear = nn.Linear(in_dim, in_dim * 2)
        else:
            self.kv = None
        if in_dim % num_heads != 0:
            raise ValueError(
                f"in_dim must be divisible by num_heads. Got {in_dim} and {num_heads}."
            )
        self.num_heads = num_heads
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights for the probe."""
        nn.init.trunc_normal_(self.query_token, std=0.02)
        if self.kv is not None:
            nn.init.trunc_normal_(self.kv.weight, std=0.02)
            nn.init.zeros_(self.kv.bias)

    def forward_for_map(self, feat_tokens: torch.Tensor) -> torch.Tensor:
        """Attention pooling for a single feature map."""
        B, D, H, W, N = feat_tokens.shape
        feat_tokens = rearrange(feat_tokens, "b d h w n -> (b h w) n d")
        collapsed_dim = B * H * W
        q = self.query_token.expand(collapsed_dim, 1, -1)
        q = q.reshape(
            collapsed_dim, 1, self.num_heads, D // self.num_heads
        )  # [B, 1, head, D_head]
        q = rearrange(q, "b h n d -> b n h d")
        if self.kv is not None:
            kv = self.kv(feat_tokens).reshape(
                collapsed_dim, N, 2, self.num_heads, D // self.num_heads
            )  # [B, N, 2, head, D_head]
            kv = rearrange(kv, "b n two h d -> two b h n d")
            k, v = torch.unbind(kv, dim=0)  # 2 * [B, head, N, D_head]
        else:
            k = feat_tokens.reshape(
                collapsed_dim, N, self.num_heads, D // self.num_heads
            )
            k = rearrange(k, "b n h d -> b h n d")

            v = feat_tokens.reshape(
                collapsed_dim, N, self.num_heads, D // self.num_heads
            )
            v = rearrange(v, "b n h d -> b h n d")

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            D // self.num_heads
        )
        attn_weights = F.softmax(attn_scores, dim=-1)
        x = torch.matmul(attn_weights, v)  # [B, head, 1, D_head]
        return x.reshape(B, D, H, W)

    def forward(self, intermediates: Any, context: ModelContext) -> FeatureMaps:
        """Forward pass for attention pooling linear probe.

        Args:
            intermediates: the output from the previous component, which must be a FeatureMaps.
                This FeatureMaps must contain one feature map with an extra dimension at the
                end, which will be pooled over.
            context: the model context.
            feat_tokens (torch.Tensor): Input feature tokens of shape (B, C, H, W, N).

        Returns:
            torch.Tensor:
                - output, attentioned pool over the last dimension (B, C, H, W)
        """
        if not isinstance(intermediates, FeatureMaps):
            raise ValueError("input to Attention Pool must be a FeatureMaps")

        features = []
        for feat in intermediates.feature_maps:
            features.append(self.forward_for_map(feat))
        return FeatureMaps(features)
