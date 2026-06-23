"""Patch-based Transformer encoder for time series modalities."""

from __future__ import annotations

import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from rslearn.models.component import FeatureExtractor, FeatureMaps, FeatureVector
from rslearn.models.tcn_encoder import prepare_ts_modality
from rslearn.train.model_context import ModelContext


def _midpoint_day_of_year(ts: tuple[datetime, datetime]) -> int:
    """Return day-of-year from the midpoint of a (start, end) timestamp tuple."""
    start, end = ts
    midpoint = start + (end - start) / 2
    return midpoint.timetuple().tm_yday


def _sinusoidal_positions(
    seq_len: int, dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Create sinusoidal positional encodings with shape [1, seq_len, dim]."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    positions = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim)
    )
    enc = torch.zeros(seq_len, dim, device=device, dtype=dtype)
    enc[:, 0::2] = torch.sin(positions * div_term)
    # For odd dims, the cosine branch has one fewer channel than sine.
    cos_dim = enc[:, 1::2].shape[1]
    enc[:, 1::2] = torch.cos(positions * div_term[:cos_dim])
    return enc.unsqueeze(0)


class StochasticDepth(nn.Module):
    """Drop residual branches stochastically during training."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        """Create stochastic depth module.

        Args:
            drop_prob: probability of dropping a residual branch during training.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth."""
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
        attention_dropout: float,
        drop_path: float,
    ) -> None:
        """Create a pre-norm Transformer encoder block.

        Args:
            d_model: token embedding dimension.
            num_heads: number of attention heads.
            mlp_ratio: hidden expansion ratio in the feed-forward network.
            dropout: dropout probability in the MLP.
            attention_dropout: dropout probability in attention.
            drop_path: stochastic depth rate for residual branches.
        """
        super().__init__()
        hidden_dim = d_model * mlp_ratio
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.drop_path1 = StochasticDepth(drop_path)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path2 = StochasticDepth(drop_path)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass."""
        attn_input = self.norm1(x)
        attn_out, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class PatchTransformerEncoder(FeatureExtractor):
    """Patch-based Transformer encoder for CTHW time series.

    Expected input per sample is ``[C, T, H, W]``. Spatial dimensions are averaged
    by :func:`prepare_ts_modality` so this module always operates on sequences.
    """

    def __init__(
        self,
        in_channels: int = 14,
        d_model: int = 192,
        d_output: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        patch_kernel_size: int = 14,
        patch_stride: int = 7,
        mlp_ratio: int = 4,
        head_mlp_ratio: int = 2,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        pooling: str = "attention",
        add_day_of_year_features: bool = True,
        add_relative_position_features: bool = False,
        position_encoding: str = "learned",
        max_position_embeddings: int = 512,
        mod_key: str = "era5_daily",
        output_spatial_size: int | None = None,
        has_mask_channel: bool = False,
        pad_value: float = 0.0,
    ) -> None:
        """Create a PatchTransformerEncoder.

        Args:
            in_channels: input variables per timestep.
            d_model: token embedding dimension.
            d_output: output embedding dimension.
            num_layers: number of Transformer encoder blocks.
            num_heads: number of attention heads.
            patch_kernel_size: kernel size (timesteps) of Conv1D patch embedding.
            patch_stride: stride of Conv1D patch embedding.
            mlp_ratio: MLP expansion ratio inside each Transformer block.
            head_mlp_ratio: MLP expansion ratio in final prediction head.
            dropout: dropout probability for token and MLP layers.
            attention_dropout: dropout in multi-head attention.
            drop_path_rate: stochastic depth rate (linearly increased per layer).
            pooling: one of ``attention``, ``gated``, ``mean``, ``cls``,
                ``cls_mean_concat``.
            add_day_of_year_features: add sin/cos day-of-year features.
            add_relative_position_features: add sin/cos relative index features.
            position_encoding: one of ``learned``, ``sinusoidal``, ``none``.
            max_position_embeddings: max token count for learned positional embeddings.
            mod_key: key in each input dict for this modality.
            output_spatial_size: if set, return replicated ``FeatureMaps``.
            has_mask_channel: if True, channel 0 is a validity mask channel.
            pad_value: value used to pad variable-length sequences.
        """
        super().__init__()
        if patch_kernel_size < 1:
            raise ValueError("patch_kernel_size must be >= 1")
        if patch_stride < 1:
            raise ValueError("patch_stride must be >= 1")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if mlp_ratio < 1 or head_mlp_ratio < 1:
            raise ValueError("mlp_ratio and head_mlp_ratio must be >= 1")
        if drop_path_rate < 0.0 or drop_path_rate >= 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")
        if pooling not in {"attention", "gated", "mean", "cls", "cls_mean_concat"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}")
        if position_encoding not in {"learned", "sinusoidal", "none"}:
            raise ValueError(f"Unsupported position_encoding: {position_encoding}")

        self.in_channels = in_channels
        self.d_model = d_model
        self.patch_kernel_size = patch_kernel_size
        self.patch_stride = patch_stride
        self.pooling = pooling
        self.mod_key = mod_key
        self.output_spatial_size = output_spatial_size
        self.has_mask_channel = has_mask_channel
        self.pad_value = pad_value
        self.position_encoding = position_encoding
        self.max_position_embeddings = max_position_embeddings
        self.add_day_of_year_features = add_day_of_year_features
        self.add_relative_position_features = add_relative_position_features

        self.patch_embed = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=self.patch_kernel_size,
            stride=self.patch_stride,
        )
        self.token_dropout = nn.Dropout(dropout)

        self.num_time_features = 0
        if add_day_of_year_features:
            self.num_time_features += 2
        if add_relative_position_features:
            self.num_time_features += 2
        self.time_embed = (
            nn.Linear(self.num_time_features, d_model)
            if self.num_time_features > 0
            else None
        )

        if position_encoding == "learned":
            self.pos_embed = nn.Parameter(
                torch.zeros(1, max_position_embeddings, d_model)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        if pooling in {"cls", "cls_mean_concat"}:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        if pooling == "attention":
            self.attn_query = nn.Parameter(torch.zeros(d_model))
            nn.init.normal_(self.attn_query, std=0.02)
        else:
            self.attn_query = None

        self.gate_proj = nn.Linear(d_model, 1) if pooling == "gated" else None

        drop_rates = torch.linspace(0.0, drop_path_rate, steps=num_layers).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path=drop_rates[i],
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

        pooled_dim = d_model if pooling != "cls_mean_concat" else 2 * d_model
        head_hidden = pooled_dim * head_mlp_ratio
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, d_output),
        )

    def _build_time_features(
        self,
        context: ModelContext,
        max_t: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Build per-timestep calendar/relative features as [B, T, F]."""
        if self.num_time_features == 0:
            return None

        batch_features = []
        for input_dict in context.inputs:
            raster = input_dict[self.mod_key]
            t_len = raster.image.shape[1]
            idx = torch.arange(t_len, device=device, dtype=dtype)

            feats = []
            if self.add_day_of_year_features:
                if raster.timestamps is not None and len(raster.timestamps) == t_len:
                    doy_vals = [_midpoint_day_of_year(ts) for ts in raster.timestamps]
                    doy = torch.tensor(doy_vals, device=device, dtype=dtype)
                else:
                    doy = (idx % 365.0) + 1.0
                doy_angle = 2.0 * math.pi * (doy - 1.0) / 365.0
                feats.extend([torch.sin(doy_angle), torch.cos(doy_angle)])

            if self.add_relative_position_features:
                rel = idx / max(t_len - 1, 1)
                rel_angle = 2.0 * math.pi * rel
                feats.extend([torch.sin(rel_angle), torch.cos(rel_angle)])

            cur = torch.stack(feats, dim=-1)  # [T, F]
            if t_len < max_t:
                cur = F.pad(cur, (0, 0, 0, max_t - t_len), value=0.0)
            batch_features.append(cur)

        return torch.stack(batch_features, dim=0)  # [B, T, F]

    def _patchify(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Patchify [B, T, C] into [B, N, K, C], and mask [B, N, K]."""
        bsz, seq_len, _ = x.shape
        kernel = self.patch_kernel_size
        stride = self.patch_stride
        if seq_len < kernel:
            pad_t = kernel - seq_len
        else:
            remainder = (seq_len - kernel) % stride
            pad_t = (stride - remainder) % stride
        if pad_t > 0:
            x = F.pad(x, (0, 0, 0, pad_t), value=0.0)
            mask = F.pad(mask, (0, pad_t), value=False)

        x = x.transpose(1, 2)  # [B, C, T]
        x = x.unfold(dimension=2, size=kernel, step=stride)  # [B, C, N, K]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, N, K, C]
        mask = mask.unfold(dimension=1, size=kernel, step=stride)  # [B, N, K]
        return x, mask

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mask-aware softmax that remains finite when all positions are invalid."""
        fill_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask, fill_value)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * mask.float()
        denom = weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return weights / denom

    def _pool_tokens(
        self, tokens: torch.Tensor, token_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool token sequence to one vector per sample."""
        if self.pooling == "mean":
            weights = token_mask.float()
            denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
            return (tokens * weights.unsqueeze(-1)).sum(dim=1) / denom

        if self.pooling == "attention":
            assert self.attn_query is not None
            scale = self.d_model**-0.5
            scores = (tokens * self.attn_query).sum(dim=-1) * scale
            weights = self._masked_softmax(scores, token_mask)
            return (tokens * weights.unsqueeze(-1)).sum(dim=1)

        if self.pooling == "gated":
            assert self.gate_proj is not None
            gates = torch.sigmoid(self.gate_proj(tokens)).squeeze(-1)
            gates = gates * token_mask.float()
            denom = gates.sum(dim=1, keepdim=True).clamp(min=1e-6)
            return (tokens * gates.unsqueeze(-1)).sum(dim=1) / denom

        if self.pooling == "cls":
            return tokens[:, 0]

        if self.pooling == "cls_mean_concat":
            cls_vec = tokens[:, 0]
            tail = tokens[:, 1:]
            tail_mask = token_mask[:, 1:]
            weights = tail_mask.float()
            denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
            mean_vec = (tail * weights.unsqueeze(-1)).sum(dim=1) / denom
            return torch.cat([cls_vec, mean_vec], dim=1)

        raise ValueError(f"Unsupported pooling mode: {self.pooling}")

    def forward(self, context: ModelContext) -> FeatureVector | FeatureMaps:
        """Extract a sequence embedding from the configured time-series modality."""
        seq, mask = prepare_ts_modality(
            context,
            self.mod_key,
            pad_value=self.pad_value,
            has_mask_channel=self.has_mask_channel,
        )  # seq [B, T, C], mask [B, T]

        seq = seq * mask.unsqueeze(-1)

        bsz, max_t, _ = seq.shape
        patch_data, patch_mask = self._patchify(seq, mask)
        token_mask = patch_mask.any(dim=-1)  # [B, N]

        seq_for_conv = seq.transpose(1, 2)  # [B, C, T]
        kernel = self.patch_kernel_size
        stride = self.patch_stride
        if max_t < kernel:
            conv_pad_t = kernel - max_t
        else:
            conv_remainder = (max_t - kernel) % stride
            conv_pad_t = (stride - conv_remainder) % stride
        if conv_pad_t > 0:
            seq_for_conv = F.pad(seq_for_conv, (0, conv_pad_t), value=0.0)
        tokens = self.patch_embed(seq_for_conv).transpose(1, 2)  # [B, N, d_model]

        time_feats = self._build_time_features(
            context, max_t=max_t, device=seq.device, dtype=seq.dtype
        )
        if time_feats is not None:
            if time_feats.shape[1] != max_t:
                raise ValueError(
                    f"time feature length {time_feats.shape[1]} does not match sequence {max_t}"
                )
            if time_feats.shape[2] != self.num_time_features:
                raise ValueError(
                    f"time feature dim {time_feats.shape[2]} does not match expected "
                    f"{self.num_time_features}"
                )
            expected_t = (
                patch_data.shape[1] - 1
            ) * self.patch_stride + self.patch_kernel_size
            if expected_t != time_feats.shape[1]:
                pad_len = expected_t - time_feats.shape[1]
                time_feats = F.pad(time_feats, (0, 0, 0, pad_len), value=0.0)
            patch_time = time_feats.unfold(
                dimension=1, size=self.patch_kernel_size, step=self.patch_stride
            )  # [B, N, F, K]
            patch_time = patch_time.permute(0, 1, 3, 2).contiguous()  # [B, N, K, F]
            patch_weights = patch_mask.float().unsqueeze(-1)
            denom = patch_weights.sum(dim=2).clamp(min=1.0)
            patch_time = (patch_time * patch_weights).sum(dim=2) / denom
            assert self.time_embed is not None
            tokens = tokens + self.time_embed(patch_time)

        if self.position_encoding == "learned":
            assert self.pos_embed is not None
            n_tokens = tokens.shape[1]
            if n_tokens > self.max_position_embeddings:
                raise ValueError(
                    f"n_tokens={n_tokens} exceeds max_position_embeddings="
                    f"{self.max_position_embeddings}"
                )
            tokens = tokens + self.pos_embed[:, :n_tokens]
        elif self.position_encoding == "sinusoidal":
            tokens = tokens + _sinusoidal_positions(
                tokens.shape[1], self.d_model, tokens.device, tokens.dtype
            )

        if self.cls_token is not None:
            cls = self.cls_token.expand(bsz, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            cls_valid = torch.ones(bsz, 1, dtype=torch.bool, device=tokens.device)
            token_mask = torch.cat([cls_valid, token_mask], dim=1)
        else:
            # Avoid all-True key padding masks which can produce invalid attention outputs.
            empty_rows = ~token_mask.any(dim=1)
            if empty_rows.any():
                token_mask = token_mask.clone()
                token_mask[empty_rows, 0] = True

        tokens = self.token_dropout(tokens)
        key_padding_mask = ~token_mask
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=key_padding_mask)
        tokens = self.final_norm(tokens)

        pooled = self._pool_tokens(tokens, token_mask)
        out = self.head(pooled)

        if self.output_spatial_size is not None:
            out = out.view(out.shape[0], -1, 1, 1).expand(
                out.shape[0], -1, self.output_spatial_size, self.output_spatial_size
            )
            return FeatureMaps([out])

        return FeatureVector(feature_vector=out)
