"""Cross-attention fusion extractor with learned context-conditioned memory tokens.

This module runs multiple encoder paths in parallel, then fuses them by:
1) treating the primary path as the query stream,
2) turning the context paths into a compact context vector,
3) mapping that context vector into a small learned memory bank (K, V),
4) cross-attending primary tokens over that memory bank.

It supports both ``FeatureVector`` and ``FeatureMaps`` outputs.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F
from einops import rearrange

from rslearn.train.model_context import ModelContext

from .component import (
    FeatureExtractor,
    FeatureMaps,
    FeatureVector,
    IntermediateComponent,
)


class CrossAttentionFusionExtractor(FeatureExtractor):
    """Late-fusion feature extractor using cross-attention over learned memory tokens.

    The primary path provides the query stream.  Context paths are compressed into
    a compact memory bank for cross-attention.
    """

    def __init__(
        self,
        primary_path: list[FeatureExtractor | IntermediateComponent],
        context_paths: list[list[FeatureExtractor | IntermediateComponent]],
        primary_output_channels: int,
        context_output_channels: list[int],
        attention_dim: int = 256,
        num_memory_tokens: int = 4,
        num_heads: int = 4,
        attention_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        memory_hidden_dim: int | None = None,
        post_fusion_mode: Literal["none", "ffn", "self_attn_ffn"] = "none",
        ffn_expansion: float = 2.0,
        ffn_activation: Literal["gelu", "swiglu"] = "gelu",
        ffn_dropout: float = 0.0,
        pre_fusion_norm: bool = True,
        pre_fusion_dropout: float = 0.0,
        normalize_memory_values: bool = True,
        context_dropout_prob: float = 0.0,
        primary_context_key: str = "path0_intermediate",
    ):
        """Create a CrossAttentionFusionExtractor.

        Args:
            primary_path: the primary/query encoder path.  The first module must
                be a ``FeatureExtractor``; subsequent modules must be
                ``IntermediateComponent`` instances.
            context_paths: one or more context encoder paths used to build the
                memory bank for cross-attention.  Each path follows the same
                structure as ``primary_path``.
            primary_output_channels: channel dimension produced by the primary
                path.
            context_output_channels: channel dimension produced by each context
                path.  Must have one entry per context path.
            attention_dim: internal attention embedding dimension.
            num_memory_tokens: number of learned context-conditioned memory tokens.
            num_heads: number of attention heads.
            attention_dropout: dropout inside multi-head attention.
            residual_dropout: dropout applied to the attention residual branch.
            memory_hidden_dim: optional hidden dimension for the context->KV MLP.
                If ``None``, a single linear projection is used.
            post_fusion_mode: optional post-cross-attention refinement:
                ``"none"`` keeps cross-attn only, ``"ffn"`` adds a tiny pre-LN FFN
                residual block, and ``"self_attn_ffn"`` adds pre-LN self-attention
                followed by FFN.
            ffn_expansion: FFN hidden expansion ratio relative to ``attention_dim``.
            ffn_activation: FFN activation type, one of ``"gelu"`` or ``"swiglu"``.
            ffn_dropout: dropout inside FFN and on FFN residual branch.
            pre_fusion_norm: if ``True``, apply a learned ``LayerNorm`` to each
                path output before fusion.
            pre_fusion_dropout: dropout probability applied to each path output
                right before fusion.
            normalize_memory_values: if ``True``, apply ``LayerNorm`` to the
                memory value vectors before cross-attention. This can help
                stabilize early training when the memory MLP weights are
                still randomly initialized. Default ``True``.
            context_dropout_prob: probability of dropping all context for a
                given sample during training. When dropped, the cross-attention
                residual is zeroed so the model falls back to primary-only
                features. Default ``0.0`` (disabled).
            primary_context_key: key used to store primary path intermediate
                output in ``context.context_dict`` for optional downstream
                auxiliary supervision.
        """
        super().__init__()

        if len(context_paths) == 0:
            raise ValueError(
                "CrossAttentionFusionExtractor requires at least one context path."
            )

        paths = [primary_path] + list(context_paths)
        for i, path in enumerate(paths):
            if len(path) == 0:
                raise ValueError(f"Path {i} is empty")
            if not isinstance(path[0], FeatureExtractor):
                raise TypeError(
                    f"The first module in path {i} must be a FeatureExtractor, "
                    f"got {type(path[0]).__name__}"
                )
            for j, module in enumerate(path[1:], start=1):
                if not isinstance(module, IntermediateComponent):
                    raise TypeError(
                        f"Module {j} in path {i} must be an IntermediateComponent, "
                        f"got {type(module).__name__}"
                    )

        if len(context_output_channels) != len(context_paths):
            raise ValueError(
                f"context_output_channels must have one entry per context path "
                f"({len(context_paths)}), got {len(context_output_channels)}"
            )
        path_output_channels = [primary_output_channels] + list(context_output_channels)
        for i, channels in enumerate(path_output_channels):
            if channels <= 0:
                raise ValueError(
                    f"path_output_channels[{i}] must be a positive int, got {channels!r}"
                )

        if attention_dim <= 0:
            raise ValueError(
                f"attention_dim must be a positive int, got {attention_dim!r}"
            )
        if num_memory_tokens <= 0:
            raise ValueError(
                f"num_memory_tokens must be a positive int, got {num_memory_tokens!r}"
            )
        if num_heads <= 0:
            raise ValueError(f"num_heads must be a positive int, got {num_heads!r}")
        if attention_dim % num_heads != 0:
            raise ValueError(
                f"attention_dim ({attention_dim}) must be divisible by num_heads ({num_heads})."
            )
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError(
                f"attention_dropout must be in [0, 1), got {attention_dropout!r}"
            )
        if not 0.0 <= residual_dropout < 1.0:
            raise ValueError(
                f"residual_dropout must be in [0, 1), got {residual_dropout!r}"
            )
        if not 0.0 <= pre_fusion_dropout < 1.0:
            raise ValueError(
                f"pre_fusion_dropout must be in [0, 1), got {pre_fusion_dropout!r}"
            )
        if not 0.0 <= context_dropout_prob < 1.0:
            raise ValueError(
                f"context_dropout_prob must be in [0, 1), got {context_dropout_prob!r}"
            )
        if memory_hidden_dim is not None and memory_hidden_dim <= 0:
            raise ValueError(
                f"memory_hidden_dim must be a positive int when set, got {memory_hidden_dim!r}"
            )
        if ffn_expansion <= 0:
            raise ValueError(f"ffn_expansion must be > 0, got {ffn_expansion!r}")
        if not 0.0 <= ffn_dropout < 1.0:
            raise ValueError(f"ffn_dropout must be in [0, 1), got {ffn_dropout!r}")

        self.paths = torch.nn.ModuleList([torch.nn.ModuleList(path) for path in paths])
        self.path_output_channels = list(path_output_channels)

        self._primary_channels = primary_output_channels
        self._context_channels = sum(context_output_channels)
        self.num_memory_tokens = num_memory_tokens
        self.attention_dim = attention_dim
        self.residual_dropout = residual_dropout
        self.post_fusion_mode = post_fusion_mode
        self.ffn_dropout = ffn_dropout
        self.context_dropout_prob = context_dropout_prob
        self.primary_context_key = primary_context_key

        self.query_in_proj = torch.nn.Linear(self._primary_channels, attention_dim)
        self.cross_attn_norm = torch.nn.LayerNorm(attention_dim)
        self.key_norm = torch.nn.LayerNorm(attention_dim)
        self.normalize_memory_values = normalize_memory_values
        if normalize_memory_values:
            self.value_norm = torch.nn.LayerNorm(attention_dim)
        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.cross_attn_alpha = torch.nn.Parameter(torch.tensor(0.0))

        if post_fusion_mode == "self_attn_ffn":
            self.self_attn_norm = torch.nn.LayerNorm(attention_dim)
            self.self_attn = torch.nn.MultiheadAttention(
                embed_dim=attention_dim,
                num_heads=num_heads,
                dropout=attention_dropout,
                batch_first=True,
            )

        if post_fusion_mode in ("ffn", "self_attn_ffn"):
            self.ffn_norm = torch.nn.LayerNorm(attention_dim)
            ffn_hidden = max(1, int(round(attention_dim * ffn_expansion)))
            if ffn_activation == "gelu":
                self.ffn = torch.nn.Sequential(
                    torch.nn.Linear(attention_dim, ffn_hidden),
                    torch.nn.GELU(),
                    torch.nn.Dropout(ffn_dropout),
                    torch.nn.Linear(ffn_hidden, attention_dim),
                )
                self._ffn_is_swiglu = False
            else:
                self.ffn_in = torch.nn.Linear(attention_dim, ffn_hidden * 2)
                self.ffn_out = torch.nn.Linear(ffn_hidden, attention_dim)
                self._ffn_is_swiglu = True

        self.query_out_proj = torch.nn.Linear(attention_dim, self._primary_channels)

        memory_out_dim = num_memory_tokens * 2 * attention_dim
        if memory_hidden_dim is not None:
            self.memory_mlp = torch.nn.Sequential(
                torch.nn.Linear(self._context_channels, memory_hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(memory_hidden_dim, memory_out_dim),
            )
        else:
            self.memory_mlp = torch.nn.Linear(self._context_channels, memory_out_dim)

        self.pre_fusion_norm = pre_fusion_norm
        self.pre_fusion_dropout = pre_fusion_dropout
        if pre_fusion_norm:
            self._pre_norm_layers = torch.nn.ModuleList(
                [torch.nn.LayerNorm(ch) for ch in path_output_channels]
            )

    def _run_path(
        self,
        path: torch.nn.ModuleList,
        context: ModelContext,
    ) -> Any:
        """Run a single encoder path and return its intermediate output."""
        out = path[0](context)
        for module in path[1:]:
            out = module(out, context)
        return out

    def _normalize_outputs(self, outputs: list[Any]) -> list[Any]:
        """Apply optional per-path LayerNorm + pre-fusion dropout."""

        def _ln(x: torch.Tensor, idx: int) -> torch.Tensor:
            ln = self._pre_norm_layers[idx]
            if x.dim() == 4:
                x = rearrange(x, "b c h w -> b h w c")
                x = ln(x)
                return rearrange(x, "b h w c -> b c h w")
            return ln(x)

        def _drop(x: torch.Tensor) -> torch.Tensor:
            if self.pre_fusion_dropout > 0.0 and self.training:
                return F.dropout(x, p=self.pre_fusion_dropout, training=True)
            return x

        result: list[Any] = []
        for i, out in enumerate(outputs):
            if isinstance(out, FeatureMaps):
                maps = out.feature_maps
                if self.pre_fusion_norm:
                    maps = [_ln(fm, i) for fm in maps]
                maps = [_drop(fm) for fm in maps]
                result.append(FeatureMaps(maps))
            elif isinstance(out, FeatureVector):
                vec = out.feature_vector
                if self.pre_fusion_norm:
                    vec = _ln(vec, i)
                vec = _drop(vec)
                result.append(FeatureVector(feature_vector=vec))
            else:
                result.append(out)
        return result

    def _apply_context_dropout(self, outputs: list[Any]) -> torch.Tensor | None:
        """Return per-sample boolean indicating context is dropped.

        When ``context_dropout_prob > 0`` and the model is training, each sample
        independently has its entire context dropped with the configured
        probability.  The returned mask is consumed in ``_cross_attend`` to zero
        the cross-attention residual for those samples.
        """
        if not self.training or self.context_dropout_prob <= 0.0:
            return None

        if isinstance(outputs[0], FeatureVector):
            batch_size = outputs[0].feature_vector.shape[0]
            device = outputs[0].feature_vector.device
        elif isinstance(outputs[0], FeatureMaps):
            batch_size = outputs[0].feature_maps[0].shape[0]
            device = outputs[0].feature_maps[0].device
        else:
            return None

        return torch.rand(batch_size, device=device) < self.context_dropout_prob

    @staticmethod
    def _validate_feature_map_scales(outputs: list[FeatureMaps]) -> int:
        """Validate that all paths produce the same number of feature map scales."""
        n_scales = len(outputs[0].feature_maps)
        for i, o in enumerate(outputs):
            if len(o.feature_maps) != n_scales:
                raise ValueError(
                    f"All paths must produce the same number of feature map scales. "
                    f"Path 0 has {n_scales} but path {i} has {len(o.feature_maps)}."
                )
        return n_scales

    def _validate_channels(self, outputs: list[Any]) -> None:
        """Validate runtime channel dimensions against path_output_channels."""
        for path_idx, out in enumerate(outputs):
            expected_channels = self.path_output_channels[path_idx]
            if isinstance(out, FeatureVector):
                actual_channels = out.feature_vector.shape[1]
                if actual_channels != expected_channels:
                    raise ValueError(
                        f"Path {path_idx} produced FeatureVector with {actual_channels} channels, "
                        f"expected {expected_channels}."
                    )
            elif isinstance(out, FeatureMaps):
                for scale_idx, fmap in enumerate(out.feature_maps):
                    actual_channels = fmap.shape[1]
                    if actual_channels != expected_channels:
                        raise ValueError(
                            f"Path {path_idx} produced FeatureMaps scale {scale_idx} with "
                            f"{actual_channels} channels, expected {expected_channels}."
                        )

    def _build_memory_kv(
        self, context_vec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build context-conditioned memory keys and values with shape [B, M, D]."""
        b = context_vec.shape[0]
        kv = self.memory_mlp(context_vec)
        kv = kv.view(b, self.num_memory_tokens, 2, self.attention_dim)
        return kv[:, :, 0, :], kv[:, :, 1, :]

    def _apply_ffn(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFN token mixing branch in attention space."""
        if self._ffn_is_swiglu:
            x12 = self.ffn_in(x)
            x_proj, x_gate = x12.chunk(2, dim=-1)
            x = x_proj * F.silu(x_gate)
            x = F.dropout(x, p=self.ffn_dropout, training=self.training)
            return self.ffn_out(x)
        return self.ffn(x)

    def _cross_attend(
        self,
        primary_tokens: torch.Tensor,
        context_vec: torch.Tensor,
        missing_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply cross-attn and optional post-fusion transformer-style blocks.

        Args:
            primary_tokens: tensor of shape [B, T, C_primary].
            context_vec: tensor of shape [B, C_context].
            missing_context: optional boolean tensor of shape [B] where ``True``
                indicates no context is available for that sample.
        """
        x = self.query_in_proj(primary_tokens)

        q_norm = self.cross_attn_norm(x)
        k, v = self._build_memory_kv(context_vec)
        k = self.key_norm(k)
        if self.normalize_memory_values:
            v = self.value_norm(v)
        attn_out, _ = self.cross_attn(q_norm, k, v, need_weights=False)
        attn_out = F.dropout(attn_out, p=self.residual_dropout, training=self.training)
        if missing_context is not None:
            attn_out = torch.where(
                missing_context.view(-1, 1, 1),
                torch.zeros_like(attn_out),
                attn_out,
            )
        x = x + self.cross_attn_alpha * attn_out

        if self.post_fusion_mode == "self_attn_ffn":
            x_norm = self.self_attn_norm(x)
            self_attn_out, _ = self.self_attn(
                x_norm, x_norm, x_norm, need_weights=False
            )
            x = x + F.dropout(
                self_attn_out, p=self.residual_dropout, training=self.training
            )

        if self.post_fusion_mode in ("ffn", "self_attn_ffn"):
            x_norm = self.ffn_norm(x)
            ffn_out = self._apply_ffn(x_norm)
            x = x + F.dropout(ffn_out, p=self.ffn_dropout, training=self.training)

        return self.query_out_proj(x)

    def _fuse_feature_vectors(
        self,
        outputs: list[FeatureVector],
        missing_context: torch.Tensor | None = None,
    ) -> FeatureVector:
        """Fuse FeatureVector outputs using cross-attention over memory tokens."""
        primary = outputs[0].feature_vector
        context = torch.cat([o.feature_vector for o in outputs[1:]], dim=1)
        fused = self._cross_attend(
            primary.unsqueeze(1), context, missing_context=missing_context
        ).squeeze(1)
        return FeatureVector(feature_vector=fused)

    def _fuse_feature_maps(
        self,
        outputs: list[FeatureMaps],
        missing_context: torch.Tensor | None = None,
    ) -> FeatureMaps:
        """Fuse FeatureMaps outputs via cross-attention over a global context memory bank."""
        self._validate_feature_map_scales(outputs)
        primary = outputs[0].feature_maps
        context_vectors: list[torch.Tensor] = []
        for out in outputs[1:]:
            per_scale = [fmap.mean(dim=[2, 3]) for fmap in out.feature_maps]
            context_vectors.append(torch.stack(per_scale, dim=0).mean(dim=0))
        context = torch.cat(context_vectors, dim=1)

        fused_maps: list[torch.Tensor] = []
        for fmap in primary:
            _, _, h, w = fmap.shape
            primary_tokens = rearrange(fmap, "b c h w -> b (h w) c")
            fused_tokens = self._cross_attend(
                primary_tokens, context, missing_context=missing_context
            )
            fused_maps.append(rearrange(fused_tokens, "b (h w) c -> b c h w", h=h, w=w))
        return FeatureMaps(feature_maps=fused_maps)

    def forward(self, context: ModelContext) -> FeatureMaps | FeatureVector:
        """Run all paths and fuse path outputs with context-memory cross-attention."""
        outputs = [self._run_path(path, context) for path in self.paths]
        outputs = self._normalize_outputs(outputs)

        first_type = type(outputs[0])
        for i, out in enumerate(outputs):
            if type(out) is not first_type:
                raise TypeError(
                    f"All encoder paths must produce the same intermediate type. "
                    f"Path 0 produced {first_type.__name__} but path {i} produced "
                    f"{type(out).__name__}."
                )

        self._validate_channels(outputs)

        missing_context = self._apply_context_dropout(outputs)
        context.context_dict[self.primary_context_key] = outputs[0]

        if isinstance(outputs[0], FeatureVector):
            return self._fuse_feature_vectors(outputs, missing_context=missing_context)  # type: ignore[arg-type]
        if isinstance(outputs[0], FeatureMaps):
            return self._fuse_feature_maps(outputs, missing_context=missing_context)  # type: ignore[arg-type]

        raise TypeError(
            f"CrossAttentionFusionExtractor only supports FeatureMaps and "
            f"FeatureVector outputs, got {first_type.__name__}."
        )
