"""Late-fusion feature extractor supporting concatenation, GLU-style gating, and modality mixing.

Refactor goals:
- Make the 3×2 matrix (fusion_mode × output_type) explicit and easy to scan.
- Keep all functionality identical: validations, lazy layers, gate dropout, gate bias init after
  materialization, FeatureMaps scale checks, GLU gating, mixing softmax per-path per-channel.
- Reduce “micro-helper sprawl” by grouping logic around *strategies* and *type adapters*.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import torch
import torch.nn.functional as F

from rslearn.train.model_context import ModelContext

from .component import (
    FeatureExtractor,
    FeatureMaps,
    FeatureVector,
    IntermediateComponent,
)

_GATE_ACTIVATIONS: dict[str, Callable[..., torch.Tensor]] = {
    "sigmoid": torch.sigmoid,
    "silu": F.silu,
    "tanh": torch.tanh,
}

_GATED_MODES = ("gated", "mixing")


# =============================================================================
# Output adapters (FeatureVector vs FeatureMaps)
# =============================================================================


class _Ops(Protocol):
    """Small adapter interface so fusion strategies can be written once."""

    kind: Literal["vector", "maps"]

    def concat(self, outputs: list[Any]) -> Any:
        """Concat outputs along channel dimension."""
        ...

    def apply_dropout_if_configured(self, x: torch.Tensor) -> torch.Tensor:
        """Dropout applied to gate inputs (not necessarily to path outputs)."""
        ...

    def validate_compatible(self, outputs: list[Any]) -> None:
        """Validate outputs are mutually compatible (e.g., same scales/spatial sizes)."""
        ...


@dataclass
class _VectorOps:
    parent: LateFusionFeatureExtractor
    kind: Literal["vector", "maps"] = "vector"

    def validate_compatible(self, outputs: list[FeatureVector]) -> None:
        # Nothing extra besides type consistency (handled in forward()).
        return

    def concat(self, outputs: list[FeatureVector]) -> FeatureVector:
        return FeatureVector(
            feature_vector=torch.cat([o.feature_vector for o in outputs], dim=1)
        )

    def apply_dropout_if_configured(self, x: torch.Tensor) -> torch.Tensor:
        return self.parent._apply_gate_dropout(x)


@dataclass
class _MapsOps:
    parent: LateFusionFeatureExtractor
    kind: Literal["vector", "maps"] = "maps"

    def validate_compatible(self, outputs: list[FeatureMaps]) -> None:
        # Ensure identical number of scales and identical H×W per scale across paths.
        LateFusionFeatureExtractor._validate_feature_map_scales(outputs)

    def concat(self, outputs: list[FeatureMaps]) -> FeatureMaps:
        n_scales = LateFusionFeatureExtractor._validate_feature_map_scales(outputs)
        fused_maps: list[torch.Tensor] = []
        for s in range(n_scales):
            fused_maps.append(torch.cat([o.feature_maps[s] for o in outputs], dim=1))
        return FeatureMaps(fused_maps)

    def apply_dropout_if_configured(self, x: torch.Tensor) -> torch.Tensor:
        return self.parent._apply_gate_dropout(x)


# =============================================================================
# Fusion strategies (concat / GLU gated / mixing)
# =============================================================================


class _FusionStrategy(Protocol):
    def fuse(self, ops: _Ops, outputs: list[Any]) -> Any: ...


@dataclass
class _ConcatFusion:
    def fuse(self, ops: _Ops, outputs: list[Any]) -> Any:
        ops.validate_compatible(outputs)
        return ops.concat(outputs)


@dataclass
class _GLUFusion:
    """GLU-style: z = value_proj(concat) ⊙ act(gate_proj(concat))."""

    parent: LateFusionFeatureExtractor

    def fuse(self, ops: _Ops, outputs: list[Any]) -> Any:
        ops.validate_compatible(outputs)
        concat_out = ops.concat(outputs)

        if ops.kind == "vector":
            vec = cast(FeatureVector, concat_out).feature_vector
            vec = ops.apply_dropout_if_configured(vec)

            value = self.parent.gated_value(vec)  # LazyLinear
            gate = self.parent._gate_act_fn(self.parent.gated_gate(vec))  # LazyLinear
            return FeatureVector(feature_vector=value * gate)

        # maps
        maps = cast(FeatureMaps, concat_out).feature_maps
        n_actual = len(maps)
        if n_actual != self.parent._gated_num_scales:
            raise ValueError(
                f"Gated fusion was configured for {self.parent._gated_num_scales} "
                f"scales but the encoder paths produced {n_actual}."
            )

        fused: list[torch.Tensor] = []
        for s, fmap in enumerate(maps):
            fmap = ops.apply_dropout_if_configured(fmap)
            value = self.parent.gated_value[s](fmap)  # LazyConv2d
            gate = self.parent._gate_act_fn(
                self.parent.gated_gate[s](fmap)
            )  # LazyConv2d
            fused.append(value * gate)
        return FeatureMaps(fused)


@dataclass
class _MixingFusion:
    """Per-channel per-path soft mixing.

    For each scale (or vector):
      1) project each path -> common C_out
      2) concat projected -> gate_logits -> reshape to B×N×C_out[×H×W]
      3) softmax over N (paths)
      4) weighted sum over paths
    """

    parent: LateFusionFeatureExtractor

    def fuse(self, ops: _Ops, outputs: list[Any]) -> Any:
        ops.validate_compatible(outputs)
        n_paths = self.parent._n_paths

        if ops.kind == "vector":
            out_ch = self.parent._mixing_out_channels[0]
            vecs = [cast(FeatureVector, o).feature_vector for o in outputs]

            # Project each path to common dim.
            projs = [
                self.parent.mixing_projs[p](vecs[p]) for p in range(n_paths)
            ]  # LazyLinear
            proj_stack = torch.stack(projs, dim=1)  # B × N × C_out

            # Gate on concatenated projected features.
            concat = torch.cat(projs, dim=1)  # B × (N·C_out)
            concat = ops.apply_dropout_if_configured(concat)

            b = concat.shape[0]
            gate_logits = self.parent.mixing_gate(concat).view(
                b, n_paths, out_ch
            )  # LazyLinear
            gate_weights = torch.softmax(gate_logits, dim=1)

            fused = (gate_weights * proj_stack).sum(dim=1)
            return FeatureVector(feature_vector=fused)

        # maps
        maps_outs = [cast(FeatureMaps, o) for o in outputs]
        n_scales = len(maps_outs[0].feature_maps)

        if n_scales != self.parent._gated_num_scales:
            raise ValueError(
                f"Mixing fusion was configured for {self.parent._gated_num_scales} "
                f"scales but the encoder paths produced {n_scales}."
            )

        fused_maps: list[torch.Tensor] = []
        for s in range(n_scales):
            out_ch = self.parent._mixing_out_channels[s]
            maps_at_scale = [o.feature_maps[s] for o in maps_outs]

            # Project each path to common dim.
            projs = [
                self.parent.mixing_projs[p][s](maps_at_scale[p])  # LazyConv2d
                for p in range(n_paths)
            ]
            proj_stack = torch.stack(projs, dim=1)  # B × N × C_out × H × W

            # Gate on concatenated projected features.
            concat = torch.cat(projs, dim=1)  # B × (N·C_out) × H × W
            concat = ops.apply_dropout_if_configured(concat)

            b, _, h, w = concat.shape
            gate_logits = self.parent.mixing_gate[s](concat).view(
                b, n_paths, out_ch, h, w
            )  # LazyConv2d
            gate_weights = torch.softmax(gate_logits, dim=1)

            fused_maps.append((gate_weights * proj_stack).sum(dim=1))

        return FeatureMaps(fused_maps)


# =============================================================================
# Main module
# =============================================================================


class LateFusionFeatureExtractor(FeatureExtractor):
    """Late-fusion feature extractor that runs parallel encoder paths and fuses their outputs.

    Supports:
      - fusion_mode="concat": concat along channels
      - fusion_mode="gated": GLU-style gating over concatenated features
      - fusion_mode="mixing": per-path soft weights (softmax) over projected features

    Output types:
      - FeatureVector
      - FeatureMaps (multi-scale)
    """

    def __init__(
        self,
        paths: list[list[FeatureExtractor | IntermediateComponent]],
        fusion_mode: str = "concat",
        gated_output_channels: int | list[int] | None = None,
        gate_activation: str = "sigmoid",
        gate_init_bias: float = 2.0,
        gate_dropout: float = 0.0,
    ):
        """Initialise the late-fusion feature extractor.

        Args:
            paths: List of encoder paths.  Each path is a list whose first element is a
                FeatureExtractor and remaining elements are IntermediateComponents.
            fusion_mode: One of ``"concat"``, ``"gated"``, or ``"mixing"``.
            gated_output_channels: Required for gated/mixing modes.  An int (vector) or
                list[int] (one per feature-map scale).
            gate_activation: Activation applied to the gate branch (``"sigmoid"``,
                ``"silu"``, or ``"tanh"``).
            gate_init_bias: Constant used to initialise gate biases after lazy
                materialisation.
            gate_dropout: Dropout probability applied to gate inputs during training.
        """
        super().__init__()

        # -- Path validation --------------------------------------------------
        if len(paths) < 1:
            raise ValueError("LateFusionFeatureExtractor requires at least one path")

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

        self.paths = torch.nn.ModuleList([torch.nn.ModuleList(path) for path in paths])

        # -- Fusion configuration ---------------------------------------------
        self.fusion_mode = fusion_mode
        self.gate_init_bias = gate_init_bias
        self.gate_dropout = gate_dropout

        if gate_activation not in _GATE_ACTIVATIONS:
            raise ValueError(
                f"Unknown gate_activation '{gate_activation}'. "
                f"Expected one of {sorted(_GATE_ACTIVATIONS)}."
            )
        self._gate_act_fn = _GATE_ACTIVATIONS[gate_activation]

        # Strategy chosen by fusion_mode (output adapter chosen at runtime).
        if fusion_mode == "concat":
            self._strategy: _FusionStrategy = _ConcatFusion()
        elif fusion_mode == "gated":
            self._strategy = _GLUFusion(self)
        elif fusion_mode == "mixing":
            self._strategy = _MixingFusion(self)
        else:
            raise ValueError(
                f"Unknown fusion_mode '{fusion_mode}'. Expected 'concat', 'gated', or 'mixing'."
            )

        # Lazily cached dispatch after first forward (so forward reads cleanly).
        self._cached_ops: _Ops | None = None

        # Gated / mixing layers ------------------------------------------------
        if fusion_mode in _GATED_MODES:
            if gated_output_channels is None:
                raise ValueError(
                    f"gated_output_channels is required when fusion_mode='{fusion_mode}'"
                )

            # Validate list contents early.
            if isinstance(gated_output_channels, list):
                if len(gated_output_channels) == 0:
                    raise ValueError("gated_output_channels must be a non-empty list")
                for idx, ch in enumerate(gated_output_channels):
                    if not isinstance(ch, int) or ch <= 0:
                        raise ValueError(
                            f"gated_output_channels[{idx}] must be a positive int, got {ch!r}"
                        )

            if isinstance(gated_output_channels, int):
                self._gated_is_spatial = False
            else:
                self._gated_is_spatial = True
                self._gated_num_scales = len(gated_output_channels)

            if fusion_mode == "gated":
                self._build_glu_layers(gated_output_channels)
            else:
                self._build_mixing_layers(gated_output_channels, len(paths))

            self._gated_output_validated = False
            self._gate_bias_initialized = False

    # ------------------------------------------------------------------
    # Layer builders (called from __init__)
    # ------------------------------------------------------------------

    def _build_glu_layers(self, gated_output_channels: int | list[int]) -> None:
        """Build value and gate projection layers for GLU-style fusion."""
        if isinstance(gated_output_channels, int):
            self.gated_value = torch.nn.LazyLinear(gated_output_channels)
            self.gated_gate = torch.nn.LazyLinear(gated_output_channels)
        else:
            self.gated_value = torch.nn.ModuleList(
                [torch.nn.LazyConv2d(c, kernel_size=1) for c in gated_output_channels]
            )
            self.gated_gate = torch.nn.ModuleList(
                [torch.nn.LazyConv2d(c, kernel_size=1) for c in gated_output_channels]
            )

    def _build_mixing_layers(
        self, gated_output_channels: int | list[int], n_paths: int
    ) -> None:
        """Build projection and gate layers for per-path mixing."""
        self._n_paths = n_paths

        if isinstance(gated_output_channels, int):
            self._mixing_out_channels = [gated_output_channels]
            self.mixing_projs = torch.nn.ModuleList(
                [torch.nn.LazyLinear(gated_output_channels) for _ in range(n_paths)]
            )
            self.mixing_gate = torch.nn.LazyLinear(n_paths * gated_output_channels)
        else:
            self._mixing_out_channels = list(gated_output_channels)
            # mixing_projs[path_idx][scale_idx]
            self.mixing_projs = torch.nn.ModuleList(
                [
                    torch.nn.ModuleList(
                        [
                            torch.nn.LazyConv2d(c, kernel_size=1)
                            for c in gated_output_channels
                        ]
                    )
                    for _ in range(n_paths)
                ]
            )
            # mixing_gate[scale_idx] -> outputs n_paths * out_ch channels
            self.mixing_gate = torch.nn.ModuleList(
                [
                    torch.nn.LazyConv2d(n_paths * c, kernel_size=1)
                    for c in gated_output_channels
                ]
            )

    # ------------------------------------------------------------------
    # Gate bias initialisation (after lazy materialisation)
    # ------------------------------------------------------------------

    def _maybe_init_gate_bias(self) -> None:
        """Set gate biases to gate_init_bias after lazy params materialise (runs once)."""
        if getattr(self, "_gate_bias_initialized", False) or self.gate_init_bias == 0.0:
            return
        if self.fusion_mode not in _GATED_MODES:
            return

        if self.fusion_mode == "gated":
            layers = (
                list(self.gated_gate) if self._gated_is_spatial else [self.gated_gate]
            )
        else:  # mixing
            layers = (
                list(self.mixing_gate) if self._gated_is_spatial else [self.mixing_gate]
            )

        for layer in layers:
            if hasattr(layer, "bias") and layer.bias is not None:
                torch.nn.init.constant_(layer.bias, self.gate_init_bias)

        self._gate_bias_initialized = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_path(self, path: torch.nn.ModuleList, context: ModelContext) -> Any:
        out = path[0](context)
        for module in path[1:]:
            out = module(out, context)
        return out

    def _apply_gate_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_dropout > 0.0 and self.training:
            return F.dropout(x, p=self.gate_dropout, training=True)
        return x

    # ------------------------------------------------------------------
    # Validation (type vs gated_output_channels)
    # ------------------------------------------------------------------

    def _validate_gated_output_type(self, output: Any) -> None:
        """Validate that encoder output type matches gated_output_channels shape (int vs list)."""
        if isinstance(output, FeatureMaps) and not self._gated_is_spatial:
            raise TypeError(
                "Encoder paths produce FeatureMaps but gated_output_channels was specified as an int. "
                "Expected a list[int] with one entry per scale."
            )
        if isinstance(output, FeatureVector) and self._gated_is_spatial:
            raise TypeError(
                "Encoder paths produce FeatureVector but gated_output_channels was specified as a list. "
                "Expected an int."
            )
        self._gated_output_validated = True

    @staticmethod
    def _validate_feature_map_scales(outputs: list[FeatureMaps]) -> int:
        """Validate all paths produce same #scales and matching spatial sizes. Return #scales."""
        n_scales = len(outputs[0].feature_maps)
        for i, o in enumerate(outputs):
            if len(o.feature_maps) != n_scales:
                raise ValueError(
                    f"All paths must produce the same number of feature map scales. "
                    f"Path 0 has {n_scales} but path {i} has {len(o.feature_maps)}."
                )

        for s in range(n_scales):
            h0 = outputs[0].feature_maps[s].shape[2]
            w0 = outputs[0].feature_maps[s].shape[3]
            for path_idx, o in enumerate(outputs[1:], start=1):
                m = o.feature_maps[s]
                if m.shape[2] != h0 or m.shape[3] != w0:
                    raise ValueError(
                        f"Spatial size mismatch at scale {s}: path 0 has ({h0}, {w0}) "
                        f"but path {path_idx} has ({m.shape[2]}, {m.shape[3]}). "
                        "All paths must produce feature maps with identical spatial dimensions at each scale."
                    )

        return n_scales

    # ------------------------------------------------------------------
    # Cached adapter selection (vector vs maps)
    # ------------------------------------------------------------------

    def _get_ops_for_outputs(self, first_output: Any) -> _Ops:
        """Pick and cache the correct adapter based on runtime output type."""
        if self._cached_ops is not None:
            return self._cached_ops

        ops: _Ops
        if isinstance(first_output, FeatureVector):
            ops = _VectorOps(self)
        elif isinstance(first_output, FeatureMaps):
            ops = _MapsOps(self)
        else:
            raise TypeError(
                "LateFusionFeatureExtractor only supports FeatureMaps and FeatureVector outputs, "
                f"got {type(first_output).__name__}."
            )
        self._cached_ops = ops
        return ops

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, context: ModelContext) -> FeatureMaps | FeatureVector:
        """Run all encoder paths and fuse their outputs."""
        # 1) Run all paths
        outputs = [self._run_path(path, context) for path in self.paths]

        # 2) All outputs must be the same type
        first_type = type(outputs[0])
        for i, o in enumerate(outputs):
            if type(o) is not first_type:
                raise TypeError(
                    "All encoder paths must produce the same intermediate type. "
                    f"Path 0 produced {first_type.__name__} but path {i} produced {type(o).__name__}."
                )

        # 3) If gated/mixing, validate output type matches gated_output_channels config (once)
        if self.fusion_mode in _GATED_MODES and not self._gated_output_validated:
            self._validate_gated_output_type(outputs[0])

        # 4) Fuse (strategy × adapter)
        ops = self._get_ops_for_outputs(outputs[0])
        result = self._strategy.fuse(ops, outputs)

        # 5) Gate bias init after lazy params materialize (once, post-first-forward through gate layers)
        if self.fusion_mode in _GATED_MODES and not self._gate_bias_initialized:
            self._maybe_init_gate_bias()

        return cast(FeatureMaps | FeatureVector, result)
