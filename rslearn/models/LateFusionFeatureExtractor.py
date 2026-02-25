"""Late-fusion feature extractor supporting concatenation, GLU-style gating, and modality mixing."""

from collections.abc import Callable
from typing import Any

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


class LateFusionFeatureExtractor(FeatureExtractor):
    """Late-fusion feature extractor that runs parallel encoder paths and fuses their outputs.

    Supports three fusion modes:

    - ``"concat"`` (default): simple concatenation along the channel dimension.
    - ``"gated"`` (GLU-style): learns to gate the fused representation via
      ``z = value_proj(concat) ⊙ act(gate_proj(concat))``, which can prevent
      one modality from washing out another.
    - ``"mixing"``: learns per-path soft weights via softmax and mixes
      independently projected modalities.
      This variant offers "choose modality A vs B" interpretability.
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
        """Create a new LateFusionFeatureExtractor.

        Args:
            paths: a list of encoder paths.  Each path is itself a list of
                modules: the first module must be a ``FeatureExtractor`` and any
                subsequent modules must be ``IntermediateComponent`` instances.
                Every path is applied to the same ``ModelContext`` and the
                results are fused.
            fusion_mode: ``"concat"`` for simple channel-wise concatenation,
                ``"gated"`` for GLU-style gated fusion, or ``"mixing"`` for
                modality-weighted mixing gated fusion.
            gated_output_channels: required when ``fusion_mode`` is
                ``"gated"`` or ``"mixing"``.  Specifies the output channel
                dimension of the gating / projection layers.
                    - If an ``int``, expects ``FeatureVector`` outputs
                    from the encoder paths.
                    - If a ``list[int]``, expects ``FeatureMaps`` outputs with one
                    entry per scale.
            gate_activation: activation function for the gate in ``"gated"``
                mode.  One of ``"sigmoid"`` (default), ``"silu"``, or
                ``"tanh"``.  Ignored in ``"concat"`` and ``"mixing"`` modes
                (``"mixing"`` uses softmax internally).
            gate_init_bias: initial bias value for gate projection layers.
                A positive value (e.g. 1.0–2.0) keeps gates more "open" at
                the start of training, which can improve stability.  Applied
                after lazy parameter materialisation on the first forward pass.
                Default ``0.0`` (no special init).
            gate_dropout: dropout probability applied to the concatenated
                input before it enters the gating / gate+value layers.  Can
                prevent the model from collapsing to "always ignore one
                modality".  Default ``0.0`` (no dropout).
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
                        f"Module {j} in path {i} must be an "
                        f"IntermediateComponent, "
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

        if fusion_mode in _GATED_MODES:
            if gated_output_channels is None:
                raise ValueError(
                    f"gated_output_channels is required when "
                    f"fusion_mode='{fusion_mode}'"
                )

            # Validate list contents.
            if isinstance(gated_output_channels, list):
                if len(gated_output_channels) == 0:
                    raise ValueError("gated_output_channels must be a non-empty list")
                for idx, ch in enumerate(gated_output_channels):
                    if not isinstance(ch, int) or ch <= 0:
                        raise ValueError(
                            f"gated_output_channels[{idx}] must be a positive "
                            f"int, got {ch!r}"
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

        elif fusion_mode != "concat":
            raise ValueError(
                f"Unknown fusion_mode '{fusion_mode}'. "
                f"Expected 'concat', 'gated', or 'mixing'."
            )

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
        """Build per-path projection and per-channel gate layers for mixing.

        The gate outputs ``n_paths * out_channels`` and is reshaped to
        ``B × n_paths × out_channels [× H × W]`` so that softmax is applied
        over the path dimension independently for each channel.
        """
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
            # mixing_gate[scale_idx] → outputs n_paths * out_ch channels
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
        """Set gate biases to ``gate_init_bias`` after lazy params materialise.

        Called once after the first forward pass through the gating layers.
        The very first batch will use the default (zero) bias; all subsequent
        batches use the configured bias.
        """
        if self._gate_bias_initialized or self.gate_init_bias == 0.0:
            return

        if self.fusion_mode == "gated":
            layers = (
                list(self.gated_gate) if self._gated_is_spatial else [self.gated_gate]
            )
        elif self.fusion_mode == "mixing":
            layers = (
                list(self.mixing_gate) if self._gated_is_spatial else [self.mixing_gate]
            )
        else:
            return

        for layer in layers:
            if hasattr(layer, "bias") and layer.bias is not None:
                torch.nn.init.constant_(layer.bias, self.gate_init_bias)

        self._gate_bias_initialized = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    def _apply_gate_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout to the concatenated input if configured."""
        if self.gate_dropout > 0.0 and self.training:
            return F.dropout(x, p=self.gate_dropout, training=True)
        return x

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_gated_output_type(self, output: Any) -> None:
        """Validate that the encoder output type matches ``gated_output_channels``.

        Called once on the first forward pass.

        Args:
            output: an actual output object from an encoder path.
        """
        if isinstance(output, FeatureMaps) and not self._gated_is_spatial:
            raise TypeError(
                "Encoder paths produce FeatureMaps but gated_output_channels "
                "was specified as an int. Expected a list[int] with one entry "
                "per scale."
            )
        if isinstance(output, FeatureVector) and self._gated_is_spatial:
            raise TypeError(
                "Encoder paths produce FeatureVector but gated_output_channels "
                "was specified as a list. Expected an int."
            )
        self._gated_output_validated = True

    @staticmethod
    def _validate_feature_map_scales(outputs: list[FeatureMaps]) -> int:
        """Validate that all paths produce the same number of scales and matching spatial sizes.

        Returns the number of scales.

        Raises:
            ValueError: if the number of scales or spatial dimensions differ
                across paths.
        """
        n_scales = len(outputs[0].feature_maps)
        for i, o in enumerate(outputs):
            if len(o.feature_maps) != n_scales:
                raise ValueError(
                    f"All paths must produce the same number of feature map "
                    f"scales.  Path 0 has {n_scales} but path {i} has "
                    f"{len(o.feature_maps)}."
                )

        for scale_idx in range(n_scales):
            h0 = outputs[0].feature_maps[scale_idx].shape[2]
            w0 = outputs[0].feature_maps[scale_idx].shape[3]
            for path_idx, o in enumerate(outputs[1:], start=1):
                m = o.feature_maps[scale_idx]
                if m.shape[2] != h0 or m.shape[3] != w0:
                    raise ValueError(
                        f"Spatial size mismatch at scale {scale_idx}: path 0 "
                        f"has ({h0}, {w0}) but path {path_idx} has "
                        f"({m.shape[2]}, {m.shape[3]}). All paths must "
                        f"produce feature maps with identical spatial "
                        f"dimensions at each scale."
                    )

        return n_scales

    # ------------------------------------------------------------------
    # Concatenation fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _concat_feature_maps(outputs: list[FeatureMaps]) -> FeatureMaps:
        """Concatenate a list of FeatureMaps along the channel dimension."""
        n_scales = LateFusionFeatureExtractor._validate_feature_map_scales(outputs)
        fused_maps: list[torch.Tensor] = []
        for scale_idx in range(n_scales):
            maps_at_scale = [o.feature_maps[scale_idx] for o in outputs]
            fused_maps.append(torch.cat(maps_at_scale, dim=1))
        return FeatureMaps(fused_maps)

    @staticmethod
    def _concat_feature_vectors(outputs: list[FeatureVector]) -> FeatureVector:
        """Concatenate a list of FeatureVectors along the channel dimension."""
        return FeatureVector(
            feature_vector=torch.cat([o.feature_vector for o in outputs], dim=1)
        )

    # ------------------------------------------------------------------
    # GLU-style gated fusion
    # ------------------------------------------------------------------

    def _gated_feature_maps(self, outputs: list[FeatureMaps]) -> FeatureMaps:
        """Apply GLU-style gated fusion to a list of FeatureMaps.

        Steps:
            1. Validate spatial sizes and concatenate all paths along the
               channel dimension.
            2. At each scale, apply two learned 1×1 convolutions — one for the
               *value* and one for the *gate* — then fuse via element-wise
               product: ``z = value_proj(x) ⊙ act(gate_proj(x))``.
        """
        concat_result = self._concat_feature_maps(outputs)

        n_actual = len(concat_result.feature_maps)
        if n_actual != self._gated_num_scales:
            raise ValueError(
                f"Gated fusion was configured for {self._gated_num_scales} "
                f"scales but the encoder paths produced {n_actual}."
            )

        fused_maps: list[torch.Tensor] = []
        for i, fmap in enumerate(concat_result.feature_maps):
            fmap = self._apply_gate_dropout(fmap)
            value = self.gated_value[i](fmap)
            gate = self._gate_act_fn(self.gated_gate[i](fmap))
            fused_maps.append(value * gate)

        return FeatureMaps(fused_maps)

    def _gated_feature_vectors(self, outputs: list[FeatureVector]) -> FeatureVector:
        """Apply GLU-style gated fusion to a list of FeatureVectors.

        Concatenates all path outputs and applies:
        ``z = value_proj(x) ⊙ act(gate_proj(x))``.
        """
        concat_result = self._concat_feature_vectors(outputs)
        vec = self._apply_gate_dropout(concat_result.feature_vector)

        value = self.gated_value(vec)
        gate = self._gate_act_fn(self.gated_gate(vec))

        return FeatureVector(feature_vector=value * gate)

    # ------------------------------------------------------------------
    # Modality-weighted mixing fusion
    # ------------------------------------------------------------------

    def _mixing_feature_maps(self, outputs: list[FeatureMaps]) -> FeatureMaps:
        """Apply per-channel modality mixing to a list of FeatureMaps.

        At each scale:
            1. Gate conv on concat → ``B × (N·C_out) × H × W``, reshaped to
               ``B × N × C_out × H × W``, then softmax over paths (dim 1).
            2. Project each path independently via a learned 1×1 conv.
            3. Weighted sum: ``z = Σ_i weights_i ⊙ proj_i(x_i)``.
        """
        n_scales = self._validate_feature_map_scales(outputs)

        if n_scales != self._gated_num_scales:
            raise ValueError(
                f"Mixing fusion was configured for {self._gated_num_scales} "
                f"scales but the encoder paths produced {n_scales}."
            )

        n_paths = self._n_paths
        fused_maps: list[torch.Tensor] = []

        for s in range(n_scales):
            out_ch = self._mixing_out_channels[s]
            maps_at_scale = [o.feature_maps[s] for o in outputs]
            concat = torch.cat(maps_at_scale, dim=1)
            concat = self._apply_gate_dropout(concat)

            b, _, h, w = concat.shape
            # B × (N·C_out) × H × W → B × N × C_out × H × W
            gate_logits = self.mixing_gate[s](concat)
            gate_logits = gate_logits.view(b, n_paths, out_ch, h, w)
            gate_weights = torch.softmax(gate_logits, dim=1)

            # B × N × C_out × H × W
            proj_stack = torch.stack(
                [self.mixing_projs[p][s](maps_at_scale[p]) for p in range(n_paths)],
                dim=1,
            )
            fused_maps.append((gate_weights * proj_stack).sum(dim=1))

        return FeatureMaps(fused_maps)

    def _mixing_feature_vectors(self, outputs: list[FeatureVector]) -> FeatureVector:
        """Apply per-channel modality mixing to a list of FeatureVectors.

        Gate linear on concat → ``B × (N·C_out)``, reshaped to
        ``B × N × C_out``, then softmax over paths (dim 1).
        """
        n_paths = self._n_paths
        out_ch = self._mixing_out_channels[0]
        vecs = [o.feature_vector for o in outputs]
        concat = torch.cat(vecs, dim=1)
        concat = self._apply_gate_dropout(concat)

        b = concat.shape[0]
        # B × (N·C_out) → B × N × C_out
        gate_logits = self.mixing_gate(concat).view(b, n_paths, out_ch)
        gate_weights = torch.softmax(gate_logits, dim=1)

        # B × N × C_out
        proj_stack = torch.stack(
            [self.mixing_projs[p](vecs[p]) for p in range(n_paths)], dim=1
        )
        fused = (gate_weights * proj_stack).sum(dim=1)

        return FeatureVector(feature_vector=fused)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, context: ModelContext) -> FeatureMaps | FeatureVector:
        """Run all encoder paths and fuse their outputs.

        Args:
            context: the model context (shared across all paths).

        Returns:
            a ``FeatureMaps`` or ``FeatureVector`` obtained by fusing the
            outputs of every path.

        Raises:
            TypeError: if the paths produce different intermediate types or an
                unsupported type.
        """
        outputs = [self._run_path(path, context) for path in self.paths]

        # All outputs must be the same type.
        first_type = type(outputs[0])
        for i, o in enumerate(outputs):
            if type(o) is not first_type:
                raise TypeError(
                    f"All encoder paths must produce the same intermediate "
                    f"type.  Path 0 produced {first_type.__name__} but path "
                    f"{i} produced {type(o).__name__}."
                )

        if self.fusion_mode in _GATED_MODES and not self._gated_output_validated:
            self._validate_gated_output_type(outputs[0])

        if isinstance(outputs[0], FeatureMaps):
            result: FeatureMaps | FeatureVector = self._fuse_feature_maps(outputs)  # type: ignore[arg-type]
        elif isinstance(outputs[0], FeatureVector):
            result = self._fuse_feature_vectors(outputs)  # type: ignore[arg-type]
        else:
            raise TypeError(
                f"LateFusionFeatureExtractor only supports FeatureMaps and "
                f"FeatureVector outputs, got {first_type.__name__}."
            )

        if self.fusion_mode in _GATED_MODES and not self._gate_bias_initialized:
            self._maybe_init_gate_bias()

        return result

    def _fuse_feature_maps(self, outputs: list[FeatureMaps]) -> FeatureMaps:
        """Dispatch to the correct FeatureMaps fusion method."""
        if self.fusion_mode == "concat":
            return self._concat_feature_maps(outputs)
        if self.fusion_mode == "gated":
            return self._gated_feature_maps(outputs)
        return self._mixing_feature_maps(outputs)

    def _fuse_feature_vectors(self, outputs: list[FeatureVector]) -> FeatureVector:
        """Dispatch to the correct FeatureVector fusion method."""
        if self.fusion_mode == "concat":
            return self._concat_feature_vectors(outputs)
        if self.fusion_mode == "gated":
            return self._gated_feature_vectors(outputs)
        return self._mixing_feature_vectors(outputs)
