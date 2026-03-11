"""Late-fusion feature extractor supporting concatenation, GLU-style gating, modality mixing, and FiLM conditioning."""

from collections.abc import Callable
from typing import Any

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

_GATE_ACTIVATIONS: dict[str, Callable[..., torch.Tensor]] = {
    "sigmoid": torch.sigmoid,
    "silu": F.silu,
    "tanh": torch.tanh,
}

_GATED_MODES = ("gated", "mixing")


class LateFusionFeatureExtractor(FeatureExtractor):
    """Late-fusion feature extractor that runs parallel encoder paths and fuses their outputs.

    Supports four fusion modes:

    - ``"concat"`` (default): simple concatenation along the channel dimension.
    - ``"gated"`` (GLU-style): learns to gate the fused representation via
      ``z = value_proj(concat) ⊙ act(gate_proj(concat))``, which can prevent
      one modality from washing out another.
    - ``"mixing"``: learns per-path soft weights via softmax and mixes
      independently projected modalities.
      This variant offers "choose modality A vs B" interpretability.
    - ``"film"`` (Feature-wise Linear Modulation): path 0 is the *primary*
      (spatial) modality; all other paths are *context*.  Context features are
      global-average-pooled to a vector and mapped through learned MLPs to
      produce per-channel scale (γ) and shift (β) parameters that modulate the
      primary features: ``output = (1 + γ) ⊙ primary + β``.  Initialised so that
      γ ≈ 0 and β ≈ 0 (identity at start), meaning training begins from the
      primary-only baseline.
    """

    def __init__(
        self,
        paths: list[list[FeatureExtractor | IntermediateComponent]],
        fusion_mode: str = "concat",
        gated_output_channels: int | list[int] | None = None,
        gate_activation: str = "sigmoid",
        gate_init_bias: float = 2.0,
        gate_dropout: float = 0.0,
        path_output_channels: list[int] | None = None,
        mixing_gate_hidden_dim: int | None = None,
        film_hidden_dim: int | None = None,
        pre_fusion_norm: bool = False,
        pre_fusion_dropout: float = 0.0,
        path_dropout_prob: float = 0.0,
        path_dropout_rescale: bool = False,
        path0_context_key: str = "path0_intermediate",
    ):
        """Create a new LateFusionFeatureExtractor.

        Args:
            paths: a list of encoder paths.  Each path is itself a list of
                modules: the first module must be a ``FeatureExtractor`` and any
                subsequent modules must be ``IntermediateComponent`` instances.
                Every path is applied to the same ``ModelContext`` and the
                results are fused.
            fusion_mode: ``"concat"`` for simple channel-wise concatenation,
                ``"gated"`` for GLU-style gated fusion, ``"mixing"`` for
                modality-weighted mixing gated fusion, or ``"film"`` for
                Feature-wise Linear Modulation conditioning.
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
                the start of training, which can improve stability.
                Default ``0.0`` (no special init).
            gate_dropout: dropout probability applied to the concatenated
                input before it enters the gating / gate+value layers.  Can
                prevent the model from collapsing to "always ignore one
                modality".  Default ``0.0`` (no dropout).  Also used as
                dropout on the FiLM context vector when ``fusion_mode="film"``.
            path_output_channels: required when ``fusion_mode`` is ``"gated"``,
                ``"mixing"``, or ``"film"``, or when ``pre_fusion_norm`` is
                ``True``.  A list with one entry per path giving the output
                channel dimension of that path.  For multi-scale
                ``FeatureMaps`` the same channel count is assumed at every
                scale.  Example: ``[768, 128]`` for a 768-channel S2 path and
                a 128-channel ERA5 path.
            mixing_gate_hidden_dim: hidden dimension for the mixing gate
                bottleneck MLP (and 1x1-conv bottleneck for spatial mixing).
                If ``None``, a small value is chosen automatically from the
                input dimension. Only relevant when ``fusion_mode="mixing"``.
            film_hidden_dim: optional hidden dimension for the FiLM gamma/beta
                MLPs.  If ``None`` (default), a single linear layer is used.
                Only relevant when ``fusion_mode="film"``.
            pre_fusion_norm: if ``True``, apply a learned ``LayerNorm`` to
                each path's output before fusion (requires
                ``path_output_channels``).  Default ``False``.
            pre_fusion_dropout: dropout probability applied to each path output
                right before fusion (after optional ``pre_fusion_norm``).
                Applied to both ``FeatureMaps`` and ``FeatureVector`` outputs.
                Default ``0.0`` (no dropout).
            path_dropout_prob: branch-level dropout probability applied
                independently to each context path (paths 1+) per sample during
                training. Path 0 is never dropped. Default ``0.0`` (disabled).
            path_dropout_rescale: if ``True``, surviving context branches are
                scaled by ``1 / (1 - path_dropout_prob)`` during training to
                preserve expectation. Default ``False``.
            path0_context_key: key used to store path0 intermediate output in
                ``context.context_dict`` for optional downstream auxiliary
                supervision.
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
        self.mixing_gate_hidden_dim = mixing_gate_hidden_dim

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
            if path_output_channels is None:
                raise ValueError(
                    f"path_output_channels is required when fusion_mode='{fusion_mode}'"
                )
            if len(path_output_channels) != len(paths):
                raise ValueError(
                    f"path_output_channels must have one entry per path "
                    f"({len(paths)}), got {len(path_output_channels)}"
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

            total_in_channels = sum(path_output_channels)

            if isinstance(gated_output_channels, int):
                self._gated_is_spatial = False
            else:
                self._gated_is_spatial = True
                self._gated_num_scales = len(gated_output_channels)

            if fusion_mode == "gated":
                self._build_glu_layers(gated_output_channels, total_in_channels)
            else:
                self._build_mixing_layers(
                    gated_output_channels,
                    len(paths),
                    path_output_channels,
                    self.mixing_gate_hidden_dim,
                )

            self._gated_output_validated = False
            self._gate_bias_initialized = False
            self._maybe_init_gate_bias()

        elif fusion_mode == "film":
            if path_output_channels is None:
                raise ValueError(
                    f"path_output_channels is required when fusion_mode='{fusion_mode}'"
                )
            if len(path_output_channels) != len(paths):
                raise ValueError(
                    f"path_output_channels must have one entry per path "
                    f"({len(paths)}), got {len(path_output_channels)}"
                )
            self._build_film_layers(path_output_channels, film_hidden_dim)

        elif fusion_mode != "concat":
            raise ValueError(
                f"Unknown fusion_mode '{fusion_mode}'. "
                f"Expected 'concat', 'gated', 'mixing', or 'film'."
            )

        # -- Pre-fusion normalisation -----------------------------------------
        self.pre_fusion_norm = pre_fusion_norm
        if not 0.0 <= pre_fusion_dropout < 1.0:
            raise ValueError(
                f"pre_fusion_dropout must be in [0, 1), got {pre_fusion_dropout!r}"
            )
        self.pre_fusion_dropout = pre_fusion_dropout
        if not 0.0 <= path_dropout_prob < 1.0:
            raise ValueError(
                f"path_dropout_prob must be in [0, 1), got {path_dropout_prob!r}"
            )
        self.path_dropout_prob = path_dropout_prob
        self.path_dropout_rescale = path_dropout_rescale
        self.path0_context_key = path0_context_key
        if pre_fusion_norm:
            if path_output_channels is None:
                raise ValueError(
                    "path_output_channels required when pre_fusion_norm=True"
                )
            self._pre_norm_layers = torch.nn.ModuleList(
                [torch.nn.LayerNorm(ch) for ch in path_output_channels]
            )

    # ------------------------------------------------------------------
    # Layer builders (called from __init__)
    # ------------------------------------------------------------------

    def _build_glu_layers(
        self, gated_output_channels: int | list[int], total_in_channels: int
    ) -> None:
        """Build value and gate projection layers for GLU-style fusion.

        Args:
            gated_output_channels: output channel dimension(s).
            total_in_channels: (i.e. the concatenated channel count).
        """
        if isinstance(gated_output_channels, int):
            self.gated_value = torch.nn.Linear(total_in_channels, gated_output_channels)
            self.gated_gate = torch.nn.Linear(total_in_channels, gated_output_channels)
        else:
            self.gated_value = torch.nn.ModuleList(
                [
                    torch.nn.Conv2d(total_in_channels, c, kernel_size=1)
                    for c in gated_output_channels
                ]
            )
            self.gated_gate = torch.nn.ModuleList(
                [
                    torch.nn.Conv2d(total_in_channels, c, kernel_size=1)
                    for c in gated_output_channels
                ]
            )

    def _build_mixing_layers(
        self,
        gated_output_channels: int | list[int],
        n_paths: int,
        path_output_channels: list[int],
        mixing_gate_hidden_dim: int | None,
    ) -> None:
        """Build per-path projection and per-channel gate layers for mixing.

        The gate outputs ``n_paths * out_channels`` and is reshaped to
        ``B × n_paths × out_channels [× H × W]`` so that softmax is applied
        over the path dimension independently for each channel.

        Args:
            gated_output_channels: output channel dimension(s).
            n_paths: number of encoder paths.
            path_output_channels: output channel dimension of each path.
            mixing_gate_hidden_dim: optional hidden dimension for the mixing
                gate bottleneck. If ``None``, an automatic size is used.
        """
        self._n_paths = n_paths

        if isinstance(gated_output_channels, int):
            self._mixing_out_channels = [gated_output_channels]
            self.mixing_projs = torch.nn.ModuleList(
                [
                    torch.nn.Linear(path_output_channels[p], gated_output_channels)
                    for p in range(n_paths)
                ]
            )
            in_dim = n_paths * gated_output_channels
            hidden_dim = self._resolve_mixing_hidden_dim(in_dim, mixing_gate_hidden_dim)
            self.mixing_gate = torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_dim, in_dim),
            )
        else:
            self._mixing_out_channels = list(gated_output_channels)
            # mixing_projs[path_idx][scale_idx]
            self.mixing_projs = torch.nn.ModuleList(
                [
                    torch.nn.ModuleList(
                        [
                            torch.nn.Conv2d(path_output_channels[p], c, kernel_size=1)
                            for c in gated_output_channels
                        ]
                    )
                    for p in range(n_paths)
                ]
            )
            # mixing_gate[scale_idx] → outputs n_paths * out_ch channels
            self.mixing_gate = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Conv2d(
                            n_paths * c,
                            self._resolve_mixing_hidden_dim(
                                n_paths * c, mixing_gate_hidden_dim
                            ),
                            kernel_size=1,
                        ),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(
                            self._resolve_mixing_hidden_dim(
                                n_paths * c, mixing_gate_hidden_dim
                            ),
                            n_paths * c,
                            kernel_size=1,
                        ),
                    )
                    for c in gated_output_channels
                ]
            )

    @staticmethod
    def _resolve_mixing_hidden_dim(in_dim: int, configured: int | None) -> int:
        """Return a small hidden size for mixing-gate bottlenecks."""
        if configured is not None:
            if configured <= 0:
                raise ValueError(
                    f"mixing_gate_hidden_dim must be a positive int, got {configured!r}"
                )
            return min(configured, in_dim)
        # Default: conservative bottleneck that scales with input size.
        return max(1, min(in_dim, max(8, min(128, in_dim // 4))))

    def _build_film_layers(
        self,
        path_output_channels: list[int],
        film_hidden_dim: int | None,
    ) -> None:
        """Build FiLM gamma (scale) and beta (shift) projection layers.

        Path 0 is the *primary* modality whose features are modulated.
        Paths 1+ are *context*; their features are global-average-pooled to
        vectors and concatenated to form the FiLM conditioning input.

        The output dimensionality equals ``path_output_channels[0]`` (the
        primary path's channel count).

        Args:
            path_output_channels: output channel dimension of each path.
            film_hidden_dim: if not None, use a two-layer MLP with this hidden
                dimension; otherwise use a single Linear layer.
        """
        self._film_primary_channels = path_output_channels[0]
        self._film_context_channels = sum(path_output_channels[1:])
        self._film_num_context_paths = max(0, len(path_output_channels) - 1)
        self._film_conditioning_dim = (
            self._film_context_channels + self._film_num_context_paths
        )

        if film_hidden_dim is not None and film_hidden_dim > 0:
            self.film_gamma = torch.nn.Sequential(
                torch.nn.Linear(self._film_conditioning_dim, film_hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(film_hidden_dim, self._film_primary_channels),
            )
            self.film_beta = torch.nn.Sequential(
                torch.nn.Linear(self._film_conditioning_dim, film_hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(film_hidden_dim, self._film_primary_channels),
            )
        else:
            self.film_gamma = torch.nn.Linear(
                self._film_conditioning_dim, self._film_primary_channels
            )
            self.film_beta = torch.nn.Linear(
                self._film_conditioning_dim, self._film_primary_channels
            )

        self._init_film_weights()

    def _init_film_weights(self) -> None:
        """Initialise FiLM layers so that γ ≈ 0 and β ≈ 0 at the start."""

        def _get_last_linear(module: torch.nn.Module) -> torch.nn.Linear:
            """Return the last Linear layer in a module (or the module itself)."""
            if isinstance(module, torch.nn.Linear):
                return module
            # Sequential: walk in reverse to find the last Linear
            for child in reversed(list(module.children())):
                if isinstance(child, torch.nn.Linear):
                    return child
            raise RuntimeError("Could not find a Linear layer in the FiLM module")

        # Gamma: output ≈ 0  →  zero weights, bias = 0
        gamma_last = _get_last_linear(self.film_gamma)
        torch.nn.init.zeros_(gamma_last.weight)
        if gamma_last.bias is not None:
            torch.nn.init.zeros_(gamma_last.bias)

        # Beta: output ≈ 0  →  zero weights, bias = 0
        beta_last = _get_last_linear(self.film_beta)
        torch.nn.init.zeros_(beta_last.weight)
        if beta_last.bias is not None:
            torch.nn.init.zeros_(beta_last.bias)

    # ------------------------------------------------------------------
    # Gate bias initialisation (after lazy materialisation)
    # ------------------------------------------------------------------

    def _maybe_init_gate_bias(self) -> None:
        """Set gate biases to ``gate_init_bias`` if not already done.

        Called from ``__init__`` after the gating layers are built.
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
            layer_to_init = layer
            if isinstance(layer, torch.nn.Sequential):
                # For bottleneck gates, biasing the final projection preserves
                # the original "open gate" initialization intent.
                for child in reversed(list(layer.children())):
                    if hasattr(child, "bias"):
                        layer_to_init = child
                        break
            if hasattr(layer_to_init, "bias") and layer_to_init.bias is not None:
                torch.nn.init.constant_(layer_to_init.bias, self.gate_init_bias)

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

    def _apply_context_path_dropout(
        self, outputs: list[Any]
    ) -> tuple[list[Any], list[torch.Tensor] | None]:
        """Apply branch-level dropout to paths 1+ and return per-path keep masks.

        Returns:
            A tuple ``(outputs_after_dropout, masks)`` where ``masks`` is either
            ``None`` (dropout inactive) or one boolean tensor per context path
            (paths 1+) with shape ``[B]`` indicating which samples kept that path.
        """
        if not self.training or self.path_dropout_prob <= 0.0 or len(outputs) <= 1:
            return outputs, None

        keep_prob = 1.0 - self.path_dropout_prob
        scale = (
            1.0 / keep_prob if (self.path_dropout_rescale and keep_prob > 0.0) else 1.0
        )
        out_after_drop: list[Any] = [outputs[0]]
        path_masks: list[torch.Tensor] = []
        for out in outputs[1:]:
            if isinstance(out, FeatureVector):
                vec = out.feature_vector
                keep_mask = (
                    torch.rand((vec.shape[0], 1), device=vec.device)
                    >= self.path_dropout_prob
                ).to(vec.dtype)
                out_after_drop.append(
                    FeatureVector(feature_vector=vec * keep_mask * scale)
                )
                path_masks.append(keep_mask[:, 0].to(torch.bool))
            elif isinstance(out, FeatureMaps):
                ref = out.feature_maps[0]
                keep_mask = (
                    torch.rand((ref.shape[0], 1, 1, 1), device=ref.device)
                    >= self.path_dropout_prob
                ).to(ref.dtype)
                out_after_drop.append(
                    FeatureMaps(
                        feature_maps=[
                            fmap * keep_mask * scale for fmap in out.feature_maps
                        ]
                    )
                )
                path_masks.append(keep_mask[:, 0, 0, 0].to(torch.bool))
            else:
                out_after_drop.append(out)
        return out_after_drop, path_masks

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

    def _mixing_feature_maps(
        self,
        outputs: list[FeatureMaps],
        path_dropout_masks: list[torch.Tensor] | None = None,
    ) -> FeatureMaps:
        """Apply per-channel modality mixing to a list of FeatureMaps.

        At each scale:
            1. Project each path to the common output dimension ``C_out``
               via a learned 1×1 conv.
            2. Concatenate projected maps and feed through the gate conv:
               ``B × (N·C_out) × H × W`` → reshaped to
               ``B × N × C_out × H × W``, then softmax over paths (dim 1).
            3. Weighted sum: ``z = Σ_i weights_i ⊙ proj_i``.
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

            # Project each path to the common output dimension.
            projs = [self.mixing_projs[p][s](maps_at_scale[p]) for p in range(n_paths)]
            if path_dropout_masks is not None:
                for p in range(1, n_paths):
                    m = path_dropout_masks[p - 1].to(
                        device=projs[p].device, dtype=projs[p].dtype
                    )
                    projs[p] = projs[p] * m[:, None, None, None]
            proj_stack = torch.stack(projs, dim=1)  # B × N × C_out × H × W

            # Gate operates on projected (uniform-dim) features.
            concat = torch.cat(projs, dim=1)  # B × (N·C_out) × H × W
            concat = self._apply_gate_dropout(concat)

            b, _, h, w = concat.shape
            # B × (N·C_out) × H × W → B × N × C_out × H × W
            gate_logits = self.mixing_gate[s](concat)
            gate_logits = gate_logits.view(b, n_paths, out_ch, h, w)
            path_presence = self._path_presence_mask(
                batch_size=b,
                n_paths=n_paths,
                device=gate_logits.device,
                path_dropout_masks=path_dropout_masks,
            )
            gate_logits = gate_logits.masked_fill(
                ~path_presence.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), -1e9
            )
            gate_weights = torch.softmax(gate_logits, dim=1)

            fused_maps.append((gate_weights * proj_stack).sum(dim=1))

        return FeatureMaps(fused_maps)

    def _mixing_feature_vectors(
        self,
        outputs: list[FeatureVector],
        path_dropout_masks: list[torch.Tensor] | None = None,
    ) -> FeatureVector:
        """Apply per-channel modality mixing to a list of FeatureVectors.

        Steps:
            1. Project each path to the common output dimension ``C_out``.
            2. Concatenate projected vectors and feed through the gate linear
               layer: ``B × (N·C_out)`` → reshaped to ``B × N × C_out``,
               then softmax over paths (dim 1).
            3. Weighted sum: ``z = Σ_i weights_i ⊙ proj_i``.
        """
        n_paths = self._n_paths
        out_ch = self._mixing_out_channels[0]
        vecs = [o.feature_vector for o in outputs]

        # Project each path to the common output dimension.
        projs = [self.mixing_projs[p](vecs[p]) for p in range(n_paths)]
        if path_dropout_masks is not None:
            for p in range(1, n_paths):
                m = path_dropout_masks[p - 1].to(
                    device=projs[p].device, dtype=projs[p].dtype
                )
                projs[p] = projs[p] * m.unsqueeze(1)
        proj_stack = torch.stack(projs, dim=1)  # B × N × C_out

        # Gate operates on projected (uniform-dim) features.
        concat = torch.cat(projs, dim=1)  # B × (N·C_out)
        concat = self._apply_gate_dropout(concat)

        b = concat.shape[0]
        # B × (N·C_out) → B × N × C_out
        gate_logits = self.mixing_gate(concat).view(b, n_paths, out_ch)
        path_presence = self._path_presence_mask(
            batch_size=b,
            n_paths=n_paths,
            device=gate_logits.device,
            path_dropout_masks=path_dropout_masks,
        )
        gate_logits = gate_logits.masked_fill(~path_presence.unsqueeze(-1), -1e9)
        gate_weights = torch.softmax(gate_logits, dim=1)

        fused = (gate_weights * proj_stack).sum(dim=1)

        return FeatureVector(feature_vector=fused)

    # ------------------------------------------------------------------
    # FiLM (Feature-wise Linear Modulation) fusion
    # ------------------------------------------------------------------

    def _extract_context_vector(self, outputs: list[FeatureMaps]) -> torch.Tensor:
        """Global-average-pool all context paths (1+) and concatenate to a vector.

        Args:
            outputs: list of FeatureMaps from all paths.

        Returns:
            context vector of shape ``[B, sum(context_channels)]``.
        """
        context_vecs: list[torch.Tensor] = []
        for o in outputs[1:]:
            for fmap in o.feature_maps:
                # [B, C, H, W] → [B, C]
                context_vecs.append(fmap.mean(dim=[2, 3]))
        return torch.cat(context_vecs, dim=1)

    def _film_feature_maps(
        self,
        outputs: list[FeatureMaps],
        path_dropout_masks: list[torch.Tensor] | None = None,
    ) -> FeatureMaps:
        """Apply FiLM conditioning to the primary path's FeatureMaps.

        Path 0 is modulated by context derived from paths 1+:
            ``output = (1 + γ(context)) ⊙ primary + β(context)``

        where γ and β are per-channel scalars broadcast over spatial dims.
        """
        self._validate_feature_map_scales(outputs)
        primary = outputs[0]

        context = self._extract_context_vector(outputs)
        context = self._append_context_presence_flags(
            context=context,
            path_dropout_masks=path_dropout_masks,
        )
        context = self._apply_gate_dropout(context)

        gamma = self.film_gamma(context)  # [B, C_primary]
        beta = self.film_beta(context)  # [B, C_primary]

        fused_maps: list[torch.Tensor] = []
        for fmap in primary.feature_maps:
            # fmap: [B, C, H, W]; gamma/beta: [B, C] → broadcast to [B, C, 1, 1]
            modulated = (1 + gamma).unsqueeze(-1).unsqueeze(-1) * fmap + beta.unsqueeze(
                -1
            ).unsqueeze(-1)
            fused_maps.append(modulated)

        return FeatureMaps(fused_maps)

    def _film_feature_vectors(
        self,
        outputs: list[FeatureVector],
        path_dropout_masks: list[torch.Tensor] | None = None,
    ) -> FeatureVector:
        """Apply FiLM conditioning to the primary path's FeatureVector.

        Path 0 is modulated by context derived from paths 1+:
            ``output = (1 + γ(context)) ⊙ primary + β(context)``
        """
        primary_vec = outputs[0].feature_vector
        context_vecs = [o.feature_vector for o in outputs[1:]]
        context = torch.cat(context_vecs, dim=1)  # [B, sum(context_channels)]
        context = self._append_context_presence_flags(
            context=context,
            path_dropout_masks=path_dropout_masks,
        )
        context = self._apply_gate_dropout(context)

        gamma = self.film_gamma(context)  # [B, C_primary]
        beta = self.film_beta(context)  # [B, C_primary]

        return FeatureVector(feature_vector=(1 + gamma) * primary_vec + beta)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _path_presence_mask(
        self,
        batch_size: int,
        n_paths: int,
        device: torch.device,
        path_dropout_masks: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        """Build a boolean mask [B, N] indicating which paths are present."""
        path_presence = torch.ones(
            (batch_size, n_paths), device=device, dtype=torch.bool
        )
        if path_dropout_masks is None:
            return path_presence
        for context_path_idx, mask in enumerate(path_dropout_masks, start=1):
            path_presence[:, context_path_idx] = mask.to(
                device=device, dtype=torch.bool
            )
        return path_presence

    def _append_context_presence_flags(
        self,
        context: torch.Tensor,
        path_dropout_masks: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        """Append one 0/1 presence flag per context path to the FiLM context vector."""
        b = context.shape[0]
        n_context = len(self.paths) - 1
        if n_context == 0:
            return context
        if path_dropout_masks is None:
            flags = torch.ones(
                (b, n_context), device=context.device, dtype=context.dtype
            )
        else:
            flags = torch.stack(
                [
                    m.to(device=context.device, dtype=context.dtype)
                    for m in path_dropout_masks
                ],
                dim=1,
            )
        return torch.cat([context, flags], dim=1)

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

        # Normalise each path's output before fusion (no-op when disabled).
        outputs = self._normalize_outputs(outputs)
        outputs, path_dropout_masks = self._apply_context_path_dropout(outputs)
        context.context_dict["path_dropout_masks"] = path_dropout_masks
        context.context_dict[self.path0_context_key] = outputs[0]

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
        # FiLM mode: no gated-output validation needed (output dim = primary path dim)

        if isinstance(outputs[0], FeatureMaps):
            result: FeatureMaps | FeatureVector = self._fuse_feature_maps(
                outputs, path_dropout_masks=path_dropout_masks
            )
        elif isinstance(outputs[0], FeatureVector):
            result = self._fuse_feature_vectors(
                outputs, path_dropout_masks=path_dropout_masks
            )
        else:
            raise TypeError(
                f"LateFusionFeatureExtractor only supports FeatureMaps and "
                f"FeatureVector outputs, got {first_type.__name__}."
            )

        return result

    def _fuse_feature_maps(
        self,
        outputs: list[FeatureMaps],
        path_dropout_masks: list[torch.Tensor] | None = None,
    ) -> FeatureMaps:
        """Dispatch to the correct FeatureMaps fusion method."""
        if self.fusion_mode == "concat":
            return self._concat_feature_maps(outputs)
        if self.fusion_mode == "gated":
            return self._gated_feature_maps(outputs)
        if self.fusion_mode == "film":
            return self._film_feature_maps(
                outputs, path_dropout_masks=path_dropout_masks
            )
        return self._mixing_feature_maps(outputs, path_dropout_masks=path_dropout_masks)

    def _fuse_feature_vectors(
        self,
        outputs: list[FeatureVector],
        path_dropout_masks: list[torch.Tensor] | None = None,
    ) -> FeatureVector:
        """Dispatch to the correct FeatureVector fusion method."""
        if self.fusion_mode == "concat":
            return self._concat_feature_vectors(outputs)
        if self.fusion_mode == "gated":
            return self._gated_feature_vectors(outputs)
        if self.fusion_mode == "film":
            return self._film_feature_vectors(
                outputs, path_dropout_masks=path_dropout_masks
            )
        return self._mixing_feature_vectors(
            outputs, path_dropout_masks=path_dropout_masks
        )
