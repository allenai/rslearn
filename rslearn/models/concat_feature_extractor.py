"""Generic late-fusion feature extractor via concatenation of parallel encoder paths."""

from typing import Any

import torch

from rslearn.train.model_context import ModelContext

from .component import (
    FeatureExtractor,
    FeatureMaps,
    FeatureVector,
    IntermediateComponent,
)


class ConcatFeatureExtractor(FeatureExtractor):
    """Late-fusion feature extractor that runs parallel encoder paths and concatenates their outputs.

    Example usage with OlmoEarth + ERA5 TCN encoder::

        encoder:
          - class_path: rslearn.models.concat_feature_extractor.ConcatFeatureExtractor
            init_args:
              paths:
                - - class_path: rslearn.models.olmoearth_pretrain.model.OlmoEarth
                    init_args: { ... }
                - - class_path: rslearn.models.tcn_encoder.TCNEncoder
                    init_args:
                      output_spatial_size: 16
                      ...
    """

    def __init__(
        self,
        paths: list[list[FeatureExtractor | IntermediateComponent]],
    ):
        """Create a new ConcatFeatureExtractor.

        Args:
            paths: a list of encoder paths.  Each path is itself a list of
                modules: the first module must be a ``FeatureExtractor`` and any
                subsequent modules must be ``IntermediateComponent`` instances.
                Every path is applied to the same ``ModelContext`` and the
                results are concatenated.
        """
        super().__init__()

        if len(paths) < 1:
            raise ValueError("ConcatFeatureExtractor requires at least one path")

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

        # Store as nested ModuleLists so all parameters are registered.
        self.paths = torch.nn.ModuleList([torch.nn.ModuleList(path) for path in paths])

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

    @staticmethod
    def _concat_feature_maps(outputs: list[FeatureMaps]) -> FeatureMaps:
        """Concatenate a list of FeatureMaps along the channel dimension.

        At each scale the spatial size is harmonised to the maximum H and W
        across the paths before concatenation.
        """
        n_scales = len(outputs[0].feature_maps)
        for i, o in enumerate(outputs):
            if len(o.feature_maps) != n_scales:
                raise ValueError(
                    f"All paths must produce the same number of feature map "
                    f"scales.  Path 0 has {n_scales} but path {i} has "
                    f"{len(o.feature_maps)}."
                )

        fused_maps: list[torch.Tensor] = []
        for scale_idx in range(n_scales):
            maps_at_scale = [o.feature_maps[scale_idx] for o in outputs]

            # Determine target spatial size (max H, max W across paths).
            max_h = max(m.shape[2] for m in maps_at_scale)
            max_w = max(m.shape[3] for m in maps_at_scale)

            aligned: list[torch.Tensor] = []
            for m in maps_at_scale:
                if m.shape[2] != max_h or m.shape[3] != max_w:
                    m = torch.nn.functional.interpolate(
                        m,
                        size=(max_h, max_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                aligned.append(m)

            fused_maps.append(torch.cat(aligned, dim=1))

        return FeatureMaps(fused_maps)

    @staticmethod
    def _concat_feature_vectors(outputs: list[FeatureVector]) -> FeatureVector:
        """Concatenate a list of FeatureVectors along the channel dimension."""
        return FeatureVector(
            feature_vector=torch.cat([o.feature_vector for o in outputs], dim=1)
        )

    def forward(self, context: ModelContext) -> FeatureMaps | FeatureVector:
        """Run all encoder paths and concatenate their outputs.

        Args:
            context: the model context (shared across all paths).

        Returns:
            a ``FeatureMaps`` or ``FeatureVector`` obtained by concatenating the
            outputs of every path along the channel dimension.

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

        if isinstance(outputs[0], FeatureMaps):
            return self._concat_feature_maps(outputs)  # type: ignore[arg-type]

        if isinstance(outputs[0], FeatureVector):
            return self._concat_feature_vectors(outputs)  # type: ignore[arg-type]

        raise TypeError(
            f"ConcatFeatureExtractor only supports FeatureMaps and "
            f"FeatureVector outputs, got {first_type.__name__}."
        )
