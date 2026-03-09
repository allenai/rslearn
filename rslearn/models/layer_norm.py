"""Layer normalization over intermediate outputs."""

from typing import Any

import torch

from rslearn.train.model_context import ModelContext

from .component import (
    FeatureMaps,
    FeatureVector,
    IntermediateComponent,
    TokenFeatureMaps,
)


class LayerNorm(IntermediateComponent):
    """Apply layer normalization to supported intermediate outputs.

    This component supports:
    - FeatureMaps
    - TokenFeatureMaps
    - FeatureVector
    - Raw torch.Tensor

    Normalization is applied over a single configured dimension.
    """

    def __init__(
        self,
        num_channels: int,
        normalize_dim: int = 1,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        """Initialize a LayerNorm component.

        Args:
            num_channels: size of the dimension to normalize over.
            normalize_dim: tensor dimension to normalize over. Defaults to 1.
            eps: a value added to the denominator for numerical stability.
            elementwise_affine: if True, this module has learnable affine parameters.
        """
        super().__init__()
        self.num_channels = num_channels
        self.normalize_dim = normalize_dim
        self.layer = torch.nn.LayerNorm(
            normalized_shape=num_channels,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )

    def _apply_to_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply layer norm to a tensor."""
        if tensor.ndim < 1:
            raise ValueError("input tensor to LayerNorm must have at least 1 dimension")

        dim = self.normalize_dim
        if dim < 0:
            dim += tensor.ndim
        if dim < 0 or dim >= tensor.ndim:
            raise ValueError(
                f"normalize_dim {self.normalize_dim} is out of range for tensor "
                f"with {tensor.ndim} dimensions"
            )

        size = tensor.shape[dim]
        if size != self.num_channels:
            raise ValueError(
                "input size at normalize_dim does not match configured num_channels: "
                f"{size} != {self.num_channels}"
            )

        # LayerNorm normalizes over trailing dimensions, so move the configured
        # normalization dim to the end, apply layer norm, then move it back.
        if dim == tensor.ndim - 1:
            return self.layer(tensor)
        permuted = torch.movedim(tensor, dim, -1)
        normalized = self.layer(permuted)
        return torch.movedim(normalized, -1, dim)

    def forward(self, intermediates: Any, context: ModelContext) -> Any:
        """Apply layer normalization to the given intermediate."""
        if isinstance(intermediates, FeatureMaps):
            return FeatureMaps(
                [
                    self._apply_to_tensor(feat_map)
                    for feat_map in intermediates.feature_maps
                ]
            )

        if isinstance(intermediates, TokenFeatureMaps):
            return TokenFeatureMaps(
                [
                    self._apply_to_tensor(feat_map)
                    for feat_map in intermediates.feature_maps
                ]
            )

        if isinstance(intermediates, FeatureVector):
            return FeatureVector(self._apply_to_tensor(intermediates.feature_vector))

        if isinstance(intermediates, torch.Tensor):
            return self._apply_to_tensor(intermediates)

        raise ValueError(
            "input to LayerNorm must be FeatureMaps, TokenFeatureMaps, FeatureVector, "
            "or torch.Tensor"
        )
