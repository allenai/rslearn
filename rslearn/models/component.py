"""Model component API."""

import abc
from dataclasses import dataclass
from typing import Any

import torch

from rslearn.train.model_context import ModelContext, ModelOutput


class FeatureExtractor(torch.nn.Module, abc.ABC):
    """A feature extractor that performs initial processing of the inputs.

    The FeatureExtractor is the first component in the encoders list for
    SingleTaskModel and MultiTaskModel.
    """

    @abc.abstractmethod
    def forward(self, context: ModelContext) -> Any:
        """Extract an initial intermediate from the model context.

        Args:
            context: the model context.

        Returns:
            any intermediate to pass to downstream components. Oftentimes this is a
                FeatureMaps.
        """
        raise NotImplementedError


class IntermediateComponent(torch.nn.Module, abc.ABC):
    """An intermediate component in the model.

    In SingleTaskModel and MultiTaskModel, modules after the first module
    in the encoders list are IntermediateComponents, as are modules before the last
    module in the decoders list(s).
    """

    @abc.abstractmethod
    def forward(self, intermediates: Any, context: ModelContext) -> Any:
        """Process the given intermediate into another intermediate.

        Args:
            intermediates: the output from the previous component (either a
                FeatureExtractor or another IntermediateComponent).
            context: the model context.

        Returns:
            any intermediate to pass to downstream components.
        """
        raise NotImplementedError


class Predictor(torch.nn.Module, abc.ABC):
    """A predictor that computes task-specific outputs and a loss dict.

    In SingleTaskModel and MultiTaskModel, the last module(s) in the decoders list(s)
    are Predictors.
    """

    @abc.abstractmethod
    def forward(
        self,
        intermediates: Any,
        context: ModelContext,
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> ModelOutput:
        """Compute task-specific outputs and loss dict.

        Args:
            intermediates: the output from the previous component.
            context: the model context.
            targets: the training targets, or None during prediction.

        Returns:
            a tuple of the task-specific outputs (which should be compatible with the
                configured Task) and loss dict. The loss dict maps from a name for each
                loss to a scalar tensor.
        """
        raise NotImplementedError


@dataclass
class FeatureMaps:
    """An intermediate output type for multi-resolution feature maps."""

    # List of BxCxHxW feature maps at different scales, ordered from highest resolution
    # (most fine-grained) to lowest resolution (coarsest).
    feature_maps: list[torch.Tensor]


@dataclass
class TokenFeatureMaps:
    """An intermediate output type for multi-resolution BCHWN feature maps with a token dimension.

    Unlike `FeatureMaps`, these include an additional dimension for unpooled tokens.

    The number of tokens N is fixed across the batch, but the optional `masks` field
    can indicate which tokens are valid at each position so that variable-length
    token sequences (e.g. a different number of timesteps per sample) can be
    represented. Components that consume a TokenFeatureMaps should ignore tokens
    where the mask is False; when `masks` is None, all tokens are valid.
    """

    # List of BxCxHxWxN feature maps at different scales, ordered from highest resolution
    # (most fine-grained) to lowest resolution (coarsest).
    feature_maps: list[torch.Tensor]

    # Optional list of BxHxWxN bool masks, one per feature map, where True indicates a
    # valid token and False indicates a padded/missing token that should be ignored.
    masks: list[torch.Tensor] | None = None

    def __post_init__(self) -> None:
        """Validate that the masks (if any) align with the feature maps."""
        if self.masks is None:
            return
        if len(self.masks) != len(self.feature_maps):
            raise ValueError(
                f"TokenFeatureMaps has {len(self.feature_maps)} feature maps but "
                f"{len(self.masks)} masks"
            )
        for feat, mask in zip(self.feature_maps, self.masks):
            b, _, h, w, n = feat.shape
            if mask.shape != (b, h, w, n):
                raise ValueError(
                    f"TokenFeatureMaps mask shape {tuple(mask.shape)} does not match "
                    f"feature map shape {tuple(feat.shape)} (expected BHWN)"
                )
            if mask.dtype != torch.bool:
                raise ValueError(
                    f"TokenFeatureMaps masks must be bool tensors, got {mask.dtype}"
                )

    def get_masks(self) -> list[torch.Tensor | None]:
        """Get one optional BxHxWxN bool mask per feature map.

        Returns:
            a list aligned with `feature_maps`, where each entry is the mask for the
                corresponding feature map, or None if `masks` is None (all tokens
                valid).
        """
        if self.masks is None:
            return [None] * len(self.feature_maps)
        return list(self.masks)


@dataclass
class FeatureVector:
    """An intermediate output type for a flat feature vector."""

    # Flat BxC feature vector.
    feature_vector: torch.Tensor
