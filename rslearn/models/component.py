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
    """An intermediate output type for multi-resolution feature maps.

    Unlike `FeatureMaps`, these include an additional dimension for unpooled tokens.
    """

    # List of BxCxHxWxN feature maps at different scales, ordered from highest resolution
    # (most fine-grained) to lowest resolution (coarsest).
    feature_maps: list[torch.Tensor]


@dataclass
class FeatureVector:
    """An intermediate output type for a flat feature vector."""

    # Flat BxC feature vector.
    feature_vector: torch.Tensor
