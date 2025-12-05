"""A single convolutional layer."""

from typing import Any

import torch

from rslearn.train.model_context import ModelContext

from .component import FeatureMaps, IntermediateComponent


class Conv(IntermediateComponent):
    """A single convolutional layer.

    It inputs a set of feature maps; the conv layer is applied to each feature map
    independently, and list of outputs is returned.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: str | int = "same",
        stride: int = 1,
        activation: torch.nn.Module = torch.nn.ReLU(inplace=True),
    ):
        """Initialize a Conv.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: kernel size, see torch.nn.Conv2D.
            padding: padding to apply, see torch.nn.Conv2D.
            stride: stride to apply, see torch.nn.Conv2D.
            activation: activation to apply after convolution
        """
        super().__init__()

        self.layer = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, stride=stride
        )
        self.activation = activation

    def forward(self, intermediates: Any, context: ModelContext) -> FeatureMaps:
        """Apply conv layer on each feature map.

        Args:
            intermediates: the previous output, which must be a FeatureMaps.
            context: the model context.

        Returns:
            the resulting feature maps after applying the same Conv2d on each one.
        """
        if not isinstance(intermediates, FeatureMaps):
            raise ValueError("input to Conv must be FeatureMaps")

        new_features = []
        for feat_map in intermediates.feature_maps:
            feat_map = self.layer(feat_map)
            feat_map = self.activation(feat_map)
            new_features.append(feat_map)
        return FeatureMaps(new_features)
