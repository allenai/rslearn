"""A single convolutional layer."""

from typing import Any

import torch

from rslearn.train.model_context import ModelContext

from .component import FeatureMaps, IntermediateComponent


class Conv(IntermediateComponent):
    """A single convolutional layer.

    It inputs a set of feature maps; the conv layer is applied to each feature map
    independently, and list of outputs is returned.

    Optionally, when ``context_key`` is set, the same ``Conv2d`` + activation is
    also applied to a ``FeatureMaps`` entry stored in
    ``context.context_dict[context_key]``, and the result is written back.  This
    allows auxiliary features (e.g. path0 intermediates from a late-fusion
    encoder) to flow through the **same decoder weights** as the main features,
    which is useful for auxiliary-loss regularisation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: str | int = "same",
        stride: int = 1,
        activation: torch.nn.Module = torch.nn.ReLU(inplace=True),
        context_key: str | None = None,
    ):
        """Initialize a Conv.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: kernel size, see torch.nn.Conv2D.
            padding: padding to apply, see torch.nn.Conv2D.
            stride: stride to apply, see torch.nn.Conv2D.
            activation: activation to apply after convolution
            context_key: optional key into ``context.context_dict``.  When set,
                the Conv will also read a ``FeatureMaps`` stored under this key,
                apply the same ``Conv2d`` + activation to each feature map, and
                write the result back.
        """
        super().__init__()

        self.layer = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, stride=stride
        )
        self.activation = activation
        self.context_key = context_key

    def _apply_conv(self, feature_maps: FeatureMaps) -> FeatureMaps:
        """Apply ``self.layer`` + ``self.activation`` to every map in *feature_maps*."""
        new_features = []
        for feat_map in feature_maps.feature_maps:
            feat_map = self.layer(feat_map)
            feat_map = self.activation(feat_map)
            new_features.append(feat_map)
        return FeatureMaps(new_features)

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

        result = self._apply_conv(intermediates)

        # Optionally process the auxiliary FeatureMaps stored in context_dict.
        if self.context_key is not None:
            ctx_value = context.context_dict.get(self.context_key)
            if ctx_value is not None:
                if not isinstance(ctx_value, FeatureMaps):
                    raise ValueError(
                        f"Conv context_key '{self.context_key}' expected a "
                        f"FeatureMaps in context_dict, got "
                        f"{type(ctx_value).__name__}."
                    )
                context.context_dict[self.context_key] = self._apply_conv(ctx_value)

        return result
