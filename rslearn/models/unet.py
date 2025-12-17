"""UNet-style decoder."""

from typing import Any

import torch
import torch.nn.functional as F

from rslearn.train.model_context import ModelContext

from .component import (
    FeatureMaps,
    IntermediateComponent,
)


class UNetDecoder(IntermediateComponent):
    """UNet-style decoder.

    It inputs multi-scale features. Starting from last (lowest resolution) feature map,
    it applies convolutional layers and upsampling iteratively while concatenating with
    the higher resolution feature maps when the resolution matches.
    """

    def __init__(
        self,
        in_channels: list[tuple[int, int]],
        out_channels: int | None,
        conv_layers_per_resolution: int = 1,
        kernel_size: int = 3,
        num_channels: dict[int, int] = {},
        target_resolution_factor: int = 1,
        original_size_to_interpolate: tuple[int, int] | None = None,
    ) -> None:
        """Initialize a UNetDecoder.

        Args:
            in_channels: list of (downsample factor, num channels) indicating the
                resolution (1/downsample_factor of input resolution) and number of
                channels in each feature map of the multi-scale features.
            out_channels: channels to output at each pixel, or None to skip the output
                layer.
            conv_layers_per_resolution: number of convolutional layers to apply after
                each up-sampling operation
            kernel_size: kernel size to use in convolutional layers
            num_channels: override number of output channels to use at different
                downsample factors.
            target_resolution_factor: output features at 1/target_resolution_factor
                relative to the input resolution. The default is 1 which outputs pixel
                level features.
            original_size_to_interpolate: the original size to interpolate the output to.
        """
        super().__init__()

        # Create convolutional and upsampling layers.
        # We have one Sequential of conv and potentially multiple upsampling layers for
        # each sequence in between concatenation with an input feature map.
        layers = []
        cur_layers = []
        cur_factor = in_channels[-1][0]
        cur_channels = in_channels[-1][1]
        cur_layers.extend(
            [
                torch.nn.Conv2d(
                    in_channels=cur_channels,
                    out_channels=cur_channels,
                    kernel_size=kernel_size,
                    padding="same",
                ),
                torch.nn.ReLU(inplace=True),
            ]
        )
        channels_by_factor = {factor: channels for factor, channels in in_channels}
        while cur_factor > target_resolution_factor:
            # Add upsampling layer.
            cur_layers.append(torch.nn.Upsample(scale_factor=2))
            cur_factor //= 2
            # If we need to concatenate here, then stop the current layers and add them
            # to the list.
            # Also update the number of channels to match the feature map that we'll be
            # concatenating with.
            if cur_factor in channels_by_factor:
                layers.append(torch.nn.Sequential(*cur_layers))
                # Number of output channels for this layer can be configured
                # per-resolution by the user, otherwise we default to the feature map
                # channels at the corresponding downsample factor.
                cur_out_channels = num_channels.get(
                    cur_factor, channels_by_factor[cur_factor]
                )
                cur_layers = [
                    torch.nn.Conv2d(
                        in_channels=cur_channels + channels_by_factor[cur_factor],
                        out_channels=cur_out_channels,
                        kernel_size=kernel_size,
                        padding="same",
                    ),
                    torch.nn.ReLU(inplace=True),
                ]
                cur_channels = cur_out_channels
            else:
                # Since there is no feature map at the next downsample factor, the
                # default is to keep the same number of channels (but the user can
                # still override it with num_channels).
                cur_out_channels = num_channels.get(cur_factor, cur_channels)
                cur_layers.extend(
                    [
                        torch.nn.Conv2d(
                            in_channels=cur_channels,
                            out_channels=cur_out_channels,
                            kernel_size=kernel_size,
                            padding="same",
                        ),
                        torch.nn.ReLU(inplace=True),
                    ]
                )
                cur_channels = cur_out_channels

            # Add remaining conv layers.
            for _ in range(conv_layers_per_resolution - 1):
                cur_layers.extend(
                    [
                        torch.nn.Conv2d(
                            in_channels=cur_channels,
                            out_channels=cur_channels,
                            kernel_size=kernel_size,
                            padding="same",
                        ),
                        torch.nn.ReLU(inplace=True),
                    ]
                )

        if out_channels is not None:
            cur_layers.append(
                torch.nn.Conv2d(
                    in_channels=cur_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                ),
            )
        layers.append(torch.nn.Sequential(*cur_layers))
        self.layers = torch.nn.ModuleList(layers)
        self.original_size_to_interpolate = original_size_to_interpolate

    def _resize(self, features: torch.Tensor) -> torch.Tensor:
        """Interpolate the features to the original size."""
        return F.interpolate(
            features,
            size=self.original_size_to_interpolate,
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, intermediates: Any, context: ModelContext) -> FeatureMaps:
        """Compute output from multi-scale feature map.

        Args:
            intermediates: the output from the previous model component, which must be a FeatureMaps.
            context: the model context.

        Returns:
            output FeatureMaps consisting of one map. The embedding size is equal to the
                configured out_channels.
        """
        if not isinstance(intermediates, FeatureMaps):
            raise ValueError("input to UNetDecoder must be a FeatureMaps")

        # Reverse the features since we will pass them in from lowest resolution to highest.
        in_features = list(reversed(intermediates.feature_maps))
        cur_features = self.layers[0](in_features[0])
        for in_feat, layer in zip(in_features[1:], self.layers[1:]):
            cur_features = layer(torch.cat([cur_features, in_feat], dim=1))
        if self.original_size_to_interpolate is not None:
            cur_features = self._resize(cur_features)
        return FeatureMaps([cur_features])
