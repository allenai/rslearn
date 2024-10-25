"""UNet-style decoder."""

from typing import Any

import torch


class UNetDecoder(torch.nn.Module):
    """UNet-style decoder.

    It inputs multi-scale features. Starting from last (lowest resolution) feature map,
    it applies convolutional layers and upsampling iteratively while concatenating with
    the higher resolution feature maps when the resolution matches.
    """

    def __init__(
        self,
        in_channels: list[tuple[int, int]],
        out_channels: int,
        conv_layers_per_resolution: int = 1,
        kernel_size: int = 3,
    ) -> None:
        """Initialize a UNetDecoder.

        Args:
            in_channels: list of (downsample factor, num channels) indicating the
                resolution (1/downsample_factor of input resolution) and number of
                channels in each feature map of the multi-scale features.
            out_channels: channels to output at each pixel.
            conv_layers_per_resolution: number of convolutional layers to apply after
                each up-sampling operation
            kernel_size: kernel size to use in convolutional layers
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
        while cur_factor > 1:
            # Add upsampling layer.
            cur_layers.append(torch.nn.Upsample(scale_factor=2))
            cur_factor //= 2
            # If we need to concatenate here, then stop the current layers and add them
            # to the list.
            # Also update the number of channels to match the feature map that we'll be
            # concatenating with.
            if cur_factor in channels_by_factor:
                layers.append(torch.nn.Sequential(*cur_layers))
                cur_layers = [
                    torch.nn.Conv2d(
                        in_channels=cur_channels + channels_by_factor[cur_factor],
                        out_channels=channels_by_factor[cur_factor],
                        kernel_size=kernel_size,
                        padding="same",
                    ),
                    torch.nn.ReLU(inplace=True),
                ]
                cur_channels = channels_by_factor[cur_factor]
            else:
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

    def forward(
        self, in_features: list[torch.Tensor], inputs: list[dict[str, Any]]
    ) -> torch.Tensor:
        """Compute output from multi-scale feature map.

        Args:
            in_features: list of feature maps at different resolutions.
            inputs: original inputs (ignored).

        Returns:
            output image
        """
        # Reverse the features since we will pass them in from lowest resolution to highest.
        in_features = list(reversed(in_features))
        cur_features = self.layers[0](in_features[0])
        for in_feat, layer in zip(in_features[1:], self.layers[1:]):
            cur_features = layer(torch.cat([cur_features, in_feat], dim=1))
        return cur_features
