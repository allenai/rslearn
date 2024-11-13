"""Simple pooling decoder."""

from typing import Any

import torch


class PoolingDecoder(torch.nn.Module):
    """Decoder that computes flat vector from a 2D feature map.

    It inputs multi-scale features, but only uses the last feature map. Then applies a
    configurable number of convolutional layers before pooling, and a configurable
    number of fully connected layers after pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_conv_layers: int = 0,
        num_fc_layers: int = 0,
        conv_channels: int = 128,
        fc_channels: int = 512,
    ) -> None:
        """Initialize a PoolingDecoder.

        Args:
            in_channels: input channels (channels in the last feature map passed to
                this module)
            out_channels: channels for the output flat feature vector
            num_conv_layers: number of convolutional layers to apply, default 0
            num_fc_layers: number of fully-connected layers to apply, default 0
            conv_channels: number of channels to use for convolutional layers
            fc_channels: number of channels to use for fully-connected layers
        """
        super().__init__()
        conv_layers = []
        prev_channels = in_channels
        for _ in range(num_conv_layers):
            conv_layer = torch.nn.Sequential(
                torch.nn.Conv2d(prev_channels, conv_channels, 3, padding=1),
                torch.nn.ReLU(inplace=True),
            )
            conv_layers.append(conv_layer)
            prev_channels = conv_channels
        self.conv_layers = torch.nn.Sequential(*conv_layers)

        fc_layers = []
        for _ in range(num_fc_layers):
            fc_layer = torch.nn.Sequential(
                torch.nn.Linear(prev_channels, fc_channels),
                torch.nn.ReLU(inplace=True),
            )
            fc_layers.append(fc_layer)
            prev_channels = fc_channels
        self.fc_layers = torch.nn.Sequential(*fc_layers)

        self.output_layer = torch.nn.Linear(prev_channels, out_channels)

    def forward(
        self, features: list[torch.Tensor], inputs: list[dict[str, Any]]
    ) -> torch.Tensor:
        """Compute flat output vector from multi-scale feature map.

        Args:
            features: list of feature maps at different resolutions.
            inputs: original inputs (ignored).

        Returns:
            flat feature vector
        """
        # Only use last feature map.
        features = features[-1]

        features = self.conv_layers(features)
        features = torch.amax(features, dim=(2, 3))
        features = self.fc_layers(features)
        return self.output_layer(features)
