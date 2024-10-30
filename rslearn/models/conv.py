"""A single convolutional layer."""

import torch


class Conv(torch.nn.Module):
    """A single convolutional layer.

    It inputs a set of feature maps; the conv layer is applied to each feature map
    independently, and list of outputs is returned.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: str = "same",
        stride: int = 1,
        activation: torch.nn.Module = torch.nn.ReLU(inplace=True),
    ):
        """Initialize a Conv.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: kernel size
            padding: either "same" or "valid" to control padding
            stride: stride to apply.
            activation: activation to apply after convolution
        """
        super().__init__()

        self.layer = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, stride=stride
        )
        self.activation = activation

    def forward(
        self, features: list[torch.Tensor], inputs: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Compute flat output vector from multi-scale feature map.

        Args:
            features: list of feature maps at different resolutions.
            inputs: original inputs (ignored).

        Returns:
            flat feature vector
        """
        new_features = []
        for feat_map in features:
            feat_map = self.layer(feat_map)
            feat_map = self.activation(feat_map)
            new_features.append(feat_map)
        return new_features
