"""Concatenate feature map with features from input data."""

from typing import Any

import torch


class ConcatenateFeatures(torch.nn.Module):
    """Concatenate feature map with additional raw data inputs."""

    def __init__(
        self,
        key: str,
        in_channels: int | None = None,
        conv_channels: int = 64,
        out_channels: int | None = None,
        num_conv_layers: int = 1,
        kernel_size: int = 3,
    ):
        """Create a new ConcatenateFeatures.

        Args:
            key: the key of the input_dict to concatenate.
            in_channels: number of input channels of the additional features.
            conv_channels: number of channels of the convolutional layers.
            out_channels: number of output channels of the additional features.
            num_conv_layers: number of convolutional layers to apply to the additional features.
            kernel_size: kernel size of the convolutional layers.
        """
        super().__init__()
        self.key = key

        if num_conv_layers > 0:
            if in_channels is None or out_channels is None:
                raise ValueError(
                    "in_channels and out_channels must be specified if num_conv_layers > 0"
                )

        conv_layers = []
        for i in range(num_conv_layers):
            conv_in = in_channels if i == 0 else conv_channels
            conv_out = out_channels if i == num_conv_layers - 1 else conv_channels
            conv_layers.append(
                torch.nn.Conv2d(
                    in_channels=conv_in,
                    out_channels=conv_out,
                    kernel_size=kernel_size,
                    padding="same",
                )
            )
            if i < num_conv_layers - 1:
                conv_layers.append(torch.nn.ReLU(inplace=True))

        self.conv_layers = torch.nn.Sequential(*conv_layers)

    def forward(
        self, features: list[torch.Tensor], inputs: list[dict[str, Any]]
    ) -> list[torch.Tensor]:
        """Concatenate the feature map with the raw data inputs.

        Args:
            features: list of feature maps at different resolutions.
            inputs: original inputs.

        Returns:
            concatenated feature maps.
        """
        if not features:
            raise ValueError("Expected at least one feature map, got none.")

        add_data = torch.stack([input_data[self.key] for input_data in inputs], dim=0)
        add_features = self.conv_layers(add_data)

        new_features: list[torch.Tensor] = []
        for feature_map in features:
            # Shape of feature map: BCHW
            feat_h, feat_w = feature_map.shape[2], feature_map.shape[3]

            resized_add_features = add_features
            # Resize additional features to match each feature map size if needed
            if add_features.shape[2] != feat_h or add_features.shape[3] != feat_w:
                resized_add_features = torch.nn.functional.interpolate(
                    add_features,
                    size=(feat_h, feat_w),
                    mode="bilinear",
                    align_corners=False,
                )

            new_features.append(torch.cat([feature_map, resized_add_features], dim=1))

        return new_features
