"""Concatenate feature map with features from input data."""

from typing import Any

import torch


class ConcatenateFeatures(torch.nn.Module):
    """Concatenate feature map with additional raw data inputs."""

    def __init__(self, key: str):
        """Create a new ConcatenateFeatures.

        Args:
            key: the key of the input_dict to concatenate.
        """
        super().__init__()
        self.key = key

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
        if len(features) != 1:
            raise ValueError(
                "Expect exactly one feature map to concatenate with additional data, got %d",
                len(features),
            )

        # Shape of feature map: BCHW
        base_features = features[0]
        feat_h, feat_w = base_features.shape[2], base_features.shape[3]

        concatenated_features = []
        for idx, input_data in enumerate(inputs):
            # Shape of additional features: CHW
            additional_features = input_data[self.key]
            if (
                additional_features.shape[1] != feat_w
                or additional_features.shape[2] != feat_h
            ):
                raise ValueError(
                    "Feature map and additional features have different shapes and cannot be concatenated"
                )
            concatenated_sample_features = torch.cat(
                [base_features[idx], additional_features], dim=0
            )
            concatenated_features.append(concatenated_sample_features)

        new_features = torch.stack(concatenated_features, dim=0)

        return [new_features]
