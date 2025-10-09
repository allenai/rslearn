"""The ResizeFeatures module."""

import torch


class ResizeFeatures(torch.nn.Module):
    """Resize input features to new sizes."""

    def __init__(
        self,
        out_sizes: list[tuple[int, int]],
        mode: str = "bilinear",
    ):
        """Initialize a ResizeFeatures.

        Args:
            out_sizes: the output sizes of the feature maps. There must be one entry
                for each input feature map.
            mode: mode to pass to torch.nn.Upsample, e.g. "bilinear" (default) or
                "nearest".
        """
        super().__init__()
        layers = []
        for size in out_sizes:
            layers.append(
                torch.nn.Upsample(
                    size=size,
                    mode=mode,
                )
            )
        self.layers = torch.nn.ModuleList(layers)

    def forward(
        self, features: list[torch.Tensor], inputs: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Resize the input feature maps to new sizes.

        Args:
            features: list of feature maps at different resolutions.
            inputs: original inputs (ignored).

        Returns:
            resized feature maps
        """
        return [self.layers[idx](feat_map) for idx, feat_map in enumerate(features)]
