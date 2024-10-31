"""An upsampling layer."""

import torch


class Upsample(torch.nn.Module):
    """Upsamples each input feature map by the same factor."""

    def __init__(
        self,
        scale_factor: int,
        mode: str = "bilinear",
    ):
        """Initialize an Upsample.

        Args:
            scale_factor: the upsampling factor, e.g. 2 to double the size.
            mode: "nearest" or "bilinear".
        """
        super().__init__()
        self.layer = torch.nn.Upsample(scale_factor=scale_factor, mode=mode)

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
        return [self.layer(feat_map) for feat_map in features]
