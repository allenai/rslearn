"""Module wrappers."""

import torch


class DecoderModuleWrapper(torch.nn.Module):
    """Wrapper for a module that processes features to work in decoder.

    The module should input feature map and produce a new feature map.

    We wrap it to process each feature map in multi-scale features which is what's used
    for most decoders.
    """

    def __init__(
        self,
        module: torch.nn.Module,
    ):
        """Initialize a DecoderModuleWrapper.

        Args:
            module: the module to wrap
        """
        super().__init__()
        self.module = module

    def forward(
        self, features: list[torch.Tensor], inputs: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Apply the wrapped module on each feature map.

        Args:
            features: list of feature maps at different resolutions.
            inputs: original inputs (ignored).

        Returns:
            new features
        """
        new_features = []
        for feat_map in features:
            feat_map = self.module(feat_map)
            new_features.append(feat_map)
        return new_features
