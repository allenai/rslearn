"""Pooling modules post-encoder."""

from typing import Any

import torch


class BasePool(torch.nn.Module):
    """Base class for pooling modules. Used for type checking."""

    def __init__(self) -> None:
        """Initialize a new BasePool."""
        super().__init__()


class AveragePool(BasePool):
    """Simple average pool over the modalities, so we get one BCHW feature map."""

    def __init__(self) -> None:
        """Initialize a new ModalityPool."""
        super().__init__()

    def forward(
        self,
        features: list[torch.Tensor],
        inputs: list[dict[str, Any]],
        return_weights: bool = False,
    ) -> tuple[list[torch.Tensor], torch.Tensor] | list[torch.Tensor]:
        """Pool over the modalities.

        Args:
            features: 1-list of M x B x C x H x W tensor
            inputs: list of input dicts (not used)
            return_weights: whether to return the weights (always uniform)

        Returns:
            1-list of BCHW tensor averaged over the modalities, and optional weights tensor
        """
        x = features[0].mean(dim=0)  # BCHW
        weights = torch.ones(features[0].shape[-1]) / features[0].shape[-1]
        if return_weights:
            return [x], weights
        return [x]


class AttentivePool(BasePool):
    """Attentive pooling over the modalities, so we get one BCHW feature map."""

    def __init__(
        self, n_bandsets: int, n_channels: int, height: int, width: int
    ) -> None:
        """Initialize a new AttentivePool.

        Args:
            n_bandsets: number of bandset features across all modalities
            n_channels: number of channels in the input features
            height: height of the input features
            width: width of the input features
        """
        super().__init__()
        self.linear = torch.nn.Linear(
            n_bandsets * n_channels * height * width, n_bandsets
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(
        self,
        features: list[torch.Tensor],
        inputs: list[dict[str, Any]],
        return_weights: bool = False,
    ) -> tuple[list[torch.Tensor], torch.Tensor] | list[torch.Tensor]:
        """Pool over the modalities via attentive pooling.

        Given the MBCHW tensor, compute attention weights by projecting the features
        to MB and applying softmax over M axis. We use the resulting MB tensor to linearly
        combine the MBCHW tensor along M axis, giving us a BCHW tensor.

        Args:
            features: 1-list of MBCHW tensor
            inputs: list of input dicts (not used)
            return_weights: whether to return the attention weights

        Returns:
            1-list of BCHW tensor, and optional weights tensor
        """
        x_in = features[0]  # M, B, C, H, W
        x_in = x_in.permute(1, 0, 2, 3, 4)  # B, M, C, H, W
        x = x_in.flatten(start_dim=1)  # B, MCHW
        x = self.softmax(self.linear(x))  # B, M
        weights = x.view(*x.shape, 1, 1, 1)  # B, M, 1, 1, 1
        pooled = (weights * x_in).sum(dim=1)  # B, C, H, W
        if return_weights:
            return [pooled], weights[:, :, 0, 0, 0]
        return [pooled]
