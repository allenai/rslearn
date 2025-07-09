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
        self, features: list[torch.Tensor], inputs: list[dict[str, Any]]
    ) -> list[torch.Tensor]:
        """Pool over the modalities.

        Args:
            features: 1-list of MBCHW tensor
            inputs: list of input dicts (not used)

        Returns:
            1-list of BCHW tensor averaged over the modalities
        """
        return [features[0].mean(dim=0)]


class AttentivePool(BasePool):
    """Attentive pooling over the modalities, so we get one BCHW feature map."""

    def __init__(self, n_channels: int, height: int, width: int) -> None:
        """Initialize a new AttentivePool.

        Args:
            n_channels: number of channels in the input features
            height: height of the input features
            width: width of the input features
        """
        super().__init__()
        self.linear = torch.nn.Linear(n_channels * height * width, 1)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(
        self,
        features: list[torch.Tensor],
        inputs: list[dict[str, Any]],
    ) -> list[torch.Tensor]:
        """Pool over the modalities via attentive pooling.

        Given the MBCHW features, we compute a linear projection MBCHW -> MB
        and apply softmax over M axis. We use the resulting MB tensor to linearly
        combine the MBCHW tensor along M axis, giving us a BCHW tensor.

        Args:
            features: 1-list of MBCHW tensor
            inputs: list of input dicts (not used)

        Returns:
            1-list of BCHW tensor
        """
        x = features[0]
        M, B, *_ = x.shape
        flattened = x.flatten(start_dim=2)  # M, B, CHW
        out = self.softmax(self.linear(flattened))  # M, B
        weighted = out.view(M, B, 1, 1, 1) * x  # M, B, C, H, W
        pooled = weighted.sum(dim=0)  # B, C, H, W
        return [pooled]
