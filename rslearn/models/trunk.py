"""Trunk module for decoder."""

from abc import ABC, abstractmethod
from typing import Any

import torch

from rslearn.log_utils import get_logger
from rslearn.models.task_embedding import BaseTaskEmbedding

logger = get_logger(__name__)


class DecoderTrunkLayer(torch.nn.Module, ABC):
    """Trunk layer for decoder."""

    def __init__(self) -> None:
        """Initialize the DecoderTrunkLayer module."""
        super().__init__()

    @abstractmethod
    def forward(
        self, x: torch.Tensor, task_embedding: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: input tensor of shape (batch_size, seq_len, dim)
            task_embedding: task embedding tensor of shape (batch_size, dim), or None

        Returns:
            dict with key "outputs" (output tensor of shape (batch_size, seq_len, dim))
            and optionally other keys.
        """

    @abstractmethod
    def apply_auxiliary_losses(
        self, trunk_out: dict[str, Any], outs: dict[str, Any]
    ) -> None:
        """Apply auxiliary losses in-place.

        Args:
            trunk_out: The output of the trunk.
            outs: The output of the decoders, with key "loss_dict" containing the losses.
        """


class DecoderTrunk(torch.nn.Module):
    """Trunk module for decoder, including arbitrary layers plus an optional task embedding."""

    def __init__(
        self,
        task_embedding: BaseTaskEmbedding | None = None,
        layers: list[DecoderTrunkLayer] | None = None,
    ) -> None:
        """Initialize the DecoderTrunk module.

        Args:
            task_embedding: Task-specific embedding module, or None if not using task embedding.
            layers: List of other shared layers. The first one should expect a
                B x T x C tensor, and the last should output a B x T x C tensor.
                All layers must output a dict with key "outputs" (output tensor of shape
                (B, T, C)) and optionally other keys.
        """
        super().__init__()
        self.layers = torch.nn.ModuleList(layers or [])
        self.task_embedding = task_embedding

        # If we have multiple instances of the same layer class, output keys will get overwritten
        if layers is not None:
            types = [type(layer) for layer in layers]
            if len(set(types)) != len(types):
                logger.warning(
                    "Multiple instances of the same layer class found in trunk. "
                    "Only the keys from the last instance will be used"
                )

    def register_tasks(self, task_names: list[str]) -> None:
        """Register tasks.

        Args:
            task_names: list of task names
        """
        if self.task_embedding is not None:
            self.task_embedding.register_tasks(task_names)

    def forward(
        self,
        features: list[torch.tensor],
        inputs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Forward pass.

        Args:
            features: The encoder features, a 1-list of B x C x H x W features.
            inputs: The original inputs to the encoder.

        Returns:
            dict with key "outputs" (output tensor of shape (batch_size, seq_len, dim))
            and optionally other keys from the other layers.
        """
        embeds = None
        if self.task_embedding is not None:
            embeds = self.task_embedding.compute_embeds(features, inputs)
            features = self.task_embedding(features, inputs, embeds=embeds)

        if not self.layers:
            return {"outputs": features}

        assert len(features) == 1, "DecoderTrunk only supports one feature map"
        x = torch.einsum("bchw->bhwc", features[0])
        x = torch.flatten(x, start_dim=1, end_dim=2)  # B x T x C, T = HW
        out = {}
        for layer in self.layers:
            layer_out = layer(x, task_embedding=embeds)
            x = layer_out.pop("outputs")  # unspecified shape
            out.update(layer_out)
        x = torch.einsum("btc->bct", x)  # B x C x T
        x = x.view(*features[0].shape)  # B x C x H x W

        out["outputs"] = [x]
        return out

    def apply_auxiliary_losses(
        self, trunk_out: dict[str, Any], outs: dict[str, Any]
    ) -> None:
        """Apply auxiliary losses in-place.

        Each layer handles its own auxiliary losses, assuming the loss key is `loss_dict`.

        Args:
            trunk_out: The output of the trunk.
            outs: The output of the decoders, with key "loss_dict" containing the losses.
        """
        for layer in self.layers:
            layer.apply_auxiliary_losses(trunk_out, outs)
