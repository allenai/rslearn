"""SingleTaskModel for rslearn."""

from typing import Any

import torch


class SingleTaskModel(torch.nn.Module):
    """Standard model wrapper.

    SingleTaskModel first passes its inputs through the sequential encoder models.

    Then, it passes the computed features through the decoder models, obtaining the
    outputs and targets from the last module (which also receives the targets).
    """

    def __init__(self, encoder: list[torch.nn.Module], decoder: list[torch.nn.Module]):
        """Initialize a new SingleTaskModel.

        Args:
            encoder: modules to compute intermediate feature representations.
            decoder: modules to compute outputs and loss.
        """
        super().__init__()
        self.encoder = torch.nn.Sequential(*encoder)
        self.decoder = torch.nn.ModuleList(decoder)

    def forward(
        self,
        inputs: list[dict[str, Any]],
        targets: list[dict[str, Any]] | None = None,
    ) -> tuple[list[Any], dict[str, torch.Tensor]]:
        """Apply the sequence of modules on the inputs.

        Args:
            inputs: list of input dicts
            targets: optional list of target dicts

        Returns:
            tuple (outputs, loss_dict) from the last module.
        """
        features = self.encoder(inputs)
        cur = features
        for module in self.decoder[:-1]:
            cur = module(cur, inputs)

        return self.decoder[-1](cur, inputs, targets)
