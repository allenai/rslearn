"""SingleTaskModel for rslearn."""

from typing import Any

import torch

from rslearn.train.model_context import ModelContext, ModelOutput

from .component import FeatureExtractor, IntermediateComponent, Predictor


class SingleTaskModel(torch.nn.Module):
    """Standard model wrapper.

    SingleTaskModel first passes its inputs through the sequential encoder models.

    Then, it passes the computed features through the decoder models, obtaining the
    outputs and targets from the last module (which also receives the targets).
    """

    def __init__(
        self,
        encoder: list[FeatureExtractor | IntermediateComponent],
        decoder: list[IntermediateComponent | Predictor],
    ):
        """Initialize a new SingleTaskModel.

        Args:
            encoder: modules to compute intermediate feature representations. The first
                module must be a FeatureExtractor, and following modules must be
                IntermediateComponents.
            decoder: modules to compute outputs and loss. The last module must be a
                Predictor, while the previous modules must be IntermediateComponents.
        """
        super().__init__()
        self.encoder = torch.nn.ModuleList(encoder)
        self.decoder = torch.nn.ModuleList(decoder)

    def forward(
        self,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Apply the sequence of modules on the inputs.

        Args:
            context: the model context.
            targets: optional list of target dicts
            info: optional dictionary of info to pass to the last module

        Returns:
            the model output.
        """
        cur = self.encoder[0](context)
        for module in self.encoder[1:]:
            cur = module(cur, context)
        for module in self.decoder[:-1]:
            cur = module(cur, context)
        return self.decoder[-1](cur, context, targets)
