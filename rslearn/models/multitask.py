"""MultiTaskModel for rslearn."""

from typing import Any

import torch

from rslearn.log_utils import get_logger
from rslearn.models.task_embedding import BaseTaskEmbedding

logger = get_logger(__name__)


class MultiTaskModel(torch.nn.Module):
    """MultiTask model wrapper.

    MultiTaskModel first passes its inputs through the sequential encoder models.

    Then, it applies one sequential decoder for each configured task. It computes
    outputs and loss using the final module in the decoder.
    """

    def __init__(
        self,
        encoder: list[torch.nn.Module],
        decoders: dict[str, list[torch.nn.Module]],
        lazy_decode: bool = False,
        loss_weights: dict[str, float] | None = None,
        task_embedding: BaseTaskEmbedding | None = None,
        decoder_to_target: dict[str, list[str]] | None = None,
    ):
        """Initialize a new MultiTaskModel.

        Args:
            encoder: modules to compute intermediate feature representations.
            decoders: modules to compute outputs and loss, should match number of tasks.
            lazy_decode: if True, only decode the outputs specified in the batch.
            loss_weights: weights for each task's loss (default: None = equal weights)
            task_embedding: if provided, use this task embedding module to add task-specific
                embeddings to the features (recommended if merging decoder heads)
            decoder_to_target: mapping from decoder id to list of task names
                (specify if merging heads, otherwise leave as None)
        """
        super().__init__()
        self.lazy_decode = lazy_decode
        self.decoder_to_target: dict[str, list[str]] = decoder_to_target or {}
        self.encoder = torch.nn.Sequential(*encoder)
        self.decoders = torch.nn.ModuleDict(
            {name: torch.nn.ModuleList(decoder) for name, decoder in decoders.items()}
        )

        if decoder_to_target is None:
            self.decoder_to_target = {name: [name] for name in decoders.keys()}
        else:
            logger.info(f"merged decoders: {decoder_to_target}")

        self.target_to_decoder = {}
        for decoder_id, task_names in decoder_to_target.items():  # type: ignore
            for task_name in task_names:
                self.target_to_decoder[task_name] = decoder_id

        if loss_weights is None:
            loss_weights = {name: 1.0 for name in self.target_to_decoder.keys()}
        for name in self.target_to_decoder.keys():
            if name not in loss_weights:
                logger.warning(f"task {name} not in loss_weights, setting to 1.0")
                loss_weights[name] = 1.0
        self.loss_weights = loss_weights
        logger.info(f"loss_weights: {self.loss_weights}")

        if task_embedding is not None:
            self.task_embedding = task_embedding
            self.task_embedding.register_tasks(list(self.target_to_decoder.keys()))
            logger.info("registered decoders with task embedding")

    def apply_decoder(
        self,
        features: list[torch.Tensor],
        inputs: list[dict[str, Any]],
        targets: list[dict[str, Any]] | None,
        decoder: list[torch.nn.Module],
        task_name: str,
        outputs: list[dict[str, Any]],
        losses: dict[str, torch.Tensor],
    ) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
        """Apply a decoder to a list of inputs and targets.

        Args:
            features: list of features
            inputs: list of input dicts
            targets: list of target dicts
            decoder: list of decoder modules
            task_name: the name of the task
            outputs: list of output dicts
            losses: dictionary of loss values

        Returns:
            tuple of (outputs, losses)
        """
        # First, apply all but the last module in the decoder to the features
        cur = features
        for module in decoder[:-1]:
            cur = module(cur, inputs)

        if targets is None:
            cur_targets = None
        else:
            cur_targets = [target[task_name] for target in targets]

        # Then, apply the last module to the features and targets
        cur_output, cur_loss_dict = decoder[-1](cur, inputs, cur_targets)
        for idx, entry in enumerate(cur_output):
            outputs[idx][task_name] = entry
        for loss_name, loss_value in cur_loss_dict.items():
            losses[f"{task_name}_{loss_name}"] = (
                loss_value * self.loss_weights[task_name]
            )
        return outputs, losses

    def forward(
        self,
        inputs: list[dict[str, Any]],
        targets: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
        """Apply the sequence of modules on the inputs.

        Args:
            inputs: list of input dicts
            targets: optional list of target dicts

        Returns:
            tuple (outputs, loss_dict) from the last module.
        """
        features = self.encoder(inputs)
        outputs: list[dict[str, Any]] = [{} for _ in inputs]
        losses: dict[str, torch.Tensor] = {}

        if self.task_embedding is not None:
            features = self.task_embedding(features, inputs)

        if self.lazy_decode:
            # Assume that all inputs have the same dataset_source
            dataset_source = inputs[0]["dataset_source"]
            decoder = self.decoders[
                self.target_to_decoder.get(dataset_source, dataset_source)
            ]
            self.apply_decoder(
                features, inputs, targets, decoder, dataset_source, outputs, losses
            )
        else:
            for decoder_name, decoder in self.decoders.items():
                for task_name in self.decoder_to_target.get(decoder_name, decoder_name):
                    self.apply_decoder(
                        features, inputs, targets, decoder, task_name, outputs, losses
                    )

        return outputs, losses
