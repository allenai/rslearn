"""MultiTaskModel for rslearn."""

from collections import defaultdict
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
        merge_heads: bool = False,
        task_embedding: BaseTaskEmbedding | None = None,
    ):
        """Initialize a new MultiTaskModel.

        Args:
            encoder: modules to compute intermediate feature representations.
            decoders: modules to compute outputs and loss, should match number of tasks.
            lazy_decode: if True, only decode the outputs specified in the batch.
            loss_weights: weights for each task's loss (default: None = equal weights)
            merge_heads: if True, merge the heads of decoders with the same task into a single head
            task_embedding: if provided, use this task embedding module to add task-specific embeddings to the features
                (required if merge_heads is True)
        """
        super().__init__()
        self.lazy_decode = lazy_decode
        self.merge_heads = merge_heads
        self.encoder = torch.nn.Sequential(*encoder)
        self.decoders = torch.nn.ModuleDict(
            {name: torch.nn.ModuleList(decoder) for name, decoder in decoders.items()}
        )
        self.target_to_decoder: dict[str, str] = {}
        self.decoder_to_target: dict[str, list[str]] = {}

        if lazy_decode:
            logger.info(
                "lazy decoding enabled, check dataset source is consistent across batch"
            )

        if loss_weights is None:
            loss_weights = {name: 1.0 for name in decoders.keys()}
        for name in decoders.keys():
            if name not in loss_weights:
                logger.warning(f"task {name} not in loss_weights, setting to 1.0")
                loss_weights[name] = 1.0
        self.loss_weights = loss_weights
        logger.info(f"loss_weights: {self.loss_weights}")

        if merge_heads:
            assert lazy_decode, "merge_heads requires lazy_decode"
            assert (
                task_embedding is not None
            ), "task_embedding is required if merge_heads is True"
            self.task_embedding = task_embedding
            self.task_embedding.register_tasks(list(decoders.keys()))

            used = {}
            last_to_all: dict[torch.nn.Module, torch.nn.ModuleList] = {}
            self.decoder_to_target = defaultdict(list)  # type: ignore
            for name, decoder_list in self.decoders.items():
                # For some reason ModuleLists are hashable, but this works for us
                new_decoder_name = decoder_list[-1].__class__.__name__
                used[decoder_list] = new_decoder_name
                self.target_to_decoder[name] = new_decoder_name
                self.decoder_to_target[new_decoder_name].append(name)

                # For now, assume decoders are architecturally the same if they
                # have the same last layer
                # TODO: stack output channels (out_channels, num_channels for detect)
                # and modify the targets correspondingly
                if last_to_all.get(decoder_list[-1], decoder_list) != decoder_list:
                    raise ValueError(
                        "All decoders with the same head must be identical"
                    )
                last_to_all[decoder_list[-1]] = decoder_list

            self.decoder_to_target = dict(self.decoder_to_target)
            self.decoders = torch.nn.ModuleDict({v: k for k, v in used.items()})
            logger.info(f"merged decoders {self.decoder_to_target}")
            logger.info(
                "if decoder params were restored, they may persist in the merged heads"
            )
            logger.info(
                f"using task embedding {self.task_embedding.__class__.__name__}"
            )

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

        if self.merge_heads:
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
