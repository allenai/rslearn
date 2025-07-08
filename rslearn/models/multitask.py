"""MultiTaskModel for rslearn."""

from typing import Any

import torch


def apply_decoder(
    features: list[torch.Tensor],
    inputs: list[dict[str, Any]],
    targets: list[dict[str, Any]] | None,
    decoder: list[torch.nn.Module],
    name: str,
    outputs: list[dict[str, Any]],
    losses: dict[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    """Apply a decoder to a list of inputs and targets.
    name is the name of the decoder/task (which must match).
    Return the output and loss dictionary.
    """
    # First, apply all but the last module in the decoder to the features
    cur = features
    for module in decoder[:-1]:
        cur = module(cur, inputs)

    if targets is None:
        cur_targets = None
    else:
        cur_targets = [target[name] for target in targets]

    # Then, apply the last module to the features and targets
    cur_output, cur_loss_dict = decoder[-1](cur, inputs, cur_targets)
    for idx, entry in enumerate(cur_output):
        outputs[idx][name] = entry
    for loss_name, loss_value in cur_loss_dict.items():
        losses[f"{name}_{loss_name}"] = loss_value
    return outputs, losses


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
    ):
        """Initialize a new MultiTaskModel.

        Args:
            encoder: modules to compute intermediate feature representations.
            decoders: modules to compute outputs and loss, should match number of tasks.
            lazy_decode: if True, only decode the outputs specified in the batch.
        """
        super().__init__()
        self.encoder = torch.nn.Sequential(*encoder)
        self.decoders = torch.nn.ModuleDict(
            {name: torch.nn.ModuleList(decoder) for name, decoder in decoders.items()}
        )
        self.lazy_decode = lazy_decode
        if lazy_decode:
            print(
                "INFO: lazy decoding enabled, check source is consistent across batch"
            )

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
        # ========= PATCH
        from copy import deepcopy

        inputs = deepcopy(inputs)
        targets = deepcopy(targets)
        for i in range(len(inputs)):
            inputs[i] = {
                **inputs[i]["TEST"],
                "dataset_source": inputs[i]["dataset_source"],
            }
        if targets is not None:
            for i in range(len(targets)):
                targets[i] = {**targets[i]["TEST"]}
        print("PATCHED INPUTS TEST - DECODER")
        # ========= PATCH
        features = self.encoder(inputs)
        outputs: list[dict[str, Any]] = [{} for _ in inputs]
        losses = {}
        if self.lazy_decode:
            # Assume that all inputs have the same dataset_source
            dataset_source = inputs[0]["dataset_source"]
            decoder = self.decoders[dataset_source]
            apply_decoder(
                features, inputs, targets, decoder, dataset_source, outputs, losses
            )
        else:
            for name, decoder in self.decoders.items():
                apply_decoder(features, inputs, targets, decoder, name, outputs, losses)
        return outputs, losses


class MultiDatasetMultiTaskModel(torch.nn.Module):
    """Multi-dataset multi-task model wrapper.

    MultiDatasetMultiTaskModel first passes its inputs through the sequential encoder models.

    Then, it applies one sequential decoder for each configured task. It computes
    outputs and loss using the final module in the selected decoder.
    """

    def __init__(
        self,
        encoder: list[torch.nn.Module],
        decoders: dict[str, dict[str, list[torch.nn.Module]]],
    ):
        """Initialize a new MultiDatasetMultiTaskModel.

        Args:
            encoder: modules to compute intermediate feature representations.
            decoders: modules to compute outputs and loss (nested by dataset source and task name).
        """
        super().__init__()
        self.encoder = torch.nn.Sequential(*encoder)
        self.decoders = torch.nn.ModuleDict(
            {
                src: torch.nn.ModuleDict(
                    {
                        task_name: torch.nn.ModuleList(decoder)
                        for task_name, decoder in task_dict.items()
                    }
                )
                for src, task_dict in decoders.items()
            }
        )

    def forward(
        self,
        inputs: list[dict[str, str | dict[str, Any]]],
        targets: list[dict[str, dict[str, Any]]] | None = None,
    ) -> tuple[list[dict[str, dict[str, Any]]], dict[str, dict[str, torch.Tensor]]]:
        """Apply the sequence of modules on the inputs.

        Args:
            inputs: list of input dicts (nested by dataset source and task name)
            targets: optional list of target dicts (nested by dataset source and task name)

        Returns:
            tuple (outputs, loss_dict) from the last module.
            The outputs and loss_dict are nested by dataset source and task name.
        """
        print("ABOUT TO ENCODE INPUTS!")
        features = self.encoder(inputs)
        outputs: list[dict[str, Any]] = [{} for _ in inputs]
        losses: dict[str, torch.Tensor] = {}

        # Assume that all inputs have the same dataset_source, so we can collapse the
        # nesting to only one level (task), but we need to unsqueeze b/c of MultiDatasetTask
        dataset_source: str = inputs[0]["dataset_source"]  # type: ignore
        print("ABOUT TO DECODE OUTPUTS! ", dataset_source)
        decoders = self.decoders[dataset_source]
        print("ABOUT TO APPLY DECODERS! ", decoders)
        for task_name, decoder in decoders.items():  # type: ignore
            apply_decoder(
                features, inputs, targets, decoder, task_name, outputs, losses
            )
            print("FINISHED DECODING! ", task_name)

        # Unsqueeze the losses to match what's expected by MultiDatasetTask
        unsqueezed_outputs = [{dataset_source: output} for output in outputs]
        unsqueezed_losses = {dataset_source: losses}
        print("FINISHED UNSQUEEZING! ", unsqueezed_outputs, unsqueezed_losses)
        return unsqueezed_outputs, unsqueezed_losses
