"""MultiTaskModel for rslearn."""

from typing import Any, Optional

import torch


class MultiTaskModel(torch.nn.Module):
    """MultiTask model wrapper.

    MultiTask first passes its inputs through the sequential encoder models.

    Then, it applies one sequential decoder for each configured task. It computes
    outputs and loss using the final module in the decoder.
    """

    def __init__(
        self, encoder: list[torch.nn.Module], decoders: dict[str, list[torch.nn.Module]]
    ):
        """Initialize a new MultiTaskModel.

        Args:
            encoder: modules to compute intermediate feature representations.
            decoders: modules to compute outputs and loss, should match number of tasks.
        """
        super().__init__()
        self.encoder = torch.nn.Sequential(*encoder)
        self.decoders = torch.nn.ModuleDict(
            {name: torch.nn.ModuleList(decoder) for name, decoder in decoders.items()}
        )

    def forward(
        self,
        inputs: list[dict[str, Any]],
        targets: Optional[list[dict[str, Any]]] = None,
    ):
        """Apply the sequence of modules on the inputs.

        Args:
            inputs: list of input dicts
            targets: optional list of target dicts

        Returns:
            tuple (outputs, loss_dict) from the last module.
        """
        def extract_task_dict(d, task_name):
            # In MultiTask we will assign the inputs/targets for individual tasks with
            # task name prefix in the overall dicts.
            # So here we have to undo that prefix.
            # We retain the items that do not have the prefix since some might be
            # global input that were passthrough.
            prefix = task_name + "_"
            new_dict = {}
            for k, v in d.items():
                if k.startswith(prefix):
                    k = k[len(prefix):]
                new_dict[k] = v
            return new_dict

        features = self.encoder(inputs)
        outputs = [{} for _ in inputs]
        losses = {}
        for name, decoder in self.decoders.items():
            cur = features
            for module in decoder[:-1]:
                cur = module(cur, inputs)
            cur_inputs = [extract_task_dict(inp, name) for inp in inputs]
            cur_targets = [extract_task_dict(target, name) for target in targets]
            cur_output, cur_loss_dict = decoder[-1](cur, cur_inputs, cur_targets)
            for idx, entry in enumerate(cur_output):
                outputs[idx][name] = entry
            for loss_name, loss_value in cur_loss_dict.items():
                losses[f"{name}_{loss_name}"] = loss_value
        return outputs, losses
