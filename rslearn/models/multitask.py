"""MultiTaskModel for rslearn."""

from copy import deepcopy
from typing import Any

import torch

from rslearn.log_utils import get_logger
from rslearn.models.trunk import DecoderTrunk

logger = get_logger(__name__)


def sort_keys(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively (half in place) sort the keys of a dictionary.

    Need this so that the order of task embeddings indexing is consistent.

    Args:
        d (dict[str, Any]): The dictionary to sort.
    """
    d = {k: d[k] for k in sorted(d)}
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = sort_keys(v)
    return d


def deepcopy_tensordict(d: dict[Any, Any]) -> dict[Any, Any]:
    """Deepcopy a dict with torch.Tensor, dict, and other types.

    Make sure tensor copying is handled properly.

    Args:
        d: the dict to deepcopy
    """
    new_d = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            new_d[k] = torch.clone(v)
        elif isinstance(v, dict):
            new_d[k] = deepcopy_tensordict(v)
        else:
            new_d[k] = deepcopy(v)
    return new_d


class MultiTaskModel(torch.nn.Module):
    """MultiTask model wrapper.

    MultiTaskModel first passes its inputs through the sequential encoder models.

    Then, it applies one sequential decoder for each configured task. It computes
    outputs and loss using the final module in the decoder.

    Optionally include a shared trunk module to postprocess the encoder features.
    """

    def __init__(
        self,
        encoder: list[torch.nn.Module],
        decoders: dict[str, list[torch.nn.Module]],
        lazy_decode: bool = False,
        loss_weights: dict[str, float] | None = None,
        trunk: DecoderTrunk | None = None,
    ):
        """Initialize a new MultiTaskModel.

        Args:
            encoder: modules to compute intermediate feature representations.
            decoders: modules to compute outputs and loss, should match number of tasks.
            lazy_decode: if True, only decode the outputs specified in the batch.
            loss_weights: weights for each task's loss (default: None = equal weights).
            trunk: if provided, use this trunk module to postprocess the features
                (recommend including a task-specific embedding module here).
        """
        super().__init__()
        self.lazy_decode = lazy_decode
        self.encoder = torch.nn.Sequential(*encoder)
        self.decoders = torch.nn.ModuleDict(
            sort_keys(
                {
                    name: torch.nn.ModuleList(decoder)
                    for name, decoder in decoders.items()
                }
            )
        )
        self._init_loss_weights(loss_weights, list(self.decoders.keys()))
        self._init_trunk(trunk, list(self.decoders.keys()))

    def _init_loss_weights(
        self, loss_weights: dict[str, float] | None, task_names: list[str]
    ) -> None:
        """Initialize the loss weights for the tasks.

        Args:
            loss_weights: weights for each task's loss (default: None = equal weights).
            task_names: list of task names.
        """
        if loss_weights is None:
            loss_weights = {name: 1.0 for name in task_names}
        for name in task_names:
            if name not in loss_weights:
                logger.warning(f"task {name} not in loss_weights, setting to 1.0")
                loss_weights[name] = 1.0
        self.loss_weights = sort_keys(loss_weights)
        logger.info(f"loss_weights: {self.loss_weights}")

    def _init_trunk(self, trunk: DecoderTrunk | None, task_names: list[str]) -> None:
        """Initialize the trunk module.

        Args:
            trunk: the trunk module.
            task_names: list of task names.
        """
        self.trunk = trunk
        if trunk is not None:
            trunk.register_tasks(task_names)
            logger.info("registered decoders with trunk")

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

    def _get_tasks_from_decoder(self, decoder: str) -> list[str]:
        """Get the tasks corresponding to this decoder.

        Args:
            decoder: the name of the decoder
        """
        return [decoder]

    def apply_decoders(
        self,
        features: list[torch.Tensor],
        inputs: list[dict[str, Any]],
        targets: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """Apply all the decoders to the features and targets.

        Args:
            features: list of features
            inputs: list of input dicts
            targets: list of target dicts

        Returns:
            dict of outputs and losses
        """
        outputs: list[dict[str, torch.Tensor | dict]] = [{} for _ in inputs]
        losses: dict[str, torch.Tensor] = {}

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
                for task_name in self._get_tasks_from_decoder(decoder_name):
                    self.apply_decoder(
                        features, inputs, targets, decoder, task_name, outputs, losses
                    )

        return {
            "outputs": outputs,
            "loss_dict": losses,
        }

    def forward(
        self,
        inputs: list[dict[str, Any]],
        targets: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Apply the sequence of modules on the inputs, including shared trunk.

        Args:
            inputs: list of input dicts
            targets: optional list of target dicts

        Returns:
            dict with keys "outputs" and "loss_dict".
        """
        features = self.encoder(inputs)
        if self.trunk is not None:
            trunk_out = self.trunk(features, inputs)
            outs = self.apply_decoders(trunk_out.pop("outputs"), inputs, targets)
            self.trunk.apply_auxiliary_losses(trunk_out, outs)
            return outs | trunk_out
        else:
            return self.apply_decoders(features, inputs, targets)


class MultiTaskMergedModel(MultiTaskModel):
    """MultiTaskMerged model.

    MultiTaskModel first passes its inputs through the sequential encoder models.

    Then, it applies one sequential decoder for each configured task. It computes
    outputs and loss using the final module in the decoder.

    Also, possibly merges labels across a single task type.
    """

    def __init__(
        self,
        encoder: list[torch.nn.Module],
        decoders: dict[str, list[torch.nn.Module]],
        decoder_to_target: dict[str, list[str]],
        task_label_offsets: dict[str, dict[str, Any]],
        lazy_decode: bool = False,
        loss_weights: dict[str, float] | None = None,
        trunk: DecoderTrunk | None = None,
    ):
        """Initialize a new MultiTaskModel.

        Args:
            encoder: modules to compute intermediate feature representations.
            decoders: modules to compute outputs and loss, should match number of tasks.
            decoder_to_target: mapping from decoder id to list of task names
                (specify if merging heads, otherwise leave as None).
            task_label_offsets: mapping from task name to dict of info (output_key, offset)
                (specify if merging label groups across a single task).
            lazy_decode: if True, only decode the outputs specified in the batch.
            loss_weights: weights for each task's loss (default: None = equal weights).
            trunk: if provided, use this trunk module to postprocess the features
                (recommend including a task-specific embedding module here).
        """
        # Can't use super() because we need to skip calls to _init_loss_weights and _init_trunk
        torch.nn.Module.__init__(self)

        self.lazy_decode = lazy_decode
        self.encoder = torch.nn.Sequential(*encoder)
        self.decoders = torch.nn.ModuleDict(
            sort_keys(
                {
                    name: torch.nn.ModuleList(decoder)
                    for name, decoder in decoders.items()
                }
            )
        )
        self.task_label_offsets = task_label_offsets

        self.decoder_to_target = sort_keys(decoder_to_target)
        logger.info(f"merged decoders: {self.decoder_to_target}")

        self.target_to_decoder = {}
        for decoder_id, task_names in self.decoder_to_target.items():
            for task_name in task_names:
                self.target_to_decoder[task_name] = decoder_id
        self.target_to_decoder = sort_keys(self.target_to_decoder)

        self._init_loss_weights(loss_weights, list(self.target_to_decoder.keys()))
        self._init_trunk(trunk, list(self.target_to_decoder.keys()))

    def merge_task_labels(
        self,
        targets: list[dict[str, Any]] | None,
        task_name: str,
    ) -> list[dict[str, Any]] | None:
        """Merge the task labels by adding an offset to the label key.

        Make a clone before doing this because we may use targets elsewhere.

        Args:
            targets: the target dicts
            task_name: the name of the task
        """
        if targets is None:
            return targets
        offset = self.task_label_offsets[task_name]["offset"]
        outputs_key = self.task_label_offsets[task_name]["outputs_key"]
        offset_targets = []
        for target in targets:
            offset_target = deepcopy_tensordict(target)
            spliced = offset_target[task_name]
            if torch.is_floating_point(spliced[outputs_key]):
                logger.warning(
                    f"task {task_name} has targets of type "
                    f"{spliced[outputs_key].dtype}, "
                    f"expected int (shape {spliced[outputs_key].shape})"
                )
            with torch.no_grad():
                spliced[outputs_key] += offset
            offset_targets.append(offset_target)
        return offset_targets

    def unmerge_output_labels(
        self, outputs: list[dict[str, torch.Tensor | dict]], task_name: str
    ) -> None:
        """Unmerge the task labels in place.

        For most tasks, this means chopping off the corresponding label dimensions.
        For some, we might just need to subtract an offset from the target (ex: segmentation).
        Assume first dimension is the number of outputs.

        Args:
            outputs: the predictions
            task_name: the name of the task
        """
        offset = self.task_label_offsets[task_name]["offset"]
        num_outputs = self.task_label_offsets[task_name]["num_outputs"]
        output_key = self.task_label_offsets[task_name]["outputs_key"]
        with torch.no_grad():
            for i, output in enumerate(outputs):
                if not output:
                    # Possible if there are no detections
                    continue
                output = output[task_name]
                if isinstance(output, dict):
                    # For some tasks (eg object detection), we have discrete label
                    # predictions instead of a distribution over labels
                    output[output_key] -= offset
                elif isinstance(output, torch.Tensor):
                    # For classification/segmentation tasks, we have a distribution
                    # over labels, so we need to scale the predictions so that they
                    # sum to 1 since we chop off some of the probability densities
                    tensor: torch.Tensor = output[offset : offset + num_outputs, ...]
                    tensor /= tensor.sum(dim=0, keepdim=True).type(torch.float32)
                    outputs[i][task_name] = tensor

    def forward(
        self,
        inputs: list[dict[str, Any]],
        targets: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Apply the sequence of modules on the inputs.

        Args:
            inputs: list of input dicts
            targets: optional list of target dicts

        Returns:
            dict with keys "outputs" and "loss_dict", and possibly other keys.
        """
        dataset_source = inputs[0].get("dataset_source", None)
        targets = self.merge_task_labels(targets, dataset_source)
        outs = super().forward(inputs, targets)
        self.unmerge_output_labels(outs["outputs"], dataset_source)
        return outs

    def _get_tasks_from_decoder(self, decoder: str) -> list[str]:
        """Get the tasks corresponding to this decoder.

        Args:
            decoder: the name of the decoder
        """
        return self.decoder_to_target[decoder]
