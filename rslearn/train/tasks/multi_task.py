"""Task for wrapping multiple tasks."""

from copy import deepcopy
from typing import Any

import numpy.typing as npt
import torch
from torchmetrics import Metric, MetricCollection

from rslearn.log_utils import get_logger
from rslearn.utils import Feature

from .task import Task

logger = get_logger(__name__)


class MultiTask(Task):
    """A task for training on multiple tasks."""

    def __init__(
        self,
        tasks: dict[str, Task],
        input_mapping: dict[str, dict[str, str]],
        task_label_offsets: dict[str, dict[str, int | str]] | None = None,
    ):
        """Create a new MultiTask.

        Args:
            tasks: map from task name to the task object
            input_mapping: for each task, maps which keys from the raw inputs should
                appear as potentially different keys for that task
            task_label_offsets: a dictionary mapping task name to a dictionary with
                "offset" (label offset) and "outputs_key" (key to use for the outputs).
                If specified, the labels for each task will be offset accordingly
        """
        self.tasks = tasks
        self.input_mapping = input_mapping
        self.task_label_offsets = task_label_offsets or {}

    def offset_task_labels(
        self,
        target_dict: dict[Any, Any],
        task_name: str,
    ) -> dict[Any, Any]:
        """Merge the task labels by adding an offset to the label key.

        Args:
            target_dict: the target dict
            task_name: the name of the task
        """
        offset = self.task_label_offsets[task_name]["offset"]
        outputs_key = self.task_label_offsets[task_name]["outputs_key"]
        target_dict[outputs_key] += offset
        return target_dict

    def process_inputs(
        self,
        raw_inputs: dict[str, torch.Tensor | list[Feature]],
        metadata: dict[str, Any],
        load_targets: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Processes the data into targets.

        Args:
            raw_inputs: raster or vector data to process
            metadata: metadata about the patch being read
            load_targets: whether to load the targets or only inputs

        Returns:
            tuple (input_dict, target_dict) containing the processed inputs and targets
                that are compatible with both metrics and loss functions
        """
        input_dict = {}
        target_dict = {}
        if metadata["dataset_source"] is None:
            # No multi-dataset, so always compute across all tasks
            task_iter = list(self.tasks.items())
        else:
            # Multi-dataset, so only compute for the task in this dataset
            task_iter = [
                (metadata["dataset_source"], self.tasks[metadata["dataset_source"]])
            ]

        for task_name, task in task_iter:
            cur_raw_inputs = {}
            for k, v in self.input_mapping[task_name].items():
                if k not in raw_inputs:
                    continue
                cur_raw_inputs[v] = raw_inputs[k]

            cur_input_dict, cur_target_dict = task.process_inputs(
                cur_raw_inputs, metadata=metadata, load_targets=load_targets
            )

            if self.task_label_offsets:
                cur_target_dict = self.offset_task_labels(cur_target_dict, task_name)

            input_dict[task_name] = cur_input_dict
            target_dict[task_name] = cur_target_dict

        return input_dict, target_dict

    def process_output(
        self, raw_output: Any, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Processes an output into raster or vector data.

        Args:
            raw_output: the output from prediction head.
            metadata: metadata about the patch being read

        Returns:
            either raster or vector data.
        """
        processed_output = {}
        for task_name, task in self.tasks.items():
            if task_name in raw_output:
                # In multi-dataset training, we may not have all datasets in the batch
                processed_output[task_name] = task.process_output(
                    raw_output[task_name], metadata
                )
        return processed_output

    def visualize(
        self,
        input_dict: dict[str, Any],
        target_dict: dict[str, Any] | None,
        output: dict[str, Any],
    ) -> dict[str, npt.NDArray[Any]]:
        """Visualize the outputs and targets.

        Args:
            input_dict: the input dict from process_inputs
            target_dict: the target dict from process_inputs
            output: the prediction

        Returns:
            a dictionary mapping image name to visualization image
        """
        images = {}
        for task_name, task in self.tasks.items():
            cur_target_dict = None
            if target_dict:
                cur_target_dict = target_dict[task_name]
            cur_images = task.visualize(input_dict, cur_target_dict, output[task_name])
            for label, image in cur_images.items():
                images[f"{task_name}_{label}"] = image
        return images

    def get_metrics(self) -> MetricCollection:
        """Get metrics for this task."""
        metrics = []
        for task_name, task in self.tasks.items():
            cur_metrics = {}
            for metric_name, metric in task.get_metrics().items():
                cur_metrics[metric_name] = MetricWrapper(
                    task_name, metric, self.task_label_offsets
                )
            metrics.append(MetricCollection(cur_metrics, prefix=f"{task_name}/"))
        return MetricCollection(metrics)


class MetricWrapper(Metric):
    """Wrapper for a metric from one task to operate in the multi-task setting.

    It selects the outputs and targets that are relevant to each task.
    """

    def __init__(
        self,
        task_name: str,
        metric: Metric,
        task_label_offsets: dict[str, dict[str, int | str]],
    ):
        """Create a new MetricWrapper.

        The wrapper passes the task-specific predictions and targets to the metrics of
        returned from each task.

        Args:
            task_name: the name of the task
            metric: one metric from the task to wrap
            task_label_offsets: a dictionary mapping task name to a dictionary with
                "offset" (label offset) and "outputs_key" (key to use for the outputs).
                If specified, the labels for each task will be offset accordingly.
                This must be specified if merging label across tasks, so that metrics
                can be computed correctly.
        """
        super().__init__()
        self.task_name = task_name
        self.metric = metric
        self.task_label_offsets = task_label_offsets

    def separate_task_labels(
        self,
        *,
        pred: torch.Tensor | dict | None = None,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Unmerge the task labels by subtracting an offset from the target.

        Also chop off the corresponding label dimensions in preds.

        Since we use the same pred/target tensors for all metrics in the collection,
        we need to clone them before modifying them.

        Assume first dimension is the number of outputs.

        Args:
            pred: the prediction
            target: the target
        """
        offset = self.task_label_offsets[self.task_name]["offset"]
        num_outputs = self.task_label_offsets[self.task_name]["num_outputs"]
        output_key = self.task_label_offsets[self.task_name]["outputs_key"]

        with torch.no_grad():
            if pred is not None:
                if isinstance(pred, dict):
                    # For some tasks (eg object detection), we have discrete label
                    # predictions instead of a distribution over labels
                    pred = deepcopy(pred)
                    pred[output_key] -= offset
                    return pred
                else:
                    # For classification/segmentation tasks, we have a distribution
                    # over labels, so we need to scale the predictions so that they
                    # sum to 1 since we chop off some of the probability densities
                    pred = pred.clone()[offset : offset + num_outputs, ...]  # type: ignore
                    pred /= pred.sum(dim=0, keepdim=True).type(torch.float32)
                    return pred

            elif target is not None:
                # Not one-hot encoded, so just subtract the other tasks' offset
                target = deepcopy(target)
                target[output_key] -= offset
                return target

            else:
                raise ValueError("Either pred or target must be provided")

    def update(
        self, preds: list[dict[str, Any]], targets: list[dict[str, Any]]
    ) -> None:
        """Update metric. Also unmerge task labels if they are merged.

        Args:
            preds: the predictions
            targets: the targets
        """
        try:
            self.metric.update(
                [
                    self.separate_task_labels(pred=pred[self.task_name])
                    for pred in preds
                ],
                [
                    self.separate_task_labels(target=target[self.task_name])
                    for target in targets
                ],
            )
        except KeyError:
            # In multi-dataset training, we may not have all datasets in the batch
            pass

    def compute(self) -> Any:
        """Returns the computed metric."""
        return self.metric.compute()

    def reset(self) -> None:
        """Reset metric."""
        super().reset()
        self.metric.reset()

    def plot(self, *args: list[Any], **kwargs: dict[str, Any]) -> Any:
        """Returns a plot of the metric."""
        return self.metric.plot(*args, **kwargs)
