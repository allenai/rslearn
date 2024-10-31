"""Task for wrapping multiple tasks."""

from typing import Any

import numpy.typing as npt
import torch
from torchmetrics import Metric, MetricCollection

from rslearn.utils import Feature

from .task import Task


class MultiTask(Task):
    """A task for training on multiple tasks."""

    def __init__(
        self, tasks: dict[str, Task], input_mapping: dict[str, dict[str, str]]
    ):
        """Create a new MultiTask.

        Args:
            tasks: map from task name to the task object
            input_mapping: for each task, maps which keys from the raw inputs should
                appear as potentially different keys for that task
        """
        self.tasks = tasks
        self.input_mapping = input_mapping

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
        for task_name, task in self.tasks.items():
            cur_raw_inputs = {}
            for k, v in self.input_mapping[task_name].items():
                if k not in raw_inputs:
                    continue
                cur_raw_inputs[v] = raw_inputs[k]

            cur_input_dict, cur_target_dict = task.process_inputs(
                cur_raw_inputs, metadata=metadata, load_targets=load_targets
            )
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
                cur_metrics[metric_name] = MetricWrapper(task_name, metric)
            metrics.append(MetricCollection(cur_metrics, prefix=f"{task_name}/"))
        return MetricCollection(metrics)


class MetricWrapper(Metric):
    """Wrapper for a metric from one task to operate in the multi-task setting.

    It selects the outputs and targets that are relevant to each task.
    """

    def __init__(self, task_name: str, metric: Metric):
        """Create a new MetricWrapper.

        The wrapper passes the task-specific predictions and targets to the metrics of
        returned from each task.

        Args:
            task_name: the name of the task
            metric: one metric from the task to wrap
        """
        super().__init__()
        self.task_name = task_name
        self.metric = metric

    def update(
        self, preds: list[dict[str, Any]], targets: list[dict[str, Any]]
    ) -> None:
        """Update metric.

        Args:
            preds: the predictions
            targets: the targets
        """
        self.metric.update(
            [pred[self.task_name] for pred in preds],
            [target[self.task_name] for target in targets],
        )

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
