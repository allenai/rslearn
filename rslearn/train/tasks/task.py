"""Training tasks."""

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torchmetrics import MetricCollection

from rslearn.utils import Feature


class Task:
    """Represents an ML task like object detection or segmentation.

    A task specifies how raster or vector data should be processed into inputs and
    targets that can be passed to models. It also specifies evaluation functions for
    computing metrics comparing targets/outputs.
    """

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
        raise NotImplementedError

    def process_output(
        self, raw_output: Any, metadata: dict[str, Any]
    ) -> npt.NDArray[Any] | list[Feature]:
        """Processes an output into raster or vector data.

        Args:
            raw_output: the output from prediction head.
            metadata: metadata about the patch being read

        Returns:
            either raster or vector data.
        """
        raise NotImplementedError

    def visualize(
        self,
        input_dict: dict[str, Any],
        target_dict: dict[str, Any] | None,
        output: Any,
    ) -> dict[str, npt.NDArray[Any]]:
        """Visualize the outputs and targets.

        Args:
            input_dict: the input dict from process_inputs
            target_dict: the target dict from process_inputs
            output: the prediction

        Returns:
            a dictionary mapping image name to visualization image
        """
        raise NotImplementedError

    def get_metrics(self) -> MetricCollection:
        """Get metrics for this task."""
        raise NotImplementedError


class BasicTask(Task):
    """A task that provides some support for creating visualizations."""

    def __init__(
        self,
        image_bands: tuple[int, ...] = (0, 1, 2),
        remap_values: tuple[tuple[float, float], tuple[int, int]] | None = None,
    ):
        """Initialize a new BasicTask.

        Args:
            image_bands: which bands from the input image to use for the visualization.
            remap_values: if set, remap the values from the first range to the second range
        """
        self.image_bands = image_bands
        self.remap_values = remap_values

    def visualize(
        self,
        input_dict: dict[str, Any],
        target_dict: dict[str, Any] | None,
        output: Any,
    ) -> dict[str, npt.NDArray[Any]]:
        """Visualize the outputs and targets.

        Args:
            input_dict: the input dict from process_inputs
            target_dict: the target dict from process_inputs
            output: the prediction

        Returns:
            a dictionary mapping image name to visualization image
        """
        image = input_dict["image"].cpu()
        image = image[self.image_bands, :, :]
        if self.remap_values:
            factor = (self.remap_values[1][1] - self.remap_values[1][0]) / (
                self.remap_values[0][1] - self.remap_values[0][0]
            )
            image = (image - self.remap_values[0][0]) * factor + self.remap_values[1][0]
        return {
            "image": torch.clip(image, 0, 255)
            .numpy()
            .transpose(1, 2, 0)
            .astype(np.uint8),
        }
