"""Training tasks."""

from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import torch

from rslearn.utils import Feature


class Task:
    """Represents an ML task like object detection or segmentation.

    A task specifies how raster or vector data should be processed into targets that
    can be passed to models. It also specifies evaluation functions for computing
    metrics comparing targets/outputs.
    """

    def get_target(self, data: Union[npt.NDArray[Any], list[Feature]]) -> Any:
        """Processes the data into targets.

        Args:
            data: raster or vector data to process

        Returns:
            the processed targets that are compatible with both metrics and loss
                functions
        """
        raise NotImplementedError

    def visualize(
        self, input_dict: dict[str, Any], output: Any, target: Optional[Any]
    ) -> dict[str, npt.NDArray[Any]]:
        """Visualize the outputs and targets.

        Args:
            input_dict: the input
            output: the prediction
            target: the target label from get_target

        Returns:
            a dictionary mapping image name to visualization image
        """
        raise NotImplementedError


class BasicTask(Task):
    """A task that provides some support for creating visualizations."""

    def __init__(
        self,
        image_bands: tuple[int, ...] = (0, 1, 2),
        remap_values: Optional[tuple[tuple[float, float], tuple[int, int]]] = None,
    ):
        """Initialize a new BasicTask.

        Args:
            image_bands: which bands from the input image to use for the visualization.
            remap_values: if set, remap the values from the first range to the second range
        """
        self.image_bands = image_bands
        self.remap_values = remap_values

    def visualize(
        self, input_dict: dict[str, Any], output: Any, target: Optional[Any]
    ) -> dict[str, npt.NDArray[Any]]:
        """Visualize the outputs and targets.

        Args:
            input_dict: the input
            output: the prediction
            target: the target label from get_target

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
