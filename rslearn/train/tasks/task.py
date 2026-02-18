"""Training tasks."""

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torchmetrics import MetricCollection

from rslearn.train.model_context import RasterImage, SampleMetadata
from rslearn.utils import Feature


class Task:
    """Represents an ML task like object detection or segmentation.

    A task specifies how raster or vector data should be processed into inputs and
    targets that can be passed to models. It also specifies evaluation functions for
    computing metrics comparing targets/outputs.
    """

    def process_inputs(
        self,
        raw_inputs: dict[str, RasterImage | list[Feature]],
        metadata: SampleMetadata,
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
        self, raw_output: Any, metadata: SampleMetadata
    ) -> npt.NDArray[Any] | list[Feature] | dict[str, Any]:
        """Processes an output into raster or vector data.

        Args:
            raw_output: the output from prediction head.
            metadata: metadata about the patch being read

        Returns:
            raster data, vector data, or multi-task dictionary output.
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

    @staticmethod
    def _get_window_valid_mask(
        reference_hw: torch.Tensor, metadata: SampleMetadata
    ) -> torch.Tensor:
        """Return an HW float mask of pixels that fall within window_bounds.

        Raster readers may pad regions outside window_bounds (but inside crop_bounds)
        with zeros when decoding. This mask lets tasks treat those padded pixels as
        invalid, independent of any nodata_value semantics.
        """
        if reference_hw.ndim != 2:
            raise ValueError(
                f"expected an HW tensor for reference_hw, got shape {reference_hw.shape}"
            )

        window_x0, window_y0, window_x1, window_y1 = metadata.window_bounds
        crop_x0, crop_y0, crop_x1, crop_y1 = metadata.crop_bounds
        inter_x0 = max(window_x0, crop_x0)
        inter_y0 = max(window_y0, crop_y0)
        inter_x1 = min(window_x1, crop_x1)
        inter_y1 = min(window_y1, crop_y1)

        window_valid = torch.zeros(
            reference_hw.shape, dtype=torch.float32, device=reference_hw.device
        )
        if inter_x0 < inter_x1 and inter_y0 < inter_y1:
            x0 = int(inter_x0 - crop_x0)
            y0 = int(inter_y0 - crop_y0)
            x1 = int(inter_x1 - crop_x0)
            y1 = int(inter_y1 - crop_y0)
            window_valid[y0:y1, x0:x1] = 1.0

        return window_valid

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
        raster_image = input_dict["image"]
        assert isinstance(raster_image, RasterImage)
        # We don't really handle time series here, just use the first timestep.
        image = raster_image.image.cpu()[self.image_bands, 0, :, :]
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
