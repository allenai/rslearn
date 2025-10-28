"""Embedding task."""

from typing import Any

import numpy.typing as npt
import torch
from torchmetrics import MetricCollection

from rslearn.utils import Feature

from .task import Task


class EmbeddingTask(Task):
    """A dummy task for computing embeddings.

    This task does not compute any targets or loss. Instead, it is just set up for
    inference, to save embeddings from the configured model.
    """

    def process_inputs(
        self,
        raw_inputs: dict[str, torch.Tensor],
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
        return {}, {}

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
        # Just convert the raw output to numpy array that can be saved to GeoTIFF.
        return raw_output.cpu().numpy()

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
        # EmbeddingTask is only set up to support `model predict`.
        raise NotImplementedError

    def get_metrics(self) -> MetricCollection:
        """Get the metrics for this task."""
        return MetricCollection({})


class EmbeddingHead(torch.nn.Module):
    """Head for embedding task.

    This picks one feature map from the input list of feature maps to output. It also
    returns a dummy loss.
    """

    def __init__(self, feature_map_index: int | None = 0):
        """Create a new EmbeddingHead.

        Args:
            feature_map_index: the index of the feature map to choose from the input
                list of multi-scale feature maps (default 0). If the input is already
                a single feature map, then set to None.
        """
        super().__init__()
        self.feature_map_index = feature_map_index

    def forward(
        self,
        features: torch.Tensor,
        inputs: list[dict[str, Any]],
        targets: list[dict[str, Any]] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Select the desired feature map and return it along with a dummy loss.

        Args:
            features: list of BCHW feature maps (or one feature map, if feature_map_index is None).
            inputs: original inputs (ignored).
            targets: should contain classes key that stores the per-pixel class labels.

        Returns:
            tuple of outputs and loss dict
        """
        if self.feature_map_index is not None:
            features = features[self.feature_map_index]

        return features, {"loss": 0}
