"""Embedding task."""

from typing import Any

import numpy.typing as npt
import torch
from torchmetrics import MetricCollection

from rslearn.models.component import FeatureMaps, Predictor
from rslearn.train.model_context import ModelContext, ModelOutput, SampleMetadata
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
        return {}, {}

    def process_output(
        self, raw_output: Any, metadata: SampleMetadata
    ) -> npt.NDArray[Any] | list[Feature]:
        """Processes an output into raster or vector data.

        Args:
            raw_output: the output from prediction head, which must be a CxHxW tensor.
            metadata: metadata about the patch being read

        Returns:
            either raster or vector data.
        """
        if not isinstance(raw_output, torch.Tensor) or len(raw_output.shape) != 3:
            raise ValueError(
                "output for EmbeddingTask must be a tensor with three dimensions"
            )

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


class EmbeddingHead(Predictor):
    """Head for embedding task.

    It just adds a dummy loss to act as a Predictor.
    """

    def forward(
        self,
        intermediates: Any,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Return the feature map along with a dummy loss.

        Args:
            intermediates: output from the previous model component, which must be a
                FeatureMaps consisting of a single feature map.
            context: the model context.
            targets: the targets (ignored).

        Returns:
            model output with the feature map that was input to this component along
                with a dummy loss.
        """
        if not isinstance(intermediates, FeatureMaps):
            raise ValueError("input to EmbeddingHead must be a FeatureMaps")
        if len(intermediates.feature_maps) != 1:
            raise ValueError(
                f"input to EmbeddingHead must have one feature map, but got {len(intermediates.feature_maps)}"
            )

        return ModelOutput(
            outputs=intermediates.feature_maps[0],
            loss_dict={"loss": 0},
        )
