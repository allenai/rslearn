"""rslearn PredictionWriter implementation."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from upath import UPath

from rslearn.config import (
    LayerType,
    RasterFormatConfig,
    RasterLayerConfig,
    VectorLayerConfig,
)
from rslearn.dataset import Dataset, Window
from rslearn.utils.array import copy_spatial_array
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import PixelBounds
from rslearn.utils.raster_format import RasterFormat, load_raster_format
from rslearn.utils.vector_format import VectorFormat, load_vector_format

from .lightning_module import RslearnLightningModule
from .tasks.task import Task


@dataclass
class PendingPatchOutput:
    """A patch output that hasn't been merged yet."""

    bounds: PixelBounds
    output: Any


class PatchPredictionMerger:
    """Base class for merging predictions from multiple patches."""

    def merge(self, window: Window, outputs: Sequence[PendingPatchOutput]) -> Any:
        """Merge the outputs.

        Args:
            window: the window we are merging the outputs for.
            outputs: the outputs to process.

        Returns:
            the merged outputs.
        """
        raise NotImplementedError


class VectorMerger(PatchPredictionMerger):
    """Merger for vector data that simply concatenates the features."""

    def merge(
        self, window: Window, outputs: Sequence[PendingPatchOutput]
    ) -> list[Feature]:
        """Concatenate the vector features."""
        return [feat for output in outputs for feat in output.output]


class RasterMerger(PatchPredictionMerger):
    """Merger for raster data that copies the rasters to the output."""

    def __init__(self, padding: int | None = None):
        """Create a new RasterMerger.

        Args:
            padding: the padding around the individual patch outputs to remove. This is
                typically used when leveraging overlapping patches. Portions of outputs
                at the border of the window will still be retained.
        """
        self.padding = padding

    def merge(
        self, window: Window, outputs: Sequence[PendingPatchOutput]
    ) -> npt.NDArray:
        """Merge the raster outputs."""
        num_channels = outputs[0].output.shape[0]
        dtype = outputs[0].output.dtype
        merged_image = np.zeros(
            (
                num_channels,
                window.bounds[3] - window.bounds[1],
                window.bounds[2] - window.bounds[0],
            ),
            dtype=dtype,
        )

        # Ensure the outputs are sorted by height then width.
        # This way when we merge we can be sure that outputs that are lower or further
        # to the right will overwrite earlier outputs.
        sorted_outputs = sorted(
            outputs, key=lambda output: (output.bounds[0], output.bounds[1])
        )
        for output in sorted_outputs:
            # So now we just need to compute the src_offset to copy.
            # If the output is not on the left or top boundary, then we should apply
            # the padding (if set).
            src = output.output
            src_offset = (output.bounds[0], output.bounds[1])
            if self.padding is not None and output.bounds[0] != window.bounds[0]:
                src = src[:, :, self.padding :]
                src_offset = (src_offset[0] + self.padding, src_offset[1])
            if self.padding is not None and output.bounds[1] != window.bounds[1]:
                src = src[:, self.padding :, :]
                src_offset = (src_offset[0], src_offset[1] + self.padding)

            copy_spatial_array(
                src=src,
                dst=merged_image,
                src_offset=src_offset,
                dst_offset=(window.bounds[0], window.bounds[1]),
            )

        return merged_image


class RslearnWriter(BasePredictionWriter):
    """A writer that writes predictions back into the rslearn dataset.

    The predictions are stored in a specified output layer, which must not exist yet
    for each window being processed.
    """

    layer_config: RasterLayerConfig | VectorLayerConfig

    def __init__(
        self,
        path: str,
        output_layer: str,
        path_options: dict[str, Any] | None = None,
        selector: list[str] | None = None,
        merger: PatchPredictionMerger | None = None,
        output_path: str | Path | None = None,
        layer_config: RasterLayerConfig | VectorLayerConfig | None = None,
    ):
        """Create a new RslearnWriter.

        Args:
            path: the dataset root directory.
            output_layer: which layer to write the outputs under.
            path_options: additional options for path to pass to fsspec
            selector: keys to access the desired output in the output dict if needed.
                e.g ["key1", "key2"] gets output["key1"]["key2"]
            merger: merger to use to merge outputs from overlapped patches.
            output_path: optional custom path for writing predictions. If provided,
                predictions will be written to this path instead of deriving from dataset path.
            layer_config: optional layer configuration. If provided, this config will be
                used instead of reading from the dataset config, allowing usage without
                requiring dataset config at the output path.
        """
        super().__init__(write_interval="batch")
        self.output_layer = output_layer
        self.selector = selector or []
        self.path = UPath(path, **path_options or {})
        self.output_path = (
            UPath(output_path, **path_options or {})
            if output_path is not None
            else None
        )

        # Handle dataset and layer config
        if layer_config:
            self.layer_config = layer_config
            self.dataset = None if self.output_path else Dataset(self.path)
        else:
            self.dataset = Dataset(self.path)
            if self.output_layer not in self.dataset.layers:
                raise KeyError(
                    f"Output layer '{self.output_layer}' not found in dataset layers."
                )
            raw_layer_config = self.dataset.layers[self.output_layer]
            # Type narrowing to ensure compatibility
            if isinstance(raw_layer_config, (RasterLayerConfig | VectorLayerConfig)):
                self.layer_config = raw_layer_config
            else:
                raise ValueError(
                    f"Layer config must be RasterLayerConfig or VectorLayerConfig, got {type(raw_layer_config)}"
                )

        self.format: RasterFormat | VectorFormat
        if self.layer_config.layer_type == LayerType.RASTER:
            assert isinstance(self.layer_config, RasterLayerConfig)
            band_cfg = self.layer_config.band_sets[0]
            self.format = load_raster_format(
                RasterFormatConfig(band_cfg.format["name"], band_cfg.format)
            )
        elif self.layer_config.layer_type == LayerType.VECTOR:
            assert isinstance(self.layer_config, VectorLayerConfig)
            self.format = load_vector_format(self.layer_config.format)
        else:
            raise ValueError(f"invalid layer type {self.layer_config.layer_type}")

        if merger is not None:
            self.merger = merger
        elif self.layer_config.layer_type == LayerType.RASTER:
            self.merger = RasterMerger()
        elif self.layer_config.layer_type == LayerType.VECTOR:
            self.merger = VectorMerger()

        # Map from window name to pending data to write.
        # This is used when windows are split up into patches, so the data from all the
        # patches of each window need to be reconstituted.
        self.pending_outputs: dict[str, list[PendingPatchOutput]] = {}

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: dict[str, Sequence],
        batch_indices: Sequence[int] | None,
        batch: tuple[list, list, list],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Write a batch of predictions into the rslearn dataset.

        Args:
            trainer: the trainer.
            pl_module: the LightningModule.
            prediction: the prediction to write.
            batch_indices: batch indices.
            batch: the batch that was input to the model. It should be a list of
                (inputs, targets, metadatas).
            batch_idx: the batch index.
            dataloader_idx: the index in the dataloader.
        """
        assert isinstance(pl_module, RslearnLightningModule)
        task = pl_module.task
        _, _, metadatas = batch
        self.process_output_batch(task, prediction["outputs"], metadatas)

    def process_output_batch(
        self,
        task: Task,
        prediction: Sequence,
        metadatas: Sequence,
    ) -> None:
        """Write a prediction batch with simplified API.

        write_on_batch_end wraps this function to work with lightning API, but only a
        subset of arguments are used.

        Args:
            task: the Task that we are writing outputs for.
            prediction: the list of predictions in this batch to write. These outputs
                will be processed by the task to obtain a vector (list[Feature]) or
                raster (npt.NDArray) output.
            metadatas: corresponding list of metadatas from the batch describing the
                patches that were processed.
        """
        # Process the predictions into outputs that can be written.
        outputs: list = [
            task.process_output(output, metadata)
            for output, metadata in zip(prediction, metadatas)
        ]

        for output, metadata in zip(outputs, metadatas):
            for k in self.selector:
                output = output[k]

            # # Use custom output_path if provided, otherwise use dataset path
            window_base_path = (
                self.output_path if self.output_path is not None else self.path
            )
            window = Window(
                path=Window.get_window_root(
                    window_base_path, metadata["group"], metadata["window_name"]
                ),
                group=metadata["group"],
                name=metadata["window_name"],
                projection=metadata["projection"],
                bounds=metadata["window_bounds"],
                time_range=metadata["time_range"],
            )
            self.process_output(
                window,
                metadata["patch_idx"],
                metadata["num_patches"],
                metadata["bounds"],
                output,
            )

    def process_output(
        self,
        window: Window,
        patch_idx: int,
        num_patches: int,
        cur_bounds: PixelBounds,
        output: npt.NDArray | list[Feature],
    ) -> None:
        """Process one output from the model.

        Args:
            window: the window that the output pertains to.
            patch_idx: the index of this patch for the window.
            num_patches: the total number of patches to be processed for the window.
            cur_bounds: the bounds of the current patch.
            output: the output data.
        """
        # Incorporate the output into our list of pending patch outputs.
        if window.name not in self.pending_outputs:
            self.pending_outputs[window.name] = []
        self.pending_outputs[window.name].append(PendingPatchOutput(cur_bounds, output))

        if patch_idx < num_patches - 1:
            return

        # This is the last patch so it's time to write it.
        # First get the pending output and clear it.
        pending_output = self.pending_outputs[window.name]
        del self.pending_outputs[window.name]

        # Merge outputs from overlapped patches if merger is set.
        merged_output = self.merger.merge(window, pending_output)

        if self.layer_config.layer_type == LayerType.RASTER:
            assert isinstance(self.layer_config, RasterLayerConfig)
            raster_dir = window.get_raster_dir(
                self.output_layer, self.layer_config.band_sets[0].bands
            )
            assert isinstance(self.format, RasterFormat)
            self.format.encode_raster(
                raster_dir, window.projection, window.bounds, merged_output
            )

        elif self.layer_config.layer_type == LayerType.VECTOR:
            layer_dir = window.get_layer_dir(self.output_layer)
            assert isinstance(self.format, VectorFormat)
            self.format.encode_vector(layer_dir, merged_output)

        window.mark_layer_completed(self.output_layer)
