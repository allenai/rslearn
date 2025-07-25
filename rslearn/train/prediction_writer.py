"""rslearn PredictionWriter implementation."""

from collections.abc import Sequence
from typing import Any, overload

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
from rslearn.utils.raster_format import load_raster_format
from rslearn.utils.vector_format import load_vector_format

from .lightning_module import RslearnLightningModule
from .tasks.task import Task


class PatchPredictionMerger:
    """Base class for merging predictions from multiple patches."""

    @overload
    def merge(self, outputs: list[Feature]) -> list[Feature]:
        ...

    @overload
    def merge(self, outputs: npt.NDArray[Any]) -> npt.NDArray[Any]:
        ...

    def merge(self, outputs: list[Feature] | npt.NDArray[Any]) -> list[Feature] | npt.NDArray[Any]:
        """Merge the outputs.

        Args:
            outputs: the outputs to process.

        Returns:
            the merged outputs.
        """
        raise NotImplementedError


class RslearnWriter(BasePredictionWriter):
    """A writer that writes predictions back into the rslearn dataset.

    The predictions are stored in a specified output layer, which must not exist yet
    for each window being processed.
    """

    def __init__(
        self,
        path: str,
        output_layer: str,
        path_options: dict[str, Any] = {},
        selector: list[str] = [],
        merger: PatchPredictionMerger | None = None,
    ):
        """Create a new RslearnWriter.

        Args:
            path: the dataset root directory.
            output_layer: which layer to write the outputs under.
            path_options: additional options for path to pass to fsspec
            selector: keys to access the desired output in the output dict if needed.
                e.g ["key1", "key2"] gets output["key1"]["key2"]
            merger: merger to use to merge outputs from overlapped patches.
        """
        super().__init__(write_interval="batch")
        self.output_layer = output_layer
        self.selector = selector
        self.path = UPath(path, **path_options)
        self.dataset = Dataset(self.path)
        self.layer_config = self.dataset.layers[self.output_layer]
        # TODO: This is a bit of a hack to get the type checker to be happy.
        self.format: Any
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

        self.merger = merger

        # Map from window name to pending data to write.
        # This is used when windows are split up into patches, so the data from all the
        # patches of each window need to be reconstituted.
        self.pending_outputs: dict[str, Any] = {}

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Sequence,
        batch_indices: Sequence,
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
        self.process_output_batch(task, prediction, metadatas)

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

            window = Window(
                path=Window.get_window_root(
                    self.path, metadata["group"], metadata["window_name"]
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
        if self.layer_config.layer_type == LayerType.RASTER:
            if not isinstance(output, np.ndarray):
                raise ValueError("expected output for raster layer to be numpy array")
            self._incorporate_raster_output(window, cur_bounds, output)

        elif self.layer_config.layer_type == LayerType.VECTOR:
            if not isinstance(output, list):
                raise ValueError(
                    "expected output for vector layer to be list of features"
                )
            self._incorporate_vector_output(window, cur_bounds, output)

        if patch_idx < num_patches - 1:
            return

        # This is the last patch so it's time to write it.
        # First get the pending output and clear it.
        pending_output = self.pending_outputs[window.name]
        del self.pending_outputs[window.name]

        # Merge outputs from overlapped patches if merger is set.
        if self.merger is not None:
            pending_output = self.merger.merge(pending_output)

        if self.layer_config.layer_type == LayerType.RASTER:
            assert isinstance(self.layer_config, RasterLayerConfig)
            raster_dir = window.get_raster_dir(
                self.output_layer, self.layer_config.band_sets[0].bands
            )
            self.format.encode_raster(
                raster_dir, window.projection, window.bounds, pending_output
            )

        elif self.layer_config.layer_type == LayerType.VECTOR:
            layer_dir = window.get_layer_dir(self.output_layer)
            self.format.encode_vector(layer_dir, pending_output)

        window.mark_layer_completed(self.output_layer)

    def _incorporate_raster_output(
        self,
        window: Window,
        cur_bounds: PixelBounds,
        output: npt.NDArray,
    ) -> None:
        """Incorporate the partial output into the output for this window.

        Args:
            window: the window this output corresponds to.
            cur_bounds: the bounds within the window that this output covers. This
                could be the entire window, but could also be a patch of the window if
                the window is processed in patches.
            output: the output data.
        """
        if window.name not in self.pending_outputs:
            self.pending_outputs[window.name] = np.zeros(
                (
                    output.shape[0],
                    window.bounds[3] - window.bounds[1],
                    window.bounds[2] - window.bounds[0],
                ),
                dtype=output.dtype,
            )

        # Use copy_spatial_array to handle the copy since, when using patches,
        # the last column/row of outputs might extend beyond the bounds of the
        # window.
        copy_spatial_array(
            src=output,
            dst=self.pending_outputs[window.name],
            src_offset=(cur_bounds[0], cur_bounds[1]),
            dst_offset=(window.bounds[0], window.bounds[1]),
        )

    def _incorporate_vector_output(
        self,
        window: Window,
        cur_bounds: PixelBounds,
        output: list[Feature],
    ) -> None:
        """Incorporate the partial output into the output for this window.

        Args:
            window: the window this output corresponds to.
            cur_bounds: the bounds within the window that this output covers. This
                could be the entire window, but could also be a patch of the window if
                the window is processed in patches.
            output: the output data.
        """
        if window.name not in self.pending_outputs:
            self.pending_outputs[window.name] = []
        self.pending_outputs[window.name].extend(output)
