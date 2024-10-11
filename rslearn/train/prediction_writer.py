"""rslearn PredictionWriter implementation."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from upath import UPath

from rslearn.config import LayerType, RasterFormatConfig
from rslearn.dataset import Dataset
from rslearn.utils.array import copy_spatial_array
from rslearn.utils.raster_format import load_raster_format
from rslearn.utils.vector_format import load_vector_format

from .lightning_module import RslearnLightningModule


class PatchPredictionMerger:
    """Base class for merging predictions from multiple patches."""

    def merge(
        self, outputs: Sequence[Any], metadatas: Sequence[Any]
    ) -> tuple[Sequence[Any], Sequence[Any]]:
        """Merge the outputs and metadatas.

        Args:
            outputs: the outputs to process.
            metadatas: the metadatas to process.

        Returns:
            the merged outputs and metadatas.
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
        merger: type[PatchPredictionMerger] | None = None,
        merger_args: dict[str, Any] | None = None,
    ):
        """Create a new RslearnWriter.

        Args:
            path: the dataset root directory.
            output_layer: which layer to write the outputs under.
            path_options: additional options for path to pass to fsspec
            selector: keys to access the desired output in the output dict if needed.
            merger: merger to use to merge outputs from overlapped patches.
            merger_args: arguments to pass to the merger.
        """
        super().__init__(write_interval="batch")
        self.output_layer = output_layer
        self.selector = selector
        self.path = UPath(path, **path_options)
        self.dataset = Dataset(self.path)
        self.layer_config = self.dataset.layers[self.output_layer]

        # # Check if we need to apply a merger to the outputs
        # overlap_ratio = self.dataset.split_config.overlap_ratio
        # if overlap_ratio is not None and (0 < overlap_ratio < 1):
        #     self.merger = merger
        #     self.merger_args = merger_args
        self.merger = merger
        self.merger_args = merger_args

        if self.layer_config.layer_type == LayerType.RASTER:
            band_cfg = self.layer_config.band_sets[0]
            self.format = load_raster_format(
                RasterFormatConfig(band_cfg.format["name"], band_cfg.format)
            )
        elif self.layer_config.layer_type == LayerType.VECTOR:
            self.format = load_vector_format(self.layer_config.format)
        else:
            raise ValueError(f"invalid layer type {self.layer_config.layer_type}")

        # Map from window name to pending data to write.
        # This is used when windows are split up into patches, so the data from all the
        # patches of each window need to be reconstituted.
        self.pending_outputs = {}

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Sequence[Any],
        batch_indices: Sequence[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        """Write a batch of predictions into the rslearn dataset.

        Args:
            trainer: the trainer.
            pl_module: the LightningModule.
            prediction: the prediction to write.
            batch_indices: batch indices.
            batch: the batch that was input to the model.
            batch_idx: the batch index.
            dataloader_idx: the index in the dataloader.
        """
        assert isinstance(pl_module, RslearnLightningModule)
        metadatas = batch[2]
        outputs = [
            pl_module.task.process_output(output, metadata)
            for output, metadata in zip(prediction, metadatas)
        ]

        for output, metadata in zip(outputs, metadatas):
            for k in self.selector:
                output = output[k]

            window_name = metadata["window_name"]
            cur_bounds = metadata["bounds"]
            window_bounds = metadata["window_bounds"]

            if self.layer_config.layer_type == LayerType.RASTER:
                if window_name not in self.pending_outputs:
                    self.pending_outputs[window_name] = np.zeros(
                        (
                            output.shape[0],
                            window_bounds[3] - window_bounds[1],
                            window_bounds[2] - window_bounds[0],
                        ),
                        dtype=output.dtype,
                    )

                # Use copy_spatial_array to handle the copy since, when using patches,
                # the last column/row of outputs might extend beyond the bounds of the
                # window.
                copy_spatial_array(
                    src=output,
                    dst=self.pending_outputs[window_name],
                    src_offset=(cur_bounds[0], cur_bounds[1]),
                    dst_offset=(window_bounds[0], window_bounds[1]),
                )

            elif self.layer_config.layer_type == LayerType.VECTOR:
                if window_name not in self.pending_outputs:
                    self.pending_outputs[window_name] = []

                self.pending_outputs[window_name].extend(output)

            if metadata["patch_idx"] < metadata["num_patches"] - 1:
                continue

            pending_output = self.pending_outputs[window_name]
            del self.pending_outputs[window_name]

            # This is the last patch so it's time to merge outputs from overlapped patches
            if self.merger is not None:
                prediction_merger = self.merger(**self.merger_args)
                pending_output = prediction_merger.merge(pending_output)

            # This is the last patch so it's time to write it.
            layer_dir = (
                self.dataset.path
                / "windows"
                / metadata["group"]
                / window_name
                / "layers"
                / self.output_layer
            )

            if self.layer_config.layer_type == LayerType.RASTER:
                band_dir = layer_dir / "_".join(self.layer_config.band_sets[0].bands)
                self.format.encode_raster(
                    band_dir, metadata["projection"], window_bounds, pending_output
                )

            elif self.layer_config.layer_type == LayerType.VECTOR:
                self.format.encode_vector(
                    layer_dir, metadata["projection"], pending_output
                )
