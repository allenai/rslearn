"""rslearn PredictionWriter implementation."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

from rslearn.config import LayerType, RasterFormatConfig
from rslearn.dataset import Dataset
from rslearn.utils.array import copy_spatial_array
from rslearn.utils.raster_format import load_raster_format
from rslearn.utils.vector_format import load_vector_format

from .lightning_module import RslearnLightningModule


class RslearnWriter(BasePredictionWriter):
    """A writer that writes predictions back into the rslearn dataset.

    The predictions are stored in a specified output layer, which must not exist yet
    for each window being processed.
    """

    def __init__(self, root_dir: str, output_layer: str, selector: list[str] = []):
        """Create a new RslearnWriter.

        Args:
            root_dir: the dataset root directory.
            output_layer: which layer to write the outputs under.
            selector: keys to access the desired output in the output dict if needed.
        """
        super().__init__(write_interval="batch")
        self.output_layer = output_layer
        self.selector = selector

        self.dataset = Dataset(ds_root=root_dir)
        self.layer_config = self.dataset.layers[self.output_layer]

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
        outputs = [pl_module.task.process_output(output) for output in prediction]

        _, _, metadatas = batch
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

            # This is the last patch so it's time to write it.
            pending_output = self.pending_outputs[window_name]
            del self.pending_outputs[window_name]
            layer_dir = self.dataset.file_api.get_folder(
                "windows", metadata["group"], window_name, "layers", self.output_layer
            )

            if self.layer_config.layer_type == LayerType.RASTER:
                layer_dir = layer_dir.get_folder(
                    "_".join(self.layer_config.band_sets[0].bands)
                )
                self.format.encode_raster(
                    layer_dir, metadata["projection"], window_bounds, pending_output
                )

            elif self.layer_config.layer_type == LayerType.VECTOR:
                self.format.encode_vector(
                    layer_dir, metadata["projection"], pending_output
                )
