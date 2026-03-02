"""rslearn PredictionWriter implementation."""

import json
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from upath import UPath

from rslearn.config import (
    DatasetConfig,
    LayerConfig,
    LayerType,
    StorageConfig,
)
from rslearn.dataset import Window
from rslearn.dataset.storage.storage import WindowStorage
from rslearn.log_utils import get_logger
from rslearn.train.model_context import SampleMetadata
from rslearn.utils.array import copy_spatial_array
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import PixelBounds
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import (
    RasterFormat,
    adjust_projection_and_bounds_for_array,
)
from rslearn.utils.vector_format import VectorFormat

from .lightning_module import RslearnLightningModule
from .model_context import ModelOutput
from .tasks.task import Task

logger = get_logger(__name__)


@dataclass
class PendingCropOutput:
    """A crop output that hasn't been merged yet."""

    bounds: PixelBounds
    output: Any


class CropPredictionMerger:
    """Base class for merging predictions from multiple crops."""

    def merge(
        self,
        window: Window,
        outputs: Sequence[PendingCropOutput],
        layer_config: LayerConfig,
    ) -> Any:
        """Merge the outputs.

        Args:
            window: the window we are merging the outputs for.
            outputs: the outputs to process.
            layer_config: the output layer configuration.

        Returns:
            the merged outputs.
        """
        raise NotImplementedError


class VectorMerger(CropPredictionMerger):
    """Merger for vector data that simply concatenates the features."""

    def merge(
        self,
        window: Window,
        outputs: Sequence[PendingCropOutput],
        layer_config: LayerConfig,
    ) -> list[Feature]:
        """Concatenate the vector features."""
        return [feat for output in outputs for feat in output.output]


class RasterMerger(CropPredictionMerger):
    """Merger for raster data that copies the rasters to the output."""

    def __init__(
        self,
        overlap_pixels: int | None = None,
        downsample_factor: int = 1,
        # Deprecated parameter (for backwards compatibility)
        padding: int | None = None,
    ):
        """Create a new RasterMerger.

        Args:
            overlap_pixels: the number of pixels shared between adjacent crops during
                sliding window inference. Half of this overlap is removed from each
                crop during merging (except at window boundaries where the full crop
                is retained).
            downsample_factor: the factor by which the rasters output by the task are
                lower in resolution relative to the window resolution.
            padding: deprecated, use overlap_pixels instead. The old padding value
                equals overlap_pixels // 2.
        """
        # Handle deprecated padding parameter
        if padding is not None:
            warnings.warn(
                "padding is deprecated, use overlap_pixels instead. "
                "Note: overlap_pixels = padding * 2",
                FutureWarning,
                stacklevel=2,
            )
            if overlap_pixels is not None:
                raise ValueError("Cannot specify both padding and overlap_pixels")
            overlap_pixels = padding * 2

        self.overlap_pixels = overlap_pixels
        self.downsample_factor = downsample_factor

    def merge(
        self,
        window: Window,
        outputs: Sequence[PendingCropOutput],
        layer_config: LayerConfig,
    ) -> npt.NDArray:
        """Merge the raster outputs."""
        num_channels = outputs[0].output.shape[0]
        merged_image = np.zeros(
            (
                num_channels,
                (window.bounds[3] - window.bounds[1]) // self.downsample_factor,
                (window.bounds[2] - window.bounds[0]) // self.downsample_factor,
            ),
            dtype=layer_config.band_sets[0].dtype.get_numpy_dtype(),
        )

        # Compute how many pixels to trim from each side.
        # We remove half of the overlap from each side (not at window boundaries).
        trim_pixels = (
            self.overlap_pixels // 2 if self.overlap_pixels is not None else None
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
            # the trim (if set).
            src = output.output
            src_offset = (
                output.bounds[0] // self.downsample_factor,
                output.bounds[1] // self.downsample_factor,
            )
            if trim_pixels is not None and output.bounds[0] != window.bounds[0]:
                src = src[:, :, trim_pixels:]
                src_offset = (src_offset[0] + trim_pixels, src_offset[1])
            if trim_pixels is not None and output.bounds[1] != window.bounds[1]:
                src = src[:, trim_pixels:, :]
                src_offset = (src_offset[0], src_offset[1] + trim_pixels)

            copy_spatial_array(
                src=src,
                dst=merged_image,
                src_offset=src_offset,
                dst_offset=(
                    window.bounds[0] // self.downsample_factor,
                    window.bounds[1] // self.downsample_factor,
                ),
            )

        return merged_image


class RslearnWriter(BasePredictionWriter):
    """A writer that writes predictions back into the rslearn dataset.

    The predictions are stored in a specified output layer, which must not exist yet
    for each window being processed.
    """

    def __init__(
        self,
        output_layer: str,
        path: str | None = None,
        path_options: dict[str, Any] | None = None,
        selector: list[str] | None = None,
        merger: CropPredictionMerger | None = None,
        output_path: str | Path | None = None,
        layer_config: LayerConfig | None = None,
        storage_config: StorageConfig | None = None,
    ):
        """Create a new RslearnWriter.

        Args:
            output_layer: which layer to write the outputs under.
            path: the dataset root directory. Default is None to use the same path as
                the configured data module.
            path_options: additional options for path to pass to fsspec
            selector: keys to access the desired output in the output dict if needed.
                e.g ["key1", "key2"] gets output["key1"]["key2"]
            merger: merger to use to merge outputs from overlapped crops.
            output_path: optional custom path for writing predictions. If provided,
                predictions will be written to this path instead of deriving from dataset path.
            layer_config: optional layer configuration. If provided, this config will be
                used instead of reading from the dataset config, allowing usage without
                requiring dataset config at the output path.
            storage_config: optional storage configuration, needed similar to layer_config
                if there is no dataset config.
        """
        super().__init__(write_interval="batch")
        self.output_layer = output_layer
        self.selector = selector or []

        # Save args for use in self._initialize, which is called from setup function.
        self._path = path
        self._path_options = path_options
        self._output_path = output_path
        self.layer_config = layer_config
        self.storage_config = storage_config
        self.merger = merger

        # Map from window name to pending data to write.
        # This is used when windows are split up into crops, so the data from all the
        # crops of each window need to be reconstituted.
        self.pending_outputs: dict[str, list[PendingCropOutput]] = {}
        self._initialized = False

    def _initialize(self, datamodule_path: UPath) -> None:
        """Initialize storage, format, and merger from the resolved dataset path.

        Args:
            datamodule_path: the UPath configured in the data module.

        Raises:
            ValueError: if already initialized.
        """
        if self._initialized:
            logger.info(
                "_initialize called but RslearnWriter already initialized, skipping"
            )
            return

        # Resolve the dataset path: we use self._path if set, otherwise we use the
        # datamodule_path.
        if self._path is not None:
            ds_upath = UPath(self._path, **self._path_options or {})
        else:
            ds_upath = datamodule_path

        # Resolve the output path. We output to a provided path if set, otherwise we
        # default to writing predictions to the dataset path.
        output_upath = (
            UPath(self._output_path, **self._path_options or {})
            if self._output_path is not None
            else ds_upath
        )

        # Now we can use ds_path and output_upath to set the layer config and dataset
        # storage.
        self._set_layer_config_and_dataset_storage(ds_upath, output_upath)
        assert self.layer_config is not None

        # Determine if we are outputting raster or vector data.
        self.format: RasterFormat | VectorFormat
        if self.layer_config.type == LayerType.RASTER:
            band_cfg = self.layer_config.band_sets[0]
            self.format = band_cfg.instantiate_raster_format()
        elif self.layer_config.type == LayerType.VECTOR:
            self.format = self.layer_config.instantiate_vector_format()
        else:
            raise ValueError(f"invalid layer type {self.layer_config.type}")

        # If the merger was not set, initialize it based on the layer type.
        if self.merger is None:
            if self.layer_config.type == LayerType.RASTER:
                self.merger = RasterMerger()
            elif self.layer_config.type == LayerType.VECTOR:
                self.merger = VectorMerger()

        self._initialized = True

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: str | None = None
    ) -> None:
        """Resolve path and initialize storage/format.

        Args:
            trainer: the trainer.
            pl_module: the LightningModule.
            stage: the stage (fit/validate/test/predict).
        """
        self._initialize(trainer.datamodule.path)

    def _set_layer_config_and_dataset_storage(
        self,
        ds_upath: UPath,
        output_upath: UPath,
    ) -> None:
        """Set the layer config and dataset storage fields.

        This is a helper function for _initialize. self.layer_config and
        self.storage_config should be populated with the argument to __init__.

        If self.layer_config is set, we keep it as is. If self.storage_config is set,
        we use it to instantiate self.storage as a WindowStorage using the output_upath.

        If one of them is not set, we load the dataset config from the ds_upath and use
        it to populate the field(s). Otherwise, we avoid reading the dataset config;
        this way, RslearnWriter can be used with output directories that do not contain
        the dataset config, as long as layer_config and storage_config are both provided.

        Args:
            ds_upath: the dataset path, where a dataset config can be loaded from if
                layer_config or storage_config is not provided.
            output_upath: the output directory, which could be different from the
                dataset path.

        """
        dataset_storage: WindowStorage | None = None

        # Instantiate the WindowStorage from the storage_config if provided.
        if self.storage_config:
            dataset_storage = (
                self.storage_config.instantiate_window_storage_factory().get_storage(
                    output_upath
                )
            )

        if not self.layer_config or not dataset_storage:
            # Need to load dataset config since one of LayerConfig/StorageConfig is missing.
            # We use DatasetConfig.model_validate instead of initializing the Dataset
            # because we want to get a WindowStorage that has the dataset path set to
            # output_upath instead of ds_upath.
            with (ds_upath / "config.json").open() as f:
                dataset_config = DatasetConfig.model_validate(json.load(f))

            if not self.layer_config:
                if self.output_layer not in dataset_config.layers:
                    raise KeyError(
                        f"Output layer '{self.output_layer}' not found in dataset layers."
                    )
                self.layer_config = dataset_config.layers[self.output_layer]

            if not dataset_storage:
                dataset_storage = dataset_config.storage.instantiate_window_storage_factory().get_storage(
                    output_upath
                )

        self.dataset_storage: WindowStorage = dataset_storage

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: ModelOutput,
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
        if not self._initialized:
            raise ValueError(
                "RslearnWriter not initialized; setup() must be called before write_on_batch_end"
            )
        assert isinstance(pl_module, RslearnLightningModule)
        task = pl_module.task
        _, _, metadatas = batch
        self.process_output_batch(task, prediction.outputs, metadatas)

    def process_output_batch(
        self,
        task: Task,
        prediction: Iterable[Any],
        metadatas: Iterable[SampleMetadata],
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
                crops that were processed.
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
                storage=self.dataset_storage,
                group=metadata.window_group,
                name=metadata.window_name,
                projection=metadata.projection,
                bounds=metadata.window_bounds,
                time_range=metadata.time_range,
            )
            self.process_output(
                window,
                metadata.crop_idx,
                metadata.num_crops_in_window,
                metadata.crop_bounds,
                output,
            )

    def process_output(
        self,
        window: Window,
        crop_idx: int,
        num_crops: int,
        cur_bounds: PixelBounds,
        output: npt.NDArray | list[Feature],
    ) -> None:
        """Process one output from the model.

        Args:
            window: the window that the output pertains to.
            crop_idx: the index of this crop for the window.
            num_crops: the total number of crops to be processed for the window.
            cur_bounds: the bounds of the current crop.
            output: the output data.
        """
        # Incorporate the output into our list of pending crop outputs.
        if window.name not in self.pending_outputs:
            self.pending_outputs[window.name] = []
        self.pending_outputs[window.name].append(PendingCropOutput(cur_bounds, output))
        logger.debug(
            f"Stored PendingCropOutput for crop #{crop_idx}/{num_crops} at window {window.name}"
        )

        if crop_idx < num_crops - 1:
            return

        # This is the last crop so it's time to write it.
        # First get the pending output and clear it.
        pending_output = self.pending_outputs[window.name]
        del self.pending_outputs[window.name]

        # Merge outputs from overlapped crops if merger is set.
        logger.debug(f"Merging and writing for window {window.name}")
        assert self.layer_config is not None and self.merger is not None
        merged_output = self.merger.merge(window, pending_output, self.layer_config)

        if self.layer_config.type == LayerType.RASTER:
            raster_dir = window.get_raster_dir(
                self.output_layer, self.layer_config.band_sets[0].bands
            )
            assert isinstance(self.format, RasterFormat)

            # In case the merged_output is at a different resolution than the window,
            # get adjusted projection and bounds for writing it.
            projection, bounds = adjust_projection_and_bounds_for_array(
                window.projection, window.bounds, merged_output
            )
            # Wrap CHW ndarray as CTHW RasterArray.
            raster = RasterArray(chw_array=merged_output)
            self.format.encode_raster(raster_dir, projection, bounds, raster)

        elif self.layer_config.type == LayerType.VECTOR:
            layer_dir = window.get_layer_dir(self.output_layer)
            assert isinstance(self.format, VectorFormat)
            self.format.encode_vector(layer_dir, merged_output)

        window.mark_layer_completed(self.output_layer)
