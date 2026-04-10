"""Test rslearn.train.prediction_writer."""

import json
import pathlib
from typing import Any
from unittest.mock import Mock

import numpy as np
import numpy.typing as npt
import torch
from lightning.pytorch import Trainer
from torchmetrics import MetricCollection
from upath import UPath

from rslearn.config import BandSetConfig, DType, LayerConfig, LayerType, StorageConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.dataset.storage.file import FileWindowStorage
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.train.model_context import ModelOutput, RasterImage, SampleMetadata
from rslearn.train.prediction_writer import (
    PendingCropOutput,
    RasterMerger,
    RslearnWriter,
    WeightedRasterMerger,
)
from rslearn.train.tasks.segmentation import SegmentationTask
from rslearn.train.tasks.task import Task
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import Projection


class MockDictionaryTask(Task):
    """Mock task that returns dictionary outputs for testing selector functionality."""

    def __init__(self, num_classes: int = 2):
        """Initialize mock task.

        Args:
            num_classes: number of classes for segmentation
        """
        self.num_classes = num_classes

    def process_inputs(
        self,
        raw_inputs: dict[str, RasterImage | torch.Tensor | list[Feature]],
        metadata: SampleMetadata,
        load_targets: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Process inputs (not used in prediction writer tests)."""
        return {}, {}

    def process_output(
        self, raw_output: Any, metadata: SampleMetadata
    ) -> dict[str, npt.NDArray[Any]]:
        """Process output into dictionary format to test selector.

        Args:
            raw_output: the raw tensor output from model
            metadata: metadata dict

        Returns:
            Dictionary with 'segment' key containing segmentation data and other keys
        """
        raw_output_np = raw_output.cpu().numpy()
        # Create segmentation output (argmax over classes)
        classes = raw_output_np.argmax(axis=0).astype(np.uint8)
        segmentation_output = classes[None, :, :]  # Add channel dimension

        # Return dictionary with multiple keys to test selector
        return {
            "segment": segmentation_output,
            "probabilities": raw_output_np,
            "other_data": np.zeros_like(segmentation_output),
        }

    def visualize(
        self,
        input_dict: dict[str, Any],
        target_dict: dict[str, Any] | None,
        output: Any,
    ) -> dict[str, npt.NDArray[Any]]:
        """Visualize (not used in tests)."""
        return {}

    def get_metrics(self) -> MetricCollection:
        """Get metrics (not used in tests)."""
        return MetricCollection({})


class TestRasterMerger:
    """Unit tests for RasterMerger."""

    def test_merge_no_padding(self, tmp_path: pathlib.Path) -> None:
        """Verify patches are merged when no padding is used.

        We make four 3x3 patches to cover a 4x4 window.
        """
        storage = FileWindowStorage(tmp_path)
        window = Window(
            storage=storage,
            group="fake",
            name="fake",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 4, 4),
            time_range=None,
        )
        outputs = [
            PendingCropOutput(
                bounds=(0, 0, 3, 3),
                output=0 * np.ones((1, 3, 3), dtype=np.uint8),
            ),
            PendingCropOutput(
                bounds=(0, 3, 3, 6),
                output=1 * np.ones((1, 3, 3), dtype=np.uint8),
            ),
            PendingCropOutput(
                bounds=(3, 0, 6, 3),
                output=2 * np.ones((1, 3, 3), dtype=np.uint8),
            ),
            PendingCropOutput(
                bounds=(3, 3, 6, 6),
                output=3 * np.ones((1, 3, 3), dtype=np.uint8),
            ),
        ]
        merged = RasterMerger().merge(
            window,
            outputs,
            LayerConfig(
                type=LayerType.RASTER,
                band_sets=[BandSetConfig(bands=["output"], dtype=DType.UINT8)],
            ),
        )
        assert merged.shape == (1, 4, 4)
        assert merged.dtype == np.uint8
        # The patches were disjoint, so we just check that those portions of the merged
        # image have the right value.
        assert np.all(merged[0, 0:3, 0:3] == 0)
        assert np.all(merged[0, 3:4, 0:3] == 1)
        assert np.all(merged[0, 0:3, 3:4] == 2)
        assert np.all(merged[0, 3, 3] == 3)

    def test_merge_with_overlap(self, tmp_path: pathlib.Path) -> None:
        """Verify merging works with overlap_pixels."""
        storage = FileWindowStorage(tmp_path)
        window = Window(
            storage=storage,
            group="fake",
            name="fake",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 4, 4),
            time_range=None,
        )
        # We make four 3x3 patches:
        # - (0, 0, 3, 3)
        # - (0, 1, 3, 4)
        # - (1, 0, 4, 3)
        # - (1, 1, 4, 4)
        # There are 2 shared pixels between overlapping patches so we set overlap_pixels=2.
        # This means we remove overlap_pixels//2 = 1 pixel from each side.
        outputs = [
            PendingCropOutput(
                bounds=(0, 0, 3, 3),
                output=0 * np.ones((1, 3, 3), dtype=np.int32),
            ),
            PendingCropOutput(
                bounds=(0, 1, 3, 4),
                output=1 * np.ones((1, 3, 3), dtype=np.int32),
            ),
            PendingCropOutput(
                bounds=(1, 0, 4, 3),
                output=2 * np.ones((1, 3, 3), dtype=np.int32),
            ),
            PendingCropOutput(
                bounds=(1, 1, 4, 4),
                output=3 * np.ones((1, 3, 3), dtype=np.int32),
            ),
        ]
        merged = RasterMerger(overlap_pixels=2).merge(
            window,
            outputs,
            LayerConfig(
                type=LayerType.RASTER,
                band_sets=[BandSetConfig(bands=["output"], dtype=DType.INT32)],
            ),
        )
        assert merged.shape == (1, 4, 4)
        assert merged.dtype == np.int32
        # The top-left should use the first patch.
        assert np.all(merged[0, 0:2, 0:2] == 0)
        # The bottom-left should use the second patch.
        assert np.all(merged[0, 2:4, 0:2] == 1)
        # The top-right should use the third patch.
        assert np.all(merged[0, 0:2, 2:4] == 2)
        # And finally the bottom-right should use the fourth patch.
        assert np.all(merged[0, 2:4, 2:4] == 3)

    def test_merge_respect_dtype(self, tmp_path: pathlib.Path) -> None:
        """Verify that merge respects the dtype in BandSetConfig."""
        window = Window(
            storage=FileWindowStorage(tmp_path),
            group="fake",
            name="fake",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 4, 4),
            time_range=None,
        )
        outputs = [
            PendingCropOutput(
                bounds=(0, 0, 4, 4),
                output=0 * np.ones((1, 4, 4), dtype=np.uint8),
            ),
        ]
        merged = RasterMerger().merge(
            window,
            outputs,
            LayerConfig(
                type=LayerType.RASTER,
                band_sets=[BandSetConfig(bands=["output"], dtype=DType.UINT16)],
            ),
        )
        assert merged.shape == (1, 4, 4)
        assert merged.dtype == np.uint16


class TestWeightedRasterMerger:
    """Unit tests for WeightedRasterMerger."""

    def _layer_config(self, dtype: DType = DType.FLOAT32) -> LayerConfig:
        return LayerConfig(
            type=LayerType.RASTER,
            band_sets=[BandSetConfig(bands=["output"], dtype=dtype)],
        )

    def test_single_crop_no_overlap(self, tmp_path: pathlib.Path) -> None:
        """A single crop covering the entire window should be returned as-is."""
        storage = FileWindowStorage(tmp_path)
        window = Window(
            storage=storage,
            group="fake",
            name="fake",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 4, 4),
            time_range=None,
        )
        data = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
        outputs = [PendingCropOutput(bounds=(0, 0, 4, 4), output=data)]
        merged = WeightedRasterMerger(overlap_pixels=0).merge(
            window, outputs, self._layer_config()
        )
        assert merged.shape == (1, 4, 4)
        assert merged.dtype == np.float32
        np.testing.assert_allclose(merged, data)

    def test_non_overlapping_crops(self, tmp_path: pathlib.Path) -> None:
        """Non-overlapping crops should tile without any blending."""
        storage = FileWindowStorage(tmp_path)
        window = Window(
            storage=storage,
            group="fake",
            name="fake",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 4, 4),
            time_range=None,
        )
        outputs = [
            PendingCropOutput(
                bounds=(0, 0, 2, 2),
                output=1.0 * np.ones((1, 2, 2), dtype=np.float32),
            ),
            PendingCropOutput(
                bounds=(2, 0, 4, 2),
                output=2.0 * np.ones((1, 2, 2), dtype=np.float32),
            ),
            PendingCropOutput(
                bounds=(0, 2, 2, 4),
                output=3.0 * np.ones((1, 2, 2), dtype=np.float32),
            ),
            PendingCropOutput(
                bounds=(2, 2, 4, 4),
                output=4.0 * np.ones((1, 2, 2), dtype=np.float32),
            ),
        ]
        merged = WeightedRasterMerger(overlap_pixels=0).merge(
            window, outputs, self._layer_config()
        )
        assert merged.shape == (1, 4, 4)
        np.testing.assert_allclose(merged[0, 0:2, 0:2], 1.0)
        np.testing.assert_allclose(merged[0, 0:2, 2:4], 2.0)
        np.testing.assert_allclose(merged[0, 2:4, 0:2], 3.0)
        np.testing.assert_allclose(merged[0, 2:4, 2:4], 4.0)

    def test_uniform_overlap_unchanged(self, tmp_path: pathlib.Path) -> None:
        """Blending identical values in the overlap zone should yield the same value."""
        storage = FileWindowStorage(tmp_path)
        window = Window(
            storage=storage,
            group="fake",
            name="fake",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 6, 6),
            time_range=None,
        )
        val = 5.0
        outputs = [
            PendingCropOutput(
                bounds=(0, 0, 4, 4),
                output=val * np.ones((1, 4, 4), dtype=np.float32),
            ),
            PendingCropOutput(
                bounds=(2, 0, 6, 4),
                output=val * np.ones((1, 4, 4), dtype=np.float32),
            ),
            PendingCropOutput(
                bounds=(0, 2, 4, 6),
                output=val * np.ones((1, 4, 4), dtype=np.float32),
            ),
            PendingCropOutput(
                bounds=(2, 2, 6, 6),
                output=val * np.ones((1, 4, 4), dtype=np.float32),
            ),
        ]
        merged = WeightedRasterMerger(overlap_pixels=2).merge(
            window, outputs, self._layer_config()
        )
        assert merged.shape == (1, 6, 6)
        np.testing.assert_allclose(merged, val, atol=1e-6)

    def test_overlap_blends_different_values(self, tmp_path: pathlib.Path) -> None:
        """Two crops with different constant values should blend in the overlap zone."""
        storage = FileWindowStorage(tmp_path)
        window = Window(
            storage=storage,
            group="fake",
            name="fake",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 6, 1),
            time_range=None,
        )
        # Two 4-wide crops with 2-pixel overlap in a 6-wide window.
        outputs = [
            PendingCropOutput(
                bounds=(0, 0, 4, 1),
                output=0.0 * np.ones((1, 1, 4), dtype=np.float32),
            ),
            PendingCropOutput(
                bounds=(2, 0, 6, 1),
                output=10.0 * np.ones((1, 1, 4), dtype=np.float32),
            ),
        ]
        merged = WeightedRasterMerger(overlap_pixels=2).merge(
            window, outputs, self._layer_config()
        )
        assert merged.shape == (1, 1, 6)
        # Non-overlap regions should be unblended.
        np.testing.assert_allclose(merged[0, 0, 0:2], 0.0, atol=1e-6)
        np.testing.assert_allclose(merged[0, 0, 4:6], 10.0, atol=1e-6)
        # Overlap zone (cols 2-3) should be between 0 and 10, monotonically increasing.
        assert merged[0, 0, 2] > 0.0
        assert merged[0, 0, 3] > merged[0, 0, 2]
        assert merged[0, 0, 3] < 10.0

    def test_overlap_symmetry(self, tmp_path: pathlib.Path) -> None:
        """Blending should be symmetric: swapping crop values mirrors the result."""
        storage = FileWindowStorage(tmp_path)
        window = Window(
            storage=storage,
            group="fake",
            name="fake",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 6, 1),
            time_range=None,
        )
        outputs_a = [
            PendingCropOutput(
                bounds=(0, 0, 4, 1),
                output=0.0 * np.ones((1, 1, 4), dtype=np.float32),
            ),
            PendingCropOutput(
                bounds=(2, 0, 6, 1),
                output=10.0 * np.ones((1, 1, 4), dtype=np.float32),
            ),
        ]
        outputs_b = [
            PendingCropOutput(
                bounds=(0, 0, 4, 1),
                output=10.0 * np.ones((1, 1, 4), dtype=np.float32),
            ),
            PendingCropOutput(
                bounds=(2, 0, 6, 1),
                output=0.0 * np.ones((1, 1, 4), dtype=np.float32),
            ),
        ]
        cfg = self._layer_config()
        merger = WeightedRasterMerger(overlap_pixels=2)
        merged_a = merger.merge(window, outputs_a, cfg)
        merged_b = merger.merge(window, outputs_b, cfg)
        # The blended values should be mirrors: a + b = 10 everywhere.
        np.testing.assert_allclose(merged_a + merged_b, 10.0, atol=1e-6)

    def test_downsample_factor(self, tmp_path: pathlib.Path) -> None:
        """Output size should be scaled by downsample_factor."""
        storage = FileWindowStorage(tmp_path)
        # Window 8x2 at input resolution → 4x1 at output with downsample_factor=2.
        window = Window(
            storage=storage,
            group="fake",
            name="fake",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 8, 2),
            time_range=None,
        )
        # Two crops with 2-pixel overlap at input (1-pixel at output).
        # Crop 1: input (0,0)-(6,2) → output 3x1, Crop 2: input (4,0)-(8,2) → output 2x1.
        outputs = [
            PendingCropOutput(
                bounds=(0, 0, 6, 2),
                output=1.0 * np.ones((1, 1, 3), dtype=np.float32),
            ),
            PendingCropOutput(
                bounds=(4, 0, 8, 2),
                output=1.0 * np.ones((1, 1, 2), dtype=np.float32),
            ),
        ]
        merged = WeightedRasterMerger(overlap_pixels=1, downsample_factor=2).merge(
            window, outputs, self._layer_config()
        )
        assert merged.shape == (1, 1, 4)
        np.testing.assert_allclose(merged, 1.0, atol=1e-6)

    def test_respects_output_dtype(self, tmp_path: pathlib.Path) -> None:
        """Output dtype should match the layer config."""
        storage = FileWindowStorage(tmp_path)
        window = Window(
            storage=storage,
            group="fake",
            name="fake",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 4, 4),
            time_range=None,
        )
        outputs = [
            PendingCropOutput(
                bounds=(0, 0, 4, 4),
                output=42.0 * np.ones((1, 4, 4), dtype=np.float32),
            ),
        ]
        merged = WeightedRasterMerger(overlap_pixels=0).merge(
            window, outputs, self._layer_config(DType.UINT16)
        )
        assert merged.dtype == np.uint16
        np.testing.assert_array_equal(merged, 42)

    def test_multichannel(self, tmp_path: pathlib.Path) -> None:
        """Blending should work independently per channel."""
        storage = FileWindowStorage(tmp_path)
        window = Window(
            storage=storage,
            group="fake",
            name="fake",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 6, 1),
            time_range=None,
        )
        ch = 3
        left = np.zeros((ch, 1, 4), dtype=np.float32)
        right = np.zeros((ch, 1, 4), dtype=np.float32)
        for c in range(ch):
            left[c] = float(c)
            right[c] = float(c + 10)
        outputs = [
            PendingCropOutput(bounds=(0, 0, 4, 1), output=left),
            PendingCropOutput(bounds=(2, 0, 6, 1), output=right),
        ]
        merged = WeightedRasterMerger(overlap_pixels=2).merge(
            window,
            outputs,
            LayerConfig(
                type=LayerType.RASTER,
                band_sets=[BandSetConfig(bands=["a", "b", "c"], dtype=DType.FLOAT32)],
            ),
        )
        assert merged.shape == (ch, 1, 6)
        for c in range(ch):
            np.testing.assert_allclose(merged[c, 0, 0:2], float(c), atol=1e-6)
            np.testing.assert_allclose(merged[c, 0, 4:6], float(c + 10), atol=1e-6)
            assert merged[c, 0, 2] > float(c)
            assert merged[c, 0, 3] < float(c + 10)


def test_write_raster(tmp_path: pathlib.Path) -> None:
    output_layer_name = "output"
    output_bands = ["value"]

    # Initialize dataset.
    ds_path = UPath(tmp_path)
    ds_config = {
        "layers": {
            output_layer_name: {
                "type": "raster",
                "band_sets": [
                    {
                        "dtype": "uint8",
                        "bands": output_bands,
                    }
                ],
            }
        }
    }
    with (ds_path / "config.json").open("w") as f:
        json.dump(ds_config, f)
    dataset = Dataset(ds_path)

    # Create the window.
    window_name = "default"
    window_group = "default"
    window = Window(
        storage=dataset.storage,
        group=window_group,
        name=window_name,
        projection=Projection(WGS84_PROJECTION.crs, 0.2, 0.2),
        bounds=(0, 0, 1, 1),
        time_range=None,
    )
    window.save()

    # Initialize prediction writer.
    task = SegmentationTask(num_classes=2)
    pl_module = RslearnLightningModule(
        model=torch.nn.Identity(),
        task=task,
    )
    writer = RslearnWriter(
        path=str(tmp_path),
        output_layer=output_layer_name,
    )

    # Write predictions.
    metadata = SampleMetadata(
        window_group=window.group,
        window_name=window.name,
        window_bounds=window.bounds,
        crop_bounds=window.bounds,
        crop_idx=0,
        num_crops_in_window=1,
        time_range=window.time_range,
        projection=window.projection,
        dataset_source=None,
    )
    # batch is (inputs, targets, metadatas) but writer only uses the metadatas.
    batch = ([None], [None], [metadata])
    # output for segmentation task is CHW where C axis contains per-class
    # probabilities.
    output = torch.zeros((2, 5, 5), dtype=torch.float32)
    # Create a mock trainer to satisfy type requirements
    mock_trainer = Mock(spec=Trainer)
    mock_trainer.datamodule = Mock()
    mock_trainer.datamodule.path = UPath(ds_path)
    writer.setup(mock_trainer, pl_module, stage="predict")

    writer.write_on_batch_end(
        trainer=mock_trainer,
        pl_module=pl_module,
        prediction=ModelOutput(outputs=[output], loss_dict={}),
        batch_indices=[0],
        batch=batch,
        batch_idx=0,
        dataloader_idx=0,
    )

    # Ensure the output is written.
    expected_fname = (
        window.get_raster_dir(output_layer_name, output_bands, 0) / "geotiff.tif"
    )
    assert expected_fname.exists()
    assert window.is_layer_completed(output_layer_name)


def test_write_raster_with_custom_output_path(tmp_path: pathlib.Path) -> None:
    """Test RslearnWriter with custom output_path parameter."""
    output_layer_name = "output"
    output_bands = ["value"]

    # Initialize dataset at one location.
    ds_path = UPath(tmp_path / "dataset")
    ds_path.mkdir()
    ds_config = {
        "layers": {
            output_layer_name: {
                "type": "raster",
                "band_sets": [
                    {
                        "dtype": "uint8",
                        "bands": output_bands,
                    }
                ],
            }
        }
    }
    with (ds_path / "config.json").open("w") as f:
        json.dump(ds_config, f)
    dataset = Dataset(ds_path)

    # Create the window in dataset location.
    window_name = "default"
    window_group = "default"
    window = Window(
        storage=dataset.storage,
        group=window_group,
        name=window_name,
        projection=Projection(WGS84_PROJECTION.crs, 0.2, 0.2),
        bounds=(0, 0, 1, 1),
        time_range=None,
    )
    window.save()

    # Use custom output path different from dataset path.
    output_path = tmp_path / "custom_output"
    output_path.mkdir()

    # Initialize prediction writer with custom output_path.
    task = SegmentationTask(num_classes=2)
    pl_module = RslearnLightningModule(
        model=torch.nn.Identity(),
        task=task,
    )
    writer = RslearnWriter(
        path=str(ds_path),
        output_layer=output_layer_name,
        output_path=str(output_path),
    )

    # Write predictions.
    metadata = SampleMetadata(
        window_group=window.group,
        window_name=window.name,
        window_bounds=window.bounds,
        crop_bounds=window.bounds,
        crop_idx=0,
        num_crops_in_window=1,
        time_range=window.time_range,
        projection=window.projection,
        dataset_source=None,
    )
    batch = ([None], [None], [metadata])
    output = torch.zeros((2, 5, 5), dtype=torch.float32)
    # Create a mock trainer to satisfy type requirements
    mock_trainer = Mock(spec=Trainer)
    mock_trainer.datamodule = Mock()
    mock_trainer.datamodule.path = UPath(ds_path)
    writer.setup(mock_trainer, pl_module, stage="predict")

    writer.write_on_batch_end(
        trainer=mock_trainer,
        pl_module=pl_module,
        prediction=ModelOutput(outputs=[output], loss_dict={}),
        batch_indices=[0],
        batch=batch,
        batch_idx=0,
        dataloader_idx=0,
    )

    # Ensure the output is written to the custom output path, not the dataset path.
    custom_storage = FileWindowStorage(UPath(output_path))
    custom_window = Window(
        storage=custom_storage,
        group=window_group,
        name=window_name,
        projection=Projection(WGS84_PROJECTION.crs, 0.2, 0.2),
        bounds=(0, 0, 1, 1),
        time_range=None,
    )
    expected_fname = (
        custom_window.get_raster_dir(output_layer_name, output_bands, 0) / "geotiff.tif"
    )
    assert expected_fname.exists(), "Output should be written to custom output path"
    assert custom_window.is_layer_completed(output_layer_name)

    # Ensure output was NOT written to the original dataset path.
    original_expected_fname = (
        window.get_raster_dir(output_layer_name, output_bands, 0) / "geotiff.tif"
    )
    assert not original_expected_fname.exists(), (
        "Output should not be in original dataset path"
    )


def test_write_raster_with_layer_config(tmp_path: pathlib.Path) -> None:
    """Test RslearnWriter with custom layer_config parameter."""
    output_layer_name = "output"
    output_bands = ["value"]

    # Create custom layer config without needing dataset config.
    layer_config = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[
            BandSetConfig(
                dtype=DType.UINT8,
                bands=output_bands,
                format={
                    "class_path": "rslearn.utils.raster_format.GeotiffRasterFormat"
                },
            )
        ],
    )

    # Use a path where no dataset config exists.
    output_path = UPath(tmp_path / "no_dataset_config")
    output_path.mkdir()

    # Initialize prediction writer with custom layer_config.
    task = SegmentationTask(num_classes=2)
    pl_module = RslearnLightningModule(
        model=torch.nn.Identity(),
        task=task,
    )
    writer = RslearnWriter(
        path=str(tmp_path),  # This path doesn't matter since we're using layer_config
        output_layer=output_layer_name,
        layer_config=layer_config,
        storage_config=StorageConfig(),
        output_path=str(output_path),
    )

    # Create window metadata.
    window_name = "default"
    window_group = "default"
    metadata = SampleMetadata(
        window_group=window_group,
        window_name=window_name,
        window_bounds=(0, 0, 1, 1),
        crop_bounds=(0, 0, 1, 1),
        crop_idx=0,
        num_crops_in_window=1,
        time_range=None,
        projection=Projection(WGS84_PROJECTION.crs, 0.2, 0.2),
        dataset_source=None,
    )

    # Write predictions.
    batch = ([None], [None], [metadata])
    output = torch.zeros((2, 5, 5), dtype=torch.float32)
    # Create a mock trainer to satisfy type requirements
    mock_trainer = Mock(spec=Trainer)
    mock_trainer.datamodule = Mock()
    mock_trainer.datamodule.path = UPath(tmp_path)
    writer.setup(mock_trainer, pl_module, stage="predict")

    writer.write_on_batch_end(
        trainer=mock_trainer,
        pl_module=pl_module,
        prediction=ModelOutput(outputs=[output], loss_dict={}),
        batch_indices=[0],
        batch=batch,
        batch_idx=0,
        dataloader_idx=0,
    )

    # Ensure the output is written using the custom layer config.
    window = Window(
        storage=writer.dataset_storage,
        group=window_group,
        name=window_name,
        projection=Projection(WGS84_PROJECTION.crs, 0.2, 0.2),
        bounds=(0, 0, 1, 1),
        time_range=None,
    )
    expected_fname = (
        window.get_raster_dir(output_layer_name, output_bands, 0) / "geotiff.tif"
    )
    assert expected_fname.exists(), "Output should be written with custom layer config"
    assert window.is_layer_completed(output_layer_name)


def test_selector_with_dictionary_output(tmp_path: pathlib.Path) -> None:
    """Test RslearnWriter selector functionality with dictionary outputs.

    Tests that selector=['segment'] correctly extracts the 'segment' key from
    task outputs and that the codepath in process_output_batch() is covered.
    """
    output_layer_name = "output"
    output_bands = ["value"]

    # Create layer config for raster output
    layer_config = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[
            BandSetConfig(
                dtype=DType.UINT8,
                bands=output_bands,
                format={
                    "class_path": "rslearn.utils.raster_format.GeotiffRasterFormat"
                },
            )
        ],
    )

    # Use custom output path
    output_path = UPath(tmp_path / "selector_test_output")
    output_path.mkdir()

    # Initialize prediction writer with selector=['segment']
    task = MockDictionaryTask(num_classes=3)
    pl_module = RslearnLightningModule(
        model=torch.nn.Identity(),
        task=task,
    )
    writer = RslearnWriter(
        path=str(tmp_path),
        output_layer=output_layer_name,
        selector=["segment"],  # This should extract the 'segment' key
        layer_config=layer_config,
        storage_config=StorageConfig(),
        output_path=str(output_path),
    )

    # Create test metadata
    window_name = "test_window"
    window_group = "test_group"
    metadata = SampleMetadata(
        window_group=window_group,
        window_name=window_name,
        window_bounds=(0, 0, 5, 5),
        crop_bounds=(0, 0, 5, 5),
        crop_idx=0,
        num_crops_in_window=1,
        time_range=None,
        projection=Projection(WGS84_PROJECTION.crs, 0.2, 0.2),
        dataset_source=None,
    )

    # Create model output - 3 classes, 5x5 spatial dimensions
    raw_model_output = torch.zeros((3, 5, 5), dtype=torch.float32)

    # Write predictions through the full pipeline
    batch = ([None], [None], [metadata])
    # Create a mock trainer to satisfy type requirements
    mock_trainer = Mock(spec=Trainer)
    mock_trainer.datamodule = Mock()
    mock_trainer.datamodule.path = UPath(tmp_path)
    writer.setup(mock_trainer, pl_module, stage="predict")

    writer.write_on_batch_end(
        trainer=mock_trainer,
        pl_module=pl_module,
        prediction=ModelOutput(outputs=[raw_model_output], loss_dict={}),
        batch_indices=[0],
        batch=batch,
        batch_idx=0,
        dataloader_idx=0,
    )

    # Verify the output was written to the correct location
    window = Window(
        storage=writer.dataset_storage,
        group=window_group,
        name=window_name,
        projection=Projection(WGS84_PROJECTION.crs, 0.2, 0.2),
        bounds=(0, 0, 5, 5),
        time_range=None,
    )
    expected_fname = (
        window.get_raster_dir(output_layer_name, output_bands, 0) / "geotiff.tif"
    )
    assert expected_fname.exists(), "Output should be written with selector extraction"
    assert window.is_layer_completed(output_layer_name)


def test_selector_with_nested_dictionary(tmp_path: pathlib.Path) -> None:
    """Test RslearnWriter selector with nested dictionary access.

    Tests selector=['segment', 'data'] for nested dictionary outputs.
    """

    # Mock task that returns nested dictionary
    class MockNestedTask(Task):
        def process_inputs(
            self,
            raw_inputs: dict[str, RasterImage | torch.Tensor | list[Feature]],
            metadata: SampleMetadata,
            load_targets: bool = True,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            return {}, {}

        def process_output(
            self, raw_output: Any, metadata: SampleMetadata
        ) -> dict[str, Any]:
            raw_output_np = raw_output.cpu().numpy()
            classes = raw_output_np.argmax(axis=0).astype(np.uint8)
            segmentation_output = classes[None, :, :]

            return {
                "segment": {
                    "data": segmentation_output,
                    "confidence": np.ones_like(segmentation_output, dtype=np.float32),
                },
                "other": {"info": "unused"},
            }

        def visualize(
            self,
            input_dict: dict[str, Any],
            target_dict: dict[str, Any] | None,
            output: Any,
        ) -> dict[str, npt.NDArray[Any]]:
            return {}

        def get_metrics(self) -> MetricCollection:
            from torchmetrics import MetricCollection

            return MetricCollection({})

    output_layer_name = "nested_output"
    output_bands = ["value"]

    layer_config = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[
            BandSetConfig(
                dtype=DType.UINT8,
                bands=output_bands,
                format={
                    "class_path": "rslearn.utils.raster_format.GeotiffRasterFormat"
                },
            )
        ],
    )

    output_path = UPath(tmp_path / "nested_selector_test")
    output_path.mkdir()

    # Test nested selector
    task = MockNestedTask()
    pl_module = RslearnLightningModule(
        model=torch.nn.Identity(),
        task=task,
    )
    writer = RslearnWriter(
        path=str(tmp_path),
        output_layer=output_layer_name,
        selector=["segment", "data"],  # Should extract output["segment"]["data"]
        layer_config=layer_config,
        storage_config=StorageConfig(),
        output_path=str(output_path),
    )

    # Create metadata and test data
    metadata = SampleMetadata(
        window_group="test_group",
        window_name="nested_test",
        window_bounds=(0, 0, 3, 3),
        crop_bounds=(0, 0, 3, 3),
        crop_idx=0,
        num_crops_in_window=1,
        time_range=None,
        projection=Projection(WGS84_PROJECTION.crs, 0.2, 0.2),
        dataset_source=None,
    )

    # Create simple model output
    raw_model_output = torch.zeros((2, 3, 3), dtype=torch.float32)
    raw_model_output[1, :, :] = 1.0  # All pixels should be class 1

    batch = ([None], [None], [metadata])
    # Create a mock trainer to satisfy type requirements
    mock_trainer = Mock(spec=Trainer)
    mock_trainer.datamodule = Mock()
    mock_trainer.datamodule.path = UPath(tmp_path)
    writer.setup(mock_trainer, pl_module, stage="predict")

    writer.write_on_batch_end(
        trainer=mock_trainer,
        pl_module=pl_module,
        prediction=ModelOutput(outputs=[raw_model_output], loss_dict={}),
        batch_indices=[0],
        batch=batch,
        batch_idx=0,
        dataloader_idx=0,
    )

    # Verify output was written successfully
    window = Window(
        storage=writer.dataset_storage,
        group="test_group",
        name="nested_test",
        projection=Projection(WGS84_PROJECTION.crs, 0.2, 0.2),
        bounds=(0, 0, 3, 3),
        time_range=None,
    )
    expected_fname = (
        window.get_raster_dir(output_layer_name, output_bands, 0) / "geotiff.tif"
    )
    assert expected_fname.exists(), "Nested selector should successfully write output"
    assert window.is_layer_completed(output_layer_name)


def test_write_raster_with_path_from_datamodule(tmp_path: pathlib.Path) -> None:
    """Test that RslearnWriter resolves path from trainer.datamodule when path=None."""
    output_layer_name = "output"
    output_bands = ["value"]

    ds_path = UPath(tmp_path)
    ds_config = {
        "layers": {
            output_layer_name: {
                "type": "raster",
                "band_sets": [{"dtype": "uint8", "bands": output_bands}],
            }
        }
    }
    with (ds_path / "config.json").open("w") as f:
        json.dump(ds_config, f)
    dataset = Dataset(ds_path)

    window_name = "default"
    window_group = "default"
    window = Window(
        storage=dataset.storage,
        group=window_group,
        name=window_name,
        projection=Projection(WGS84_PROJECTION.crs, 0.2, 0.2),
        bounds=(0, 0, 1, 1),
        time_range=None,
    )
    window.save()

    task = SegmentationTask(num_classes=2)
    pl_module = RslearnLightningModule(
        model=torch.nn.Identity(),
        task=task,
    )

    # Create writer without path — it should resolve from the datamodule.
    writer = RslearnWriter(output_layer=output_layer_name)

    metadata = SampleMetadata(
        window_group=window.group,
        window_name=window.name,
        window_bounds=window.bounds,
        crop_bounds=window.bounds,
        crop_idx=0,
        num_crops_in_window=1,
        time_range=window.time_range,
        projection=window.projection,
        dataset_source=None,
    )
    batch = ([None], [None], [metadata])
    output = torch.zeros((2, 5, 5), dtype=torch.float32)

    mock_trainer = Mock(spec=Trainer)
    mock_trainer.datamodule = Mock()
    mock_trainer.datamodule.path = ds_path

    writer.setup(mock_trainer, pl_module, stage="predict")

    writer.write_on_batch_end(
        trainer=mock_trainer,
        pl_module=pl_module,
        prediction=ModelOutput(outputs=[output], loss_dict={}),
        batch_indices=[0],
        batch=batch,
        batch_idx=0,
        dataloader_idx=0,
    )

    expected_fname = (
        window.get_raster_dir(output_layer_name, output_bands, 0) / "geotiff.tif"
    )
    assert expected_fname.exists()
    assert window.is_layer_completed(output_layer_name)
