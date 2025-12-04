"""Test rslearn.train.prediction_writer."""

import json
import pathlib
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from lightning.pytorch import Trainer
from torchmetrics import MetricCollection
from upath import UPath

from rslearn.config import BandSetConfig, DType, LayerConfig, LayerType
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.train.model_context import ModelOutput, SampleMetadata
from rslearn.train.prediction_writer import (
    PendingPatchOutput,
    RasterMerger,
    RslearnWriter,
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
        raw_inputs: dict[str, torch.Tensor | list[Feature]],
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
        from torchmetrics import MetricCollection

        return MetricCollection({})


class TestRasterMerger:
    """Unit tests for RasterMerger."""

    def test_merge_no_padding(self, tmp_path: pathlib.Path) -> None:
        """Verify patches are merged when no padding is used.

        We make four 3x3 patches to cover a 4x4 window.
        """
        window = Window(
            path=UPath(tmp_path),
            group="fake",
            name="fake",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 4, 4),
            time_range=None,
        )
        outputs = [
            PendingPatchOutput(
                bounds=(0, 0, 3, 3),
                output=0 * np.ones((1, 3, 3), dtype=np.uint8),
            ),
            PendingPatchOutput(
                bounds=(0, 3, 3, 6),
                output=1 * np.ones((1, 3, 3), dtype=np.uint8),
            ),
            PendingPatchOutput(
                bounds=(3, 0, 6, 3),
                output=2 * np.ones((1, 3, 3), dtype=np.uint8),
            ),
            PendingPatchOutput(
                bounds=(3, 3, 6, 6),
                output=3 * np.ones((1, 3, 3), dtype=np.uint8),
            ),
        ]
        merged = RasterMerger().merge(window, outputs)
        assert merged.shape == (1, 4, 4)
        assert merged.dtype == np.uint8
        # The patches were disjoint, so we just check that those portions of the merged
        # image have the right value.
        assert np.all(merged[0, 0:3, 0:3] == 0)
        assert np.all(merged[0, 3:4, 0:3] == 1)
        assert np.all(merged[0, 0:3, 3:4] == 2)
        assert np.all(merged[0, 3, 3] == 3)

    def test_merge_with_padding(self, tmp_path: pathlib.Path) -> None:
        """Verify merging works with padding."""
        window = Window(
            path=UPath(tmp_path),
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
        # There are 2 shared pixels between overlapping patches so we set padding=1.
        outputs = [
            PendingPatchOutput(
                bounds=(0, 0, 3, 3),
                output=0 * np.ones((1, 3, 3), dtype=np.int32),
            ),
            PendingPatchOutput(
                bounds=(0, 1, 3, 4),
                output=1 * np.ones((1, 3, 3), dtype=np.int32),
            ),
            PendingPatchOutput(
                bounds=(1, 0, 4, 3),
                output=2 * np.ones((1, 3, 3), dtype=np.int32),
            ),
            PendingPatchOutput(
                bounds=(1, 1, 4, 4),
                output=3 * np.ones((1, 3, 3), dtype=np.int32),
            ),
        ]
        merged = RasterMerger(padding=1).merge(window, outputs)
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

    # Create the window.
    window_name = "default"
    window_group = "default"
    window = Window(
        path=Window.get_window_root(ds_path, window_name, window_group),
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
        patch_bounds=window.bounds,
        patch_idx=0,
        num_patches_in_window=1,
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
    from unittest.mock import Mock

    mock_trainer = Mock(spec=Trainer)

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

    # Create the window in dataset location.
    window_name = "default"
    window_group = "default"
    window = Window(
        path=Window.get_window_root(ds_path, window_name, window_group),
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
        patch_bounds=window.bounds,
        patch_idx=0,
        num_patches_in_window=1,
        time_range=window.time_range,
        projection=window.projection,
        dataset_source=None,
    )
    batch = ([None], [None], [metadata])
    output = torch.zeros((2, 5, 5), dtype=torch.float32)
    # Create a mock trainer to satisfy type requirements
    from unittest.mock import Mock

    mock_trainer = Mock(spec=Trainer)

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
    custom_window_path = Window.get_window_root(
        UPath(output_path), window_group, window_name
    )
    custom_window = Window(
        path=custom_window_path,
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
        output_path=str(output_path),
    )

    # Create window metadata.
    window_name = "default"
    window_group = "default"
    metadata = SampleMetadata(
        window_group=window_group,
        window_name=window_name,
        window_bounds=(0, 0, 1, 1),
        patch_bounds=(0, 0, 1, 1),
        patch_idx=0,
        num_patches_in_window=1,
        time_range=None,
        projection=Projection(WGS84_PROJECTION.crs, 0.2, 0.2),
        dataset_source=None,
    )

    # Write predictions.
    batch = ([None], [None], [metadata])
    output = torch.zeros((2, 5, 5), dtype=torch.float32)
    # Create a mock trainer to satisfy type requirements
    from unittest.mock import Mock

    mock_trainer = Mock(spec=Trainer)

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
    window_path = Window.get_window_root(output_path, window_group, window_name)
    window = Window(
        path=window_path,
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
        output_path=str(output_path),
    )

    # Create test metadata
    window_name = "test_window"
    window_group = "test_group"
    metadata = SampleMetadata(
        window_group=window_group,
        window_name=window_name,
        window_bounds=(0, 0, 5, 5),
        patch_bounds=(0, 0, 5, 5),
        patch_idx=0,
        num_patches_in_window=1,
        time_range=None,
        projection=Projection(WGS84_PROJECTION.crs, 0.2, 0.2),
        dataset_source=None,
    )

    # Create model output - 3 classes, 5x5 spatial dimensions
    raw_model_output = torch.zeros((3, 5, 5), dtype=torch.float32)

    # Write predictions through the full pipeline
    batch = ([None], [None], [metadata])
    # Create a mock trainer to satisfy type requirements
    from unittest.mock import Mock

    mock_trainer = Mock(spec=Trainer)

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
    window_path = Window.get_window_root(output_path, window_group, window_name)
    window = Window(
        path=window_path,
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
            raw_inputs: dict[str, torch.Tensor | list[Feature]],
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
        output_path=str(output_path),
    )

    # Create metadata and test data
    metadata = SampleMetadata(
        window_group="test_group",
        window_name="nested_test",
        window_bounds=(0, 0, 3, 3),
        patch_bounds=(0, 0, 3, 3),
        patch_idx=0,
        num_patches_in_window=1,
        time_range=None,
        projection=Projection(WGS84_PROJECTION.crs, 0.2, 0.2),
        dataset_source=None,
    )

    # Create simple model output
    raw_model_output = torch.zeros((2, 3, 3), dtype=torch.float32)
    raw_model_output[1, :, :] = 1.0  # All pixels should be class 1

    batch = ([None], [None], [metadata])
    # Create a mock trainer to satisfy type requirements
    from unittest.mock import Mock

    mock_trainer = Mock(spec=Trainer)

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
    window_path = Window.get_window_root(output_path, "test_group", "nested_test")
    window = Window(
        path=window_path,
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
