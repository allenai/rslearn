"""Integration tests for SegmentationTask and SegmentationHead."""

import json
import pathlib

import numpy as np
import pytest
from upath import UPath

from rslearn.arg_parser import RslearnArgumentParser
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.lightning_cli import RslearnLightningCLI
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.utils.jsonargparse import init_jsonargparse
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat

NUM_CLASSES = 3


@pytest.fixture
def segmentation_dataset(tmp_path: pathlib.Path) -> Dataset:
    """Create a sample dataset with raster input, targets, and output layers.

    The dataset has:
    - An "image" layer with a single-band raster image (input)
    - A "targets" layer with segmentation class labels
    - A "predictions" layer to store the output segmentation predictions
    """
    ds_path = UPath(tmp_path)

    dataset_config = {
        "layers": {
            "image": {
                "type": "raster",
                "band_sets": [
                    {
                        "dtype": "uint8",
                        "bands": ["band"],
                    }
                ],
            },
            "targets": {
                "type": "raster",
                "band_sets": [
                    {
                        "dtype": "uint8",
                        "bands": ["class"],
                    }
                ],
            },
            "predictions": {
                "type": "raster",
                "band_sets": [
                    {
                        "dtype": "uint8",
                        "bands": ["class"],
                    }
                ],
            },
        },
    }
    ds_path.mkdir(parents=True, exist_ok=True)
    with (ds_path / "config.json").open("w") as f:
        json.dump(dataset_config, f)
    dataset = Dataset(ds_path)

    # Create a window with a 32x32 image (minimum size for Swin).
    window = Window(
        storage=dataset.storage,
        group="default",
        name="default",
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 32, 32),
        time_range=None,
    )
    window.save()

    # Add a simple input image.
    image = np.random.randint(0, 255, size=(1, 32, 32), dtype=np.uint8)
    layer_name = "image"
    layer_dir = window.get_layer_dir(layer_name)
    GeotiffRasterFormat().encode_raster(
        layer_dir / "band",
        window.projection,
        window.bounds,
        RasterArray(chw_array=image),
    )
    window.mark_layer_completed(layer_name)

    # Add segmentation target labels (class IDs from 0 to NUM_CLASSES-1).
    targets = np.random.randint(0, NUM_CLASSES, size=(1, 32, 32), dtype=np.uint8)
    layer_name = "targets"
    layer_dir = window.get_layer_dir(layer_name)
    GeotiffRasterFormat().encode_raster(
        layer_dir / "class",
        window.projection,
        window.bounds,
        RasterArray(chw_array=targets),
    )
    window.mark_layer_completed(layer_name)

    return dataset


@pytest.fixture
def segmentation_model_config() -> dict:
    """Model config using Swin-Tiny for segmentation."""
    return {
        "model": {
            "class_path": "rslearn.train.lightning_module.RslearnLightningModule",
            "init_args": {
                "model": {
                    "class_path": "rslearn.models.singletask.SingleTaskModel",
                    "init_args": {
                        "encoder": [
                            {
                                "class_path": "rslearn.models.swin.Swin",
                                "init_args": {
                                    "input_channels": 1,
                                    "arch": "swin_v2_t",
                                    # Output feature maps instead of classification.
                                    "output_layers": [7],
                                },
                            }
                        ],
                        "decoder": [
                            {
                                # Upsample to match input resolution (32x).
                                "class_path": "rslearn.models.upsample.Upsample",
                                "init_args": {
                                    "scale_factor": 32,
                                },
                            },
                            {
                                # Conv to reduce channels to num_classes.
                                "class_path": "rslearn.models.conv.Conv",
                                "init_args": {
                                    "in_channels": 768,
                                    "out_channels": NUM_CLASSES,
                                    "kernel_size": 1,
                                },
                            },
                            {
                                "class_path": "rslearn.train.tasks.segmentation.SegmentationHead",
                            },
                        ],
                    },
                },
                "optimizer": {"class_path": "rslearn.train.optimizer.AdamW"},
            },
        },
        "data": {
            "class_path": "rslearn.train.data_module.RslearnDataModule",
            "init_args": {
                "path": "${DATASET_PATH}",
                "inputs": {
                    "image": {
                        "data_type": "raster",
                        "layers": ["image"],
                        "bands": ["band"],
                        "passthrough": True,
                        "dtype": "FLOAT32",
                    },
                    "targets": {
                        "data_type": "raster",
                        "layers": ["targets"],
                        "bands": ["class"],
                        "passthrough": True,
                    },
                },
                "task": {
                    "class_path": "rslearn.train.tasks.segmentation.SegmentationTask",
                    "init_args": {
                        "num_classes": NUM_CLASSES,
                    },
                },
                "batch_size": 1,
            },
        },
        "trainer": {
            "max_epochs": 1,
            "callbacks": [
                {
                    "class_path": "rslearn.train.prediction_writer.RslearnWriter",
                    "init_args": {
                        "path": "${DATASET_PATH}",
                        "output_layer": "predictions",
                    },
                }
            ],
        },
    }


def test_segmentation_prediction_writes_to_dataset(
    segmentation_dataset: Dataset,
    segmentation_model_config: dict,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that segmentation model config can be parsed and predictions are written.

    This test:
    1. Parses a model config via RslearnLightningCLI
    2. Runs prediction with predict subcommand.
    3. Verifies predictions are written to the dataset
    """
    init_jsonargparse()

    # Write config to file.
    tmp_fname = tmp_path / "segmentation_model.yaml"
    with tmp_fname.open("w") as f:
        json.dump(segmentation_model_config, f)

    monkeypatch.setenv("DATASET_PATH", str(segmentation_dataset.path))
    RslearnLightningCLI(
        model_class=RslearnLightningModule,
        datamodule_class=RslearnDataModule,
        args=["predict", "--config", str(tmp_fname)],
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        parser_class=RslearnArgumentParser,
    )

    # Verify that predictions were written to the dataset.
    window = segmentation_dataset.load_windows()[0]
    assert window.is_layer_completed("predictions")
    array = (
        GeotiffRasterFormat()
        .decode_raster(
            window.get_raster_dir("predictions", ["class"]),
            window.projection,
            window.bounds,
        )
        .get_chw_array()
    )
    assert array.shape == (1, 32, 32)
    # Verify predictions are valid class IDs.
    assert array.min() >= 0
    assert array.max() < NUM_CLASSES


def test_segmentation_training(
    segmentation_dataset: Dataset,
    segmentation_model_config: dict,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that segmentation model can be trained for one epoch.

    This test:
    1. Parses a model config via RslearnLightningCLI
    2. Runs training with fit subcommand for 1 epoch.
    3. Verifies training completed successfully
    """
    init_jsonargparse()

    # Write config to file.
    tmp_fname = tmp_path / "segmentation_training.yaml"
    with tmp_fname.open("w") as f:
        json.dump(segmentation_model_config, f)

    monkeypatch.setenv("DATASET_PATH", str(segmentation_dataset.path))
    RslearnLightningCLI(
        model_class=RslearnLightningModule,
        datamodule_class=RslearnDataModule,
        args=["fit", "--config", str(tmp_fname)],
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        parser_class=RslearnArgumentParser,
    )

    # If we get here without exception, training completed successfully.
