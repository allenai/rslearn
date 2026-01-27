"""Integration tests for MultiTask."""

import json
import pathlib

import numpy as np
import pytest
import shapely
from upath import UPath

from rslearn.arg_parser import RslearnArgumentParser
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.lightning_cli import RslearnLightningCLI
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import STGeometry
from rslearn.utils.jsonargparse import init_jsonargparse
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat

REGRESSION_PROPERTY = "score"
CLASSIFICATION_PROPERTY = "category"
CLASSES = ["low", "medium", "high"]
NUM_CLASSES = len(CLASSES)


@pytest.fixture
def multi_task_dataset(tmp_path: pathlib.Path) -> Dataset:
    """Create a sample dataset with raster input and vector targets for two tasks.

    The dataset has:
    - An "image" layer with a single-band raster image (input)
    - A "regression_targets" layer with vector regression labels
    - A "classification_targets" layer with vector classification labels
    - A "regression_predictions" layer to store regression predictions
    - A "classification_predictions" layer to store classification predictions
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
            "regression_targets": {
                "type": "vector",
            },
            "classification_targets": {
                "type": "vector",
            },
            "regression_predictions": {
                "type": "vector",
            },
            "classification_predictions": {
                "type": "vector",
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
        image,
    )
    window.mark_layer_completed(layer_name)

    # Add vector regression target (a point feature with score property).
    target_value = 0.75
    feature = Feature(
        STGeometry(
            WGS84_PROJECTION,
            shapely.Point(16, 16),  # Center of the window
            None,
        ),
        {REGRESSION_PROPERTY: target_value},
    )
    layer_name = "regression_targets"
    layer_dir = window.get_layer_dir(layer_name)
    GeojsonVectorFormat().encode_vector(layer_dir, [feature])
    window.mark_layer_completed(layer_name)

    # Add vector classification target (a point feature with category property).
    feature = Feature(
        STGeometry(
            WGS84_PROJECTION,
            shapely.Point(16, 16),  # Center of the window
            None,
        ),
        {CLASSIFICATION_PROPERTY: "medium"},
    )
    layer_name = "classification_targets"
    layer_dir = window.get_layer_dir(layer_name)
    GeojsonVectorFormat().encode_vector(layer_dir, [feature])
    window.mark_layer_completed(layer_name)

    return dataset


@pytest.fixture
def multi_task_model_config() -> dict:
    """Model config using Swin-Tiny for multi-task learning with regression and classification."""
    return {
        "model": {
            "class_path": "rslearn.train.lightning_module.RslearnLightningModule",
            "init_args": {
                "model": {
                    "class_path": "rslearn.models.multitask.MultiTaskModel",
                    "init_args": {
                        "encoder": [
                            {
                                "class_path": "rslearn.models.swin.Swin",
                                "init_args": {
                                    "input_channels": 1,
                                    "arch": "swin_v2_t",
                                    # Output feature maps (last layer has 768 channels).
                                    "output_layers": [7],
                                },
                            }
                        ],
                        "decoders": {
                            "regression": [
                                {
                                    # Pool and project to 1 output for regression.
                                    "class_path": "rslearn.models.pooling_decoder.PoolingDecoder",
                                    "init_args": {
                                        "in_channels": 768,
                                        "out_channels": 1,
                                    },
                                },
                                {
                                    "class_path": "rslearn.train.tasks.regression.RegressionHead",
                                },
                            ],
                            "classification": [
                                {
                                    # Pool and project to num_classes outputs for classification.
                                    "class_path": "rslearn.models.pooling_decoder.PoolingDecoder",
                                    "init_args": {
                                        "in_channels": 768,
                                        "out_channels": NUM_CLASSES,
                                    },
                                },
                                {
                                    "class_path": "rslearn.train.tasks.classification.ClassificationHead",
                                },
                            ],
                        },
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
                    "regression_targets": {
                        "data_type": "vector",
                        "layers": ["regression_targets"],
                    },
                    "classification_targets": {
                        "data_type": "vector",
                        "layers": ["classification_targets"],
                    },
                },
                "task": {
                    "class_path": "rslearn.train.tasks.multi_task.MultiTask",
                    "init_args": {
                        "tasks": {
                            "regression": {
                                "class_path": "rslearn.train.tasks.regression.RegressionTask",
                                "init_args": {
                                    "property_name": REGRESSION_PROPERTY,
                                },
                            },
                            "classification": {
                                "class_path": "rslearn.train.tasks.classification.ClassificationTask",
                                "init_args": {
                                    "property_name": CLASSIFICATION_PROPERTY,
                                    "classes": CLASSES,
                                },
                            },
                        },
                        "input_mapping": {
                            "regression": {
                                "regression_targets": "targets",
                            },
                            "classification": {
                                "classification_targets": "targets",
                            },
                        },
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
                        "output_layer": "regression_predictions",
                        # Use selector to pick the regression task output.
                        "selector": ["regression"],
                    },
                },
                {
                    "class_path": "rslearn.train.prediction_writer.RslearnWriter",
                    "init_args": {
                        "path": "${DATASET_PATH}",
                        "output_layer": "classification_predictions",
                        # Use selector to pick the classification task output.
                        "selector": ["classification"],
                    },
                },
            ],
        },
    }


def test_multi_task_training(
    multi_task_dataset: Dataset,
    multi_task_model_config: dict,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that multi-task model can be trained for one epoch.

    This test:
    1. Parses a model config via RslearnLightningCLI
    2. Runs training with fit subcommand for 1 epoch.
    3. Verifies training completed successfully
    """
    init_jsonargparse()

    # Write config to file.
    tmp_fname = tmp_path / "multi_task_training.yaml"
    with tmp_fname.open("w") as f:
        json.dump(multi_task_model_config, f)

    monkeypatch.setenv("DATASET_PATH", str(multi_task_dataset.path))
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


def test_multi_task_prediction_writes_to_dataset(
    multi_task_dataset: Dataset,
    multi_task_model_config: dict,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that multi-task model predictions can be written for both tasks.

    This test:
    1. Parses a model config via RslearnLightningCLI
    2. Runs prediction with predict subcommand using two RslearnWriter callbacks.
    3. Verifies both regression and classification predictions are written to the dataset
    """
    init_jsonargparse()

    # Write config to file.
    tmp_fname = tmp_path / "multi_task_prediction.yaml"
    with tmp_fname.open("w") as f:
        json.dump(multi_task_model_config, f)

    monkeypatch.setenv("DATASET_PATH", str(multi_task_dataset.path))
    RslearnLightningCLI(
        model_class=RslearnLightningModule,
        datamodule_class=RslearnDataModule,
        args=["predict", "--config", str(tmp_fname)],
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        parser_class=RslearnArgumentParser,
    )

    # Verify that regression predictions were written to the dataset.
    window = multi_task_dataset.load_windows()[0]
    assert window.is_layer_completed("regression_predictions")
    features = GeojsonVectorFormat().decode_vector(
        window.get_layer_dir("regression_predictions"),
        window.projection,
        window.bounds,
    )
    assert len(features) == 1
    # Verify the predicted value is a number.
    assert isinstance(features[0].properties[REGRESSION_PROPERTY], float)

    # Verify that classification predictions were written to the dataset.
    assert window.is_layer_completed("classification_predictions")
    features = GeojsonVectorFormat().decode_vector(
        window.get_layer_dir("classification_predictions"),
        window.projection,
        window.bounds,
    )
    assert len(features) == 1
    # Verify the predicted class is one of the valid classes.
    assert features[0].properties[CLASSIFICATION_PROPERTY] in CLASSES
