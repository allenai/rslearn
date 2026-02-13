"""Integration tests for ClassificationTask and ClassificationHead."""

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
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat

CLASSES = ["cat", "dog", "bird"]
PROPERTY_NAME = "category"


@pytest.fixture
def classification_dataset(tmp_path: pathlib.Path) -> Dataset:
    """Create a sample dataset with raster input, vector targets, and vector output.

    The dataset has:
    - An "image" layer with a single-band raster image (input)
    - A "targets" layer with vector classification labels
    - A "predictions" layer to store the output classification predictions
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
                "type": "vector",
            },
            "predictions": {
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
        RasterArray(chw_array=image),
    )
    window.mark_layer_completed(layer_name)

    # Add vector classification target (a point feature with class property).
    target_class = CLASSES[0]
    feature = Feature(
        STGeometry(
            WGS84_PROJECTION,
            shapely.Point(16, 16),  # Center of the window
            None,
        ),
        {PROPERTY_NAME: target_class},
    )
    layer_name = "targets"
    layer_dir = window.get_layer_dir(layer_name)
    GeojsonVectorFormat().encode_vector(layer_dir, [feature])
    window.mark_layer_completed(layer_name)

    return dataset


@pytest.fixture
def classification_model_config() -> dict:
    """Model config using Swin-Tiny for classification."""
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
                                    # No output_layers means classification mode,
                                    # outputs FeatureVector directly.
                                    "num_outputs": len(CLASSES),
                                },
                            }
                        ],
                        "decoder": [
                            {
                                "class_path": "rslearn.train.tasks.classification.ClassificationHead",
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
                        "data_type": "vector",
                        "layers": ["targets"],
                    },
                },
                "task": {
                    "class_path": "rslearn.train.tasks.classification.ClassificationTask",
                    "init_args": {
                        "property_name": PROPERTY_NAME,
                        "classes": CLASSES,
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


def test_classification_prediction_writes_to_dataset(
    classification_dataset: Dataset,
    classification_model_config: dict,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that classification model config can be parsed and predictions are written.

    This test:
    1. Parses a model config via RslearnLightningCLI
    2. Runs prediction with predict subcommand.
    3. Verifies predictions are written to the dataset as vector data
    """
    init_jsonargparse()

    # Write config to file.
    tmp_fname = tmp_path / "classification_model.yaml"
    with tmp_fname.open("w") as f:
        json.dump(classification_model_config, f)

    monkeypatch.setenv("DATASET_PATH", str(classification_dataset.path))
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
    window = classification_dataset.load_windows()[0]
    assert window.is_layer_completed("predictions")
    features = GeojsonVectorFormat().decode_vector(
        window.get_layer_dir("predictions"),
        window.projection,
        window.bounds,
    )
    assert len(features) == 1
    # Verify the predicted class is one of the valid classes.
    assert features[0].properties[PROPERTY_NAME] in CLASSES


def test_classification_training(
    classification_dataset: Dataset,
    classification_model_config: dict,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that classification model can be trained for one epoch.

    This test:
    1. Parses a model config via RslearnLightningCLI
    2. Runs training with fit subcommand for 1 epoch.
    3. Verifies training completed successfully
    """
    init_jsonargparse()

    # Write config to file.
    tmp_fname = tmp_path / "classification_training.yaml"
    with tmp_fname.open("w") as f:
        json.dump(classification_model_config, f)

    monkeypatch.setenv("DATASET_PATH", str(classification_dataset.path))
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
