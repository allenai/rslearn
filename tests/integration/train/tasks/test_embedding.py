"""Integration tests for EmbeddingTask and EmbeddingHead."""

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
from rslearn.utils.raster_format import SingleImageRasterFormat


@pytest.fixture
def embedding_dataset(tmp_path: pathlib.Path) -> Dataset:
    """Create a sample dataset with a raster input and embeddings output layer.

    The dataset has:
    - An "image" layer with a single-band raster image
    - An "embeddings" layer to store the output embeddings
    """
    ds_path = UPath(tmp_path)

    # Dataset config includes the embeddings output layer similar to
    # OlmoEarthEmbeddings.md documentation.
    dataset_config = {
        "layers": {
            "image": {
                "type": "raster",
                "band_sets": [
                    {
                        "dtype": "uint8",
                        "bands": ["band"],
                        "format": {
                            "class_path": "rslearn.utils.raster_format.SingleImageRasterFormat",
                            "init_args": {"format": "png"},
                        },
                    }
                ],
            },
            "embeddings": {
                "type": "raster",
                "band_sets": [
                    {
                        "dtype": "float32",
                        # The Swin-Tiny model with output_layers=[7] produces 768 channels.
                        "num_bands": 768,
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

    # Add a simple image.
    image = np.random.randint(0, 255, size=(1, 32, 32), dtype=np.uint8)
    layer_name = "image"
    layer_dir = window.get_layer_dir(layer_name)
    SingleImageRasterFormat().encode_raster(
        layer_dir / "band",
        window.projection,
        window.bounds,
        RasterArray(chw_array=image),
    )
    window.mark_layer_completed(layer_name)

    return dataset


@pytest.fixture
def embedding_model_config() -> dict:
    """Model config for embedding prediction, similar to OlmoEarthEmbeddings.md.

    Uses Swin-Tiny encoder so the test runs faster.
    """
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
                                "class_path": "rslearn.train.tasks.embedding.EmbeddingHead"
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
                },
                "task": {
                    "class_path": "rslearn.train.tasks.embedding.EmbeddingTask",
                },
                "batch_size": 1,
            },
        },
        "trainer": {
            "callbacks": [
                {
                    "class_path": "rslearn.train.prediction_writer.RslearnWriter",
                    "init_args": {
                        "path": "${DATASET_PATH}",
                        "output_layer": "embeddings",
                    },
                }
            ],
        },
    }


def test_embedding_prediction_writes_to_dataset(
    embedding_dataset: Dataset,
    embedding_model_config: dict,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that embedding model config can be parsed and embeddings are written.

    This test:
    1. Parses a model config via RslearnLightningCLI (similar to OlmoEarthEmbeddings.md, but with Swin-Tiny)
    2. Runs prediction with predict subcommand.
    3. Verifies embeddings are written to the dataset
    """
    init_jsonargparse()

    # Write config to file.
    tmp_fname = tmp_path / "embedding_model.yaml"
    with tmp_fname.open("w") as f:
        json.dump(embedding_model_config, f)

    monkeypatch.setenv("DATASET_PATH", str(embedding_dataset.path))
    RslearnLightningCLI(
        model_class=RslearnLightningModule,
        datamodule_class=RslearnDataModule,
        args=["predict", "--config", str(tmp_fname)],
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        parser_class=RslearnArgumentParser,
    )

    # Verify that embeddings were written to the dataset.
    window = embedding_dataset.load_windows()[0]
    assert window.is_layer_completed("embeddings")
