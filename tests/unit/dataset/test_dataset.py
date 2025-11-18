"""Unit tests for Dataset class."""

import json
import tempfile
from pathlib import Path

import pytest
from upath import UPath

from rslearn.dataset import Dataset


class TestDataset:
    """Test suite for Dataset class."""

    def test_template_substitution_in_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that environment variables are substituted in dataset config.json."""
        # Set up environment variables
        monkeypatch.setenv("LABEL_LAYER", "labels")
        monkeypatch.setenv("PREDICTION_OUTPUT_LAYER", "output")
        monkeypatch.setenv("TILE_STORE_ROOT", "/path/to/tiles")

        # Create a temporary dataset directory with config.json containing template variables
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir)

            # Create config.json with template variables - use a simple config that works
            config_content = {
                "layers": {
                    "test_layer": {"layer_type": "vector"},
                    "${LABEL_LAYER}": {"layer_type": "vector"},
                    "${PREDICTION_OUTPUT_LAYER}": {"layer_type": "vector"},
                },
                "tile_store": {
                    "class_path": "rslearn.tile_stores.default.DefaultTileStore",
                    "init_args": {"path_suffix": "${TILE_STORE_ROOT}"},
                },
            }

            with (dataset_path / "config.json").open("w") as f:
                json.dump(config_content, f)

            # Load the dataset - this should trigger template substitution
            dataset = Dataset(UPath(dataset_path))

            # Verify that environment variables were substituted in tile_store config
            assert dataset.tile_store_config is not None
            assert dataset.layers.keys() == {"test_layer", "labels", "output"}
            assert (
                dataset.tile_store_config["init_args"]["path_suffix"]
                == "/path/to/tiles"
            )

    def test_template_substitution_missing_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that missing environment variables are replaced with empty strings."""
        # Don't set MISSING_VAR environment variable

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir)

            # Create config.json with a missing environment variable
            config_content = {
                "layers": {"test_layer": {"layer_type": "vector"}},
                "tile_store": {
                    "class_path": "rslearn.tile_stores.default.DefaultTileStore",
                    "init_args": {"path_suffix": "/base/path/${MISSING_VAR}/tiles"},
                },
            }

            with (dataset_path / "config.json").open("w") as f:
                json.dump(config_content, f)

            # Load the dataset
            dataset = Dataset(UPath(dataset_path))

            # Verify that missing variable was replaced with empty string
            assert dataset.tile_store_config is not None
            assert (
                dataset.tile_store_config["init_args"]["path_suffix"]
                == "/base/path//tiles"
            )  # Empty string substitution

    def test_no_template_variables(self) -> None:
        """Test that configs without template variables work normally."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir)

            # Create config.json without any template variables
            config_content = {
                "layers": {"test_layer": {"layer_type": "vector"}},
                "tile_store": {
                    "class_path": "rslearn.tile_stores.default.DefaultTileStore",
                    "init_args": {"path_suffix": "/static/path/to/tiles"},
                },
            }

            with (dataset_path / "config.json").open("w") as f:
                json.dump(config_content, f)

            # Load the dataset
            dataset = Dataset(UPath(dataset_path))

            # Verify that static values remain unchanged
            assert dataset.tile_store_config is not None
            assert (
                dataset.tile_store_config["init_args"]["path_suffix"]
                == "/static/path/to/tiles"
            )
