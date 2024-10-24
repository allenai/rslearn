import json
import os

import fiona.errors
import pytest

from rslearn.dataset import Dataset
from rslearn.dataset.manage import (
    ingest_dataset_windows,
    materialize_dataset_windows,
    prepare_dataset_windows,
)


def test_layer_alias(local_files_dataset: Dataset) -> None:
    """Set layers with alias and make sure they share the place in tile store."""
    # Src dir is set in fixture.
    src_dir = local_files_dataset.src_dir  # type: ignore
    windows = local_files_dataset.load_windows()
    window_path = windows[0].path

    # Update dataset config to use a layer with alias, and reload.
    dataset_config = {
        "layers": {
            "layer1": {
                "type": "vector",
                "alias": "common",
                "data_source": {
                    "name": "rslearn.data_sources.local_files.LocalFiles",
                    "src_dir": src_dir,
                },
            },
        },
        "tile_store": {
            "name": "file",
            "root_dir": "tiles",
        },
    }
    with (local_files_dataset.path / "config.json").open("w") as f:
        json.dump(dataset_config, f)
    local_files_dataset = Dataset(local_files_dataset.path)
    print("alias", local_files_dataset.layers["layer1"].alias)

    # Materialize and make sure it shows up in window.
    prepare_dataset_windows(local_files_dataset, windows)
    ingest_dataset_windows(local_files_dataset, windows)
    materialize_dataset_windows(local_files_dataset, windows)
    assert (window_path / "layers" / "layer1" / "data.geojson").exists()

    # Now make a different layer with same alias.
    # Then rename src_dir after preparing.
    # It should be able to materialize.
    dataset_config = {
        "layers": {
            "layer2": {
                "type": "vector",
                "alias": "common",
                "data_source": {
                    "name": "rslearn.data_sources.local_files.LocalFiles",
                    "src_dir": src_dir,
                },
            },
        },
        "tile_store": {
            "name": "file",
            "root_dir": "tiles",
        },
    }
    with (local_files_dataset.path / "config.json").open("w") as f:
        json.dump(dataset_config, f)
    local_files_dataset = Dataset(local_files_dataset.path)
    prepare_dataset_windows(local_files_dataset, windows)
    os.rename(src_dir, src_dir + ".moved")
    ingest_dataset_windows(local_files_dataset, windows)
    materialize_dataset_windows(local_files_dataset, windows)
    os.rename(src_dir + ".moved", src_dir)
    assert (window_path / "layers" / "layer2" / "data.geojson").exists()

    # Now third layer with no alias should fail.
    dataset_config = {
        "layers": {
            "layer3": {
                "type": "vector",
                "data_source": {
                    "name": "rslearn.data_sources.local_files.LocalFiles",
                    "src_dir": src_dir,
                },
            },
        },
        "tile_store": {
            "name": "file",
            "root_dir": "tiles",
        },
    }
    with (local_files_dataset.path / "config.json").open("w") as f:
        json.dump(dataset_config, f)
    local_files_dataset = Dataset(local_files_dataset.path)
    prepare_dataset_windows(local_files_dataset, windows)
    os.rename(src_dir, src_dir + ".moved")
    with pytest.raises(fiona.errors.DriverError):
        ingest_dataset_windows(local_files_dataset, windows)
