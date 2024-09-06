"""Dataset and source files for ingesting from local files."""

import json
import os
import pathlib

import pytest
import shapely
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils import Feature, STGeometry


@pytest.fixture
def local_files_dataset(tmp_path: pathlib.Path) -> Dataset:
    """Creates a dataset for ingesting local files.

    This fixture is Dataset object. It has additional src_dir specifying the directory
    of the source files, which is always a subfolder named "src_data" of the dataset
    root.

    Args:
        tmp_path: temporary path to use for dataset root.

    Returns:
        the dataset.
    """
    features = [
        Feature(
            geometry=STGeometry(WGS84_PROJECTION, shapely.Point(5, 5), None),
        ),
        Feature(
            geometry=STGeometry(WGS84_PROJECTION, shapely.Point(6, 6), None),
        ),
    ]
    src_data_dir = os.path.join(tmp_path, "src_data")
    os.makedirs(src_data_dir)
    with open(os.path.join(src_data_dir, "data.geojson"), "w") as f:
        json.dump(
            {
                "type": "FeatureCollection",
                "features": [feat.to_geojson() for feat in features],
            },
            f,
        )

    dataset_config = {
        "layers": {
            "local_file": {
                "type": "vector",
                "data_source": {
                    "name": "rslearn.data_sources.local_files.LocalFiles",
                    "src_dir": src_data_dir,
                },
            },
        },
        "tile_store": {
            "name": "file",
            "root_dir": "tiles",
        },
    }
    with open(os.path.join(tmp_path, "config.json"), "w") as f:
        json.dump(dataset_config, f)

    ds_path = UPath(tmp_path)
    Window(
        path=Window.get_window_root(ds_path, "default", "default"),
        group="default",
        name="default",
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 10, 10),
        time_range=None,
    ).save()

    dataset = Dataset(ds_path)
    dataset.src_dir = src_data_dir
    return dataset
