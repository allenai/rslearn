import json
import os
import pathlib

import shapely

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.dataset.manage import (
    ingest_dataset_windows,
    materialize_dataset_windows,
    prepare_dataset_windows,
)
from rslearn.utils import Feature, LocalFileAPI, STGeometry
from rslearn.utils.vector_format import load_vector_format


class TestLocalFiles:
    """Tests the LocalFiles data source.

    1. Create GeoJSON as a local file to extract data from.
    2. Create a corresponding dataset config file.
    3. Create a window intersecting the features.
    3. Run prepare, ingest, materialize, and make sure it gets the features.
    """

    def test_sample_dataset(self, tmp_path: pathlib.Path):
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

        ds_file_api = LocalFileAPI(str(tmp_path))
        Window(
            file_api=Window.get_window_root(ds_file_api, "default", "default"),
            group="default",
            name="default",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 10, 10),
            time_range=None,
        ).save()

        dataset = Dataset(file_api=ds_file_api)
        windows = dataset.load_windows()
        prepare_dataset_windows(dataset, windows)
        ingest_dataset_windows(dataset, windows)
        materialize_dataset_windows(dataset, windows)

        assert len(windows) == 1

        window = windows[0]
        layer_config = dataset.layers["local_file"]
        vector_format = load_vector_format(layer_config.format)
        features = vector_format.decode_vector(
            window.file_api.get_folder("layers", "local_file"), window.bounds
        )

        assert len(features) == 2
