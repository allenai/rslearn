import json
import os
import random
from pathlib import Path

import shapely
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.dataset.manage import (
    ingest_dataset_windows,
    materialize_dataset_windows,
    prepare_dataset_windows,
)
from rslearn.utils import Feature, STGeometry


class TestLocalFiles:
    """Tests that dataset works with S3 using LocalFiles data source."""

    def cleanup(self, ds_path: UPath) -> None:
        """Delete everything in the specified path."""
        for fname in ds_path.fs.find(ds_path.path):
            ds_path.fs.delete(fname)

    def test_dataset(self, tmp_path: Path) -> None:
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

        test_id = random.randint(10000, 99999)
        bucket_name = os.environ["TEST_BUCKET"]
        prefix = os.environ["TEST_PREFIX"] + f"test_{test_id}/"
        ds_path = UPath(f"gcs://{bucket_name}/{prefix}")

        dataset_config = {
            "layers": {
                "local_file": {
                    "type": "vector",
                    "data_source": {
                        "class_path": "rslearn.data_sources.local_files.LocalFiles",
                        "init_args": {
                            "src_dir": "file://" + src_data_dir,
                        },
                    },
                },
            },
        }
        ds_path.mkdir(parents=True, exist_ok=True)
        with (ds_path / "config.json").open("w") as f:
            json.dump(dataset_config, f)

        Window(
            path=Window.get_window_root(ds_path, "default", "default"),
            group="default",
            name="default",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 10, 10),
            time_range=None,
        ).save()

        dataset = Dataset(ds_path)

        windows = dataset.load_windows()
        prepare_dataset_windows(dataset, windows)
        ingest_dataset_windows(dataset, windows)
        materialize_dataset_windows(dataset, windows)

        assert len(windows) == 1

        window = windows[0]
        layer_config = dataset.layers["local_file"]
        vector_format = layer_config.instantiate_vector_format()
        features = vector_format.decode_vector(
            window.get_layer_dir("local_file"), window.projection, window.bounds
        )

        assert len(features) == 2

        self.cleanup(ds_path)
