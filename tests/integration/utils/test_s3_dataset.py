import json
import os
import random

import shapely

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.dataset.manage import (
    ingest_dataset_windows,
    materialize_dataset_windows,
    prepare_dataset_windows,
)
from rslearn.utils import Feature, S3FileAPI, STGeometry
from rslearn.utils.vector_format import load_vector_format


class TestLocalFiles:
    """Tests that dataset works with S3FileAPI using LocalFiles data source."""

    def cleanup(self, file_api: S3FileAPI):
        """Delete everything in the specified S3FileAPI."""
        for obj in file_api.bucket.objects.filter(Prefix=file_api.prefix):
            obj.delete()

    def test_dataset(self, tmp_path):
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
        ds_file_api = S3FileAPI(
            endpoint_url=os.environ["TEST_S3_ENDPOINT_URL"],
            access_key_id=os.environ["TEST_S3_ACCESS_KEY_ID"],
            secret_access_key=os.environ["TEST_S3_SECRET_ACCESS_KEY"],
            bucket_name=os.environ["TEST_S3_BUCKET_NAME"],
            prefix=os.environ["TEST_S3_PREFIX"] + f"test_{test_id}/",
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
        with ds_file_api.open("config.json", "w") as f:
            json.dump(dataset_config, f)

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

        self.cleanup(ds_file_api)

    def test_listdir(self):
        """Make sure that listdir in S3FileAPI works properly.

        It should return all the prefixes in the specified folder.
        """
        test_id = random.randint(10000, 99999)
        file_api = S3FileAPI(
            endpoint_url=os.environ["TEST_S3_ENDPOINT_URL"],
            access_key_id=os.environ["TEST_S3_ACCESS_KEY_ID"],
            secret_access_key=os.environ["TEST_S3_SECRET_ACCESS_KEY"],
            bucket_name=os.environ["TEST_S3_BUCKET_NAME"],
            prefix=os.environ["TEST_S3_PREFIX"] + f"test_{test_id}/",
        )
        with file_api.open("x/prefix1", "w"):
            pass
        with file_api.open("x/prefix2/suffix", "w"):
            pass
        with file_api.open("x/prefix1/suffix", "w"):
            pass
        with file_api.open("x/prefix3/suffix/suffix", "w"):
            pass
        prefixes = file_api.listdir("x")
        prefixes.sort()
        assert prefixes == ["prefix1", "prefix2", "prefix3"]

        self.cleanup(file_api)
