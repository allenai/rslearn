import json
import os
import random

import numpy as np
import pytest
import shapely

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils import Feature, S3FileAPI, STGeometry
from rslearn.utils.raster_format import SingleImageRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat


@pytest.fixture
def image_to_class_dataset() -> Dataset:
    """Create sample dataset with a raster input and target class.

    It consists of one window with one single-band image and a GeoJSON data with class
    ID property. The property could be used for regression too.
    """
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
            "image": {
                "type": "raster",
                "band_sets": [
                    {
                        "dtype": "uint8",
                        "bands": ["band"],
                        "format": {"name": "single_image", "format": "png"},
                    }
                ],
            },
            "label": {"type": "vector", "format": {"name": "geojson"}},
        },
        "tile_store": {
            "name": "file",
            "root_dir": "tiles",
        },
    }
    with ds_file_api.open("config.json", "w") as f:
        json.dump(dataset_config, f)

    window_file_api = Window.get_window_root(ds_file_api, "default", "default")
    window = Window(
        file_api=window_file_api,
        group="default",
        name="default",
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 4, 4),
        time_range=None,
    )
    window.save()

    # Add image where pixel value is 4*col+row.
    image = np.arange(0, 4 * 4, dtype=np.uint8)
    image = image.reshape(1, 4, 4)
    layer_file_api = window_file_api.get_folder("layers", "image")
    SingleImageRasterFormat().encode_raster(
        layer_file_api.get_folder("band"),
        window.projection,
        window.bounds,
        image,
    )
    with layer_file_api.open("completed", "w") as f:
        pass

    # Add label.
    feature = Feature(
        STGeometry(WGS84_PROJECTION, shapely.Point(1, 1), None),
        {
            "label": 1,
        },
    )
    layer_file_api = window_file_api.get_folder("layers", "label")
    GeojsonVectorFormat().encode_vector(
        layer_file_api,
        window.projection,
        [feature],
    )
    with layer_file_api.open("completed", "w") as f:
        pass

    dataset = Dataset(file_api=ds_file_api)
    yield dataset

    for obj in ds_file_api.bucket.objects.filter(Prefix=ds_file_api.prefix):
        obj.delete()
