import json
import os
import random
from collections.abc import Generator

import numpy as np
import pytest
import shapely
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils import Feature, STGeometry
from rslearn.utils.raster_format import SingleImageRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat


@pytest.fixture
def image_to_class_dataset() -> Generator[Dataset, None, None]:
    """Create sample dataset with a raster input and target class.

    It consists of one window with one single-band image and a GeoJSON data with class
    ID property. The property could be used for regression too.
    """
    test_id = random.randint(10000, 99999)
    bucket_name = os.environ["TEST_BUCKET"]
    prefix = os.environ["TEST_PREFIX"] + f"test_{test_id}/"
    ds_path = UPath(f"gcs://{bucket_name}/{prefix}")

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
    ds_path.mkdir(parents=True, exist_ok=True)
    with (ds_path / "config.json").open("w") as f:
        json.dump(dataset_config, f)

    window_path = Window.get_window_root(ds_path, "default", "default")
    window = Window(
        path=window_path,
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
    layer_dir = window_path / "layers" / "image"
    SingleImageRasterFormat().encode_raster(
        layer_dir / "band",
        window.projection,
        window.bounds,
        image,
    )
    (layer_dir / "completed").touch()

    # Add label.
    feature = Feature(
        STGeometry(WGS84_PROJECTION, shapely.Point(1, 1), None),
        {
            "label": 1,
        },
    )
    layer_dir = window_path / "layers" / "label"
    GeojsonVectorFormat().encode_vector(
        layer_dir,
        window.projection,
        [feature],
    )
    (layer_dir / "completed").touch()

    dataset = Dataset(ds_path)
    yield dataset

    for fname in ds_path.fs.find(ds_path.path):
        ds_path.fs.delete(fname)
