import json
import pathlib

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
def image_to_class_dataset(tmp_path: pathlib.Path) -> Dataset:
    """Create sample dataset with a raster input and target class.

    It consists of one window with one single-band image and a GeoJSON data with class
    ID property. The property could be used for regression too.
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
    layer_name = "image"
    layer_dir = window.get_layer_dir(layer_name)
    SingleImageRasterFormat().encode_raster(
        layer_dir / "band",
        window.projection,
        window.bounds,
        image,
    )
    window.mark_layer_completed(layer_name)

    # Add label.
    feature = Feature(
        STGeometry(window.projection, shapely.Point(1, 1), None),
        {
            "label": 1,
        },
    )
    layer_name = "label"
    layer_dir = window.get_layer_dir(layer_name)
    GeojsonVectorFormat().encode_vector(
        layer_dir,
        [feature],
    )
    window.mark_layer_completed(layer_name)

    return Dataset(ds_path)
