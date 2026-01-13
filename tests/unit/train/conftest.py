"""Shared stuff."""

import json
import pathlib
from collections.abc import Callable

import numpy.typing as npt
import pytest
import shapely
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import PixelBounds, STGeometry
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat


@pytest.fixture
def basic_classification_dataset(tmp_path: pathlib.Path) -> Dataset:
    """Create an empty dataset setup for image classification.

    This is used in test_dataset.py and test_all_patches_dataset.py.
    """
    ds_path = UPath(tmp_path)
    dataset_config = {
        "layers": {
            "image_layer1": {
                "type": "raster",
                "band_sets": [
                    {
                        "dtype": "uint8",
                        "bands": ["band"],
                    }
                ],
            },
            "image_layer2": {
                "type": "raster",
                "band_sets": [
                    {
                        "dtype": "uint8",
                        "bands": ["band"],
                    }
                ],
            },
            "vector_layer": {"type": "vector"},
        },
    }
    ds_path.mkdir(parents=True, exist_ok=True)
    with (ds_path / "config.json").open("w") as f:
        json.dump(dataset_config, f)
    return Dataset(ds_path)


def add_window(
    dataset: Dataset,
    name: str = "default",
    group: str = "default",
    images: dict[tuple[str, int], npt.NDArray] = {},
    bounds: PixelBounds = (0, 0, 4, 4),
) -> Window:
    """Add a window to the basic classification dataset.

    This is a helper function that goes with basic_classification_dataset to add
    windows to that dataset.

    Args:
        dataset: the dataset to add to.
        name: the name of the window.
        group: the group of the window.
        images: map from (layer_name, group_idx) to the image content, which should be
            1x4x4 since that is the window size.
    """
    window = Window(
        storage=dataset.storage,
        name=name,
        group=group,
        projection=WGS84_PROJECTION,
        bounds=bounds,
        time_range=None,
    )
    window.save()

    for (layer_name, group_idx), image in images.items():
        raster_dir = window.get_raster_dir(layer_name, ["band"], group_idx=group_idx)
        GeotiffRasterFormat().encode_raster(
            raster_dir, window.projection, window.bounds, image
        )
        window.mark_layer_completed(layer_name, group_idx=group_idx)

    # Add label.
    feature = Feature(
        STGeometry(window.projection, shapely.box(*bounds), None),
        {
            "label": 1,
        },
    )
    layer_dir = window.get_layer_dir("vector_layer")
    GeojsonVectorFormat().encode_vector(
        layer_dir,
        [feature],
    )
    window.mark_layer_completed("vector_layer")

    return window


@pytest.fixture
def add_window_to_basic_classification_dataset() -> Callable:
    return add_window
