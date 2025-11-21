import json
import pathlib

import numpy as np
import pytest
import shapely
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.models.pooling_decoder import PoolingDecoder
from rslearn.models.singletask import SingleTaskModel
from rslearn.models.swin import Swin
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.dataset import DataInput
from rslearn.train.tasks.classification import ClassificationHead, ClassificationTask
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
                        "format": {
                            "class_path": "rslearn.utils.raster_format.SingleImageRasterFormat",
                            "init_args": {"format": "png"},
                        },
                    }
                ],
            },
            "label": {"type": "vector"},
            "output": {"type": "vector"},
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
        STGeometry(window.projection, shapely.box(*window.bounds), None),
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


@pytest.fixture
def image_to_class_data_module(image_to_class_dataset: Dataset) -> RslearnDataModule:
    """Create an RslearnDataModule for the image_to_class_dataset."""
    image_data_input = DataInput("raster", ["image"], bands=["band"], passthrough=True)
    target_data_input = DataInput("vector", ["label"])
    task = ClassificationTask("label", ["cls0", "cls1"], read_class_id=True)
    return RslearnDataModule(
        path=image_to_class_dataset.path,
        inputs={
            "image": image_data_input,
            "targets": target_data_input,
        },
        task=task,
    )


@pytest.fixture
def image_to_class_model() -> SingleTaskModel:
    """Create a SingleTaskModel for use with image_to_class_data_module."""
    return SingleTaskModel(
        encoder=[
            Swin(arch="swin_v2_t", input_channels=1, output_layers=[3]),
        ],
        decoder=[
            PoolingDecoder(in_channels=192, out_channels=2),
            ClassificationHead(),
        ],
    )
