import json
import os
import random
from typing import Any, Union

import numpy as np
import numpy.typing as npt
import pytest

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.train.dataset import DataInput, ModelDataset, SplitConfig
from rslearn.train.tasks.task import Task
from rslearn.train.transforms.flip import Flip
from rslearn.utils import Feature, S3FileAPI
from rslearn.utils.raster_format import SingleImageRasterFormat


class FakeTask(Task):
    """A placeholder task."""

    def __init__(self):
        """Initialize a new FakeTask."""
        super().__init__()

    def process_inputs(
        self,
        raw_inputs: dict[str, Union[npt.NDArray[Any], list[Feature]]],
        load_targets: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return {}, {}


class TestTransforms:
    """Test transforms working with ModelDataset."""

    @pytest.fixture(scope="class")
    def dataset(self):
        """Create sample dataset with one window with one single-band image."""
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
        image = np.array(
            [
                [0, 4, 8, 12],
                [1, 5, 9, 13],
                [2, 6, 10, 14],
                [3, 7, 11, 15],
            ],
            dtype=np.uint8,
        )
        layer_file_api = window_file_api.get_folder("layers", "image")
        SingleImageRasterFormat().encode_raster(
            layer_file_api.get_folder("band"),
            window.projection,
            window.bounds,
            image[None, :, :],
        )
        with layer_file_api.open("completed", "w") as f:
            pass

        dataset = Dataset(file_api=ds_file_api)
        yield dataset

        for obj in ds_file_api.bucket.objects.filter(Prefix=ds_file_api.prefix):
            obj.delete()

    def test_flip(self, dataset: Dataset):
        split_config = SplitConfig(transforms=[Flip()])
        image_data_input = DataInput(
            "raster", ["image"], bands=["band"], passthrough=True
        )
        model_dataset = ModelDataset(
            dataset,
            split_config,
            {
                "image": image_data_input,
            },
            workers=1,
            task=FakeTask(),
        )
        input_dict, _, _ = model_dataset[0]
        assert input_dict["image"].shape == (1, 4, 4)
