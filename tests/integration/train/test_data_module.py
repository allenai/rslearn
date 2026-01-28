"""Tests for rslearn.train.data_module."""

import json
from pathlib import Path

import numpy as np
import pytest
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.dataset import DataInput, SplitConfig
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.utils.raster_format import GeotiffRasterFormat


class TestPredictLoader:
    """Tests for RslearnDataModule relating to the predict dataloader."""

    LAYER_NAME = "image"
    BANDS = ["band"]
    INPUTS = {
        "image": DataInput(
            data_type="raster",
            layers=[LAYER_NAME],
            bands=BANDS,
            passthrough=True,
        ),
    }
    TASK = ClassificationTask(
        property_name="prop_name",
        classes=["negative", "positive"],
    )
    SPLIT_CONFIG = SplitConfig(
        load_all_crops=True,
        patch_size=2,
        skip_targets=True,
    )

    @pytest.fixture
    def empty_image_dataset(self, tmp_path: Path) -> Dataset:
        """Create an empty dataset for image data."""
        dataset_config = {
            "layers": {
                self.LAYER_NAME: {
                    "type": "raster",
                    "band_sets": [
                        {
                            "dtype": "uint8",
                            "bands": self.BANDS,
                        }
                    ],
                },
            },
        }
        with (tmp_path / "config.json").open("w") as f:
            json.dump(dataset_config, f)

        return Dataset(UPath(tmp_path))

    def test_predict_dataloader_no_windows(self, empty_image_dataset: Dataset) -> None:
        """Verify that the dataloader works with no windows.

        We use load_all_crops for prediction. Previously there were some bugs with
        this use case.
        """
        data_module = RslearnDataModule(
            inputs=self.INPUTS,
            task=self.TASK,
            path=empty_image_dataset.path,
            batch_size=4,
            num_workers=4,
            predict_config=self.SPLIT_CONFIG,
        )
        data_module.setup("predict")
        inputs = list(data_module.predict_dataloader())
        assert len(inputs) == 0

    def test_predict_dataloader_one_window(self, empty_image_dataset: Dataset) -> None:
        """Verify that it works with one window."""
        # Make the window 4x4 so there should be 4 patches.
        # We use batch size 2 so there should be 2 batches.
        window = Window(
            storage=empty_image_dataset.storage,
            group="group",
            name="window",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 4, 4),
            time_range=None,
        )
        window.save()
        image = np.zeros((1, 4, 4), dtype=np.uint8)
        image[0, 0, 0] = 1
        GeotiffRasterFormat().encode_raster(
            window.get_raster_dir(self.LAYER_NAME, self.BANDS),
            window.projection,
            window.bounds,
            image,
        )
        window.mark_layer_completed(self.LAYER_NAME)

        data_module = RslearnDataModule(
            inputs=self.INPUTS,
            task=self.TASK,
            path=empty_image_dataset.path,
            batch_size=2,
            num_workers=4,
            predict_config=self.SPLIT_CONFIG,
        )
        data_module.setup("predict")
        inputs = list(data_module.predict_dataloader())
        print(inputs)
        assert len(inputs) == 2
        # Each batch is a (input_dicts, target_dicts, metadata_dicts) tuple.
        # We only care about the first tuple here.
        input_dicts1 = inputs[0][0]
        input_dicts2 = inputs[1][0]
        assert len(input_dicts1) == 2
        assert len(input_dicts2) == 2
        # All patches should be zero except the first one which should correspond to
        # the topleft of the image where we have set one pixel to 1.
        assert input_dicts1[0]["image"].image.max() == 1
        assert input_dicts1[1]["image"].image.max() == 0
        assert input_dicts2[0]["image"].image.max() == 0
        assert input_dicts2[1]["image"].image.max() == 0
