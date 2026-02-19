import json
import pathlib
from typing import Any

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from rasterio.crs import CRS
from upath import UPath

from rslearn.config import (
    BandSetConfig,
    DatasetConfig,
    DType,
    LayerConfig,
    LayerType,
    StorageConfig,
)
from rslearn.dataset import Dataset, Window
from rslearn.models.conv import Conv
from rslearn.models.module_wrapper import EncoderModuleWrapper
from rslearn.models.singletask import SingleTaskModel
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.dataset import DataInput, ModelDataset, SplitConfig
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.train.model_context import ModelContext, RasterImage
from rslearn.train.optimizer import AdamW
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.train.tasks.per_pixel_regression import (
    PerPixelRegressionHead,
    PerPixelRegressionTask,
)
from rslearn.utils.geometry import Projection, ResolutionFactor
from rslearn.utils.raster_format import GeotiffRasterFormat


class TestDataset:
    """Test ModelDataset."""

    def test_multiple_tags(self, tmp_path: pathlib.Path) -> None:
        """Ensure that ModelDataset filters correctly when multile tags are configured.

        Multiple tags should be treated as conjunction (logical and) so only windows
        matching all of the tags should be accepted.
        """
        ds_path = UPath(tmp_path)
        with (ds_path / "config.json").open("w") as f:
            json.dump({"layers": {}}, f)
        dataset = Dataset(ds_path)

        group = "default"
        projection = Projection(CRS.from_epsg(32614), 10, -10)
        bounds = (500000, 500000, 500040, 500040)

        def add_window(name: str, options: dict[str, Any]) -> None:
            Window(
                storage=dataset.storage,
                group=group,
                name=name,
                projection=projection,
                bounds=bounds,
                time_range=None,
                options=options,
            ).save()

        # (1) Window with first tag only (skip).
        add_window("window1", {"tag1": "yes"})
        # (2) Window with second tag only (skip).
        add_window("window2", {"tag2": "yes"})
        # (3) Window with both tags (match).
        add_window("window3", {"tag1": "yes", "tag2": "yes"})
        # (4) Window with both tags plus one more (match).
        add_window("window4", {"tag1": "yes", "tag2": "yes", "tag3": "yes"})

        model_dataset = ModelDataset(
            dataset=dataset,
            split_config=SplitConfig(tags={"tag1": "yes", "tag2": "yes"}),
            inputs={},
            task=ClassificationTask(property_name="prop_name", classes=[]),
            workers=4,
        )
        assert len(model_dataset) == 2
        window_names = {window.name for window in model_dataset.get_dataset_examples()}
        assert window_names == {"window3", "window4"}

    def test_empty_dataset(self, tmp_path: pathlib.Path) -> None:
        """Ensure ModelDataset works with no windows."""
        ds_path = UPath(tmp_path)
        with (ds_path / "config.json").open("w") as f:
            json.dump({"layers": {}}, f)
        dataset = ModelDataset(
            dataset=Dataset(ds_path),
            split_config=SplitConfig(),
            inputs={},
            task=ClassificationTask(property_name="prop_name", classes=[]),
            workers=4,
        )
        assert len(dataset) == 0

    def test_load_two_item_groups_sqlite(self, tmp_path: pathlib.Path) -> None:
        """Test ModelDataset with multiple item groups and with SQLiteWindowStorage.

        Verifies that layers specified as ["raster_layer.0", "raster_layer.1"] are
        correctly resolved via is_data_input_available and read_data_input when the
        storage backend is SQLite (where the layer_name and group_idx are stored
        separately rather than derived from a directory name).
        """
        ds_path = UPath(tmp_path)

        # Write dataset config with SQLiteWindowStorage and one single-band layer.
        cfg = DatasetConfig(
            layers=dict(
                raster_layer=LayerConfig(
                    type=LayerType.RASTER,
                    band_sets=[
                        BandSetConfig(
                            dtype=DType.UINT8,
                            bands=["B1"],
                        )
                    ],
                ),
            ),
            storage=StorageConfig(
                class_path="rslearn.dataset.storage.sqlite.SQLiteWindowStorageFactory",
                init_args={},
            ),
        )
        with (ds_path / "config.json").open("w") as f:
            f.write(cfg.model_dump_json())

        dataset = Dataset(ds_path)

        # Create a window.
        projection = Projection(CRS.from_epsg(3857), 1, -1)
        bounds = (0, 0, 4, 4)
        window = Window(
            storage=dataset.storage,
            group="default",
            name="win0",
            projection=projection,
            bounds=bounds,
            time_range=None,
        )
        window.save()

        # Write raster data for group_idx=0 (value 1) and group_idx=1 (value 2).
        for group_idx, pixel_value in [(0, 1), (1, 2)]:
            raster_dir = window.get_raster_dir(
                "raster_layer", ["B1"], group_idx=group_idx
            )
            GeotiffRasterFormat().encode_raster(
                raster_dir,
                projection,
                bounds,
                np.full((1, 4, 4), pixel_value, dtype=np.uint8),
            )
            window.mark_layer_completed("raster_layer", group_idx=group_idx)

        # Build ModelDataset with both item groups specified explicitly.
        model_dataset = ModelDataset(
            dataset=dataset,
            split_config=SplitConfig(),
            inputs=dict(
                image=DataInput(
                    data_type="raster",
                    layers=["raster_layer.0", "raster_layer.1"],
                    bands=["B1"],
                    dtype=DType.FLOAT32,
                    passthrough=True,
                    load_all_layers=True,
                ),
            ),
            task=PerPixelRegressionTask(),
            workers=0,
        )

        # The window should be found.
        assert len(model_dataset) == 1

        # Load the raw inputs.  With load_all_layers=True the two groups are
        # stacked along dim=1, giving shape (batch=1, time=2, H=4, W=4).
        raw_inputs, _, _ = model_dataset.get_raw_inputs(0)
        image_tensor = raw_inputs["image"].image
        assert image_tensor.shape == (1, 2, 4, 4), (
            f"unexpected shape {image_tensor.shape}"
        )
        # group_idx=0 was filled with 1, group_idx=1 with 2.
        assert image_tensor[0, 0].mean().item() == pytest.approx(1.0)
        assert image_tensor[0, 1].mean().item() == pytest.approx(2.0)


class TestResolutionFactor:
    """Integration test for ModelDataset with DataInputs that have resolution factor.

    We verify we can train a model to input 4x4 but output 2x2 for PerPixelRegressionTask.
    """

    def create_dataset(self, ds_path: UPath) -> Dataset:
        """Write the dataset config and return dataset."""
        cfg = DatasetConfig(
            layers=dict(
                image=LayerConfig(
                    type=LayerType.RASTER,
                    band_sets=[
                        BandSetConfig(
                            dtype=DType.UINT8,
                            bands=["B1"],
                        )
                    ],
                ),
                label=LayerConfig(
                    type=LayerType.RASTER,
                    band_sets=[
                        BandSetConfig(
                            dtype=DType.UINT8,
                            bands=["B1"],
                        )
                    ],
                ),
            )
        )
        with (ds_path / "config.json").open("w") as f:
            f.write(cfg.model_dump_json())
        return Dataset(ds_path)

    def add_window(self, ds_path: UPath, group: str, name: str) -> Window:
        """Add a window with the specified name."""
        dataset = Dataset(ds_path)
        window = Window(
            storage=dataset.storage,
            group=group,
            name=name,
            projection=Projection(CRS.from_epsg(3857), 1, -1),
            bounds=(0, 0, 4, 4),
            time_range=None,
        )
        window.save()

        # Add image layer.
        GeotiffRasterFormat().encode_raster(
            window.get_raster_dir("image", ["B1"]),
            window.projection,
            window.bounds,
            np.ones((1, 4, 4), dtype=np.uint8),
        )
        window.mark_layer_completed("image")

        # Add label layer.
        GeotiffRasterFormat().encode_raster(
            window.get_raster_dir("label", ["B1"]),
            window.projection,
            window.bounds,
            2 * np.ones((1, 4, 4), dtype=np.uint8),
        )
        window.mark_layer_completed("label")

        return window

    def test_per_pixel_regression(self, tmp_path: pathlib.Path) -> None:
        """Run the test with PerPixelRegressionTask."""
        ds_path = UPath(tmp_path)
        self.create_dataset(ds_path)
        for idx in range(16):
            self.add_window(ds_path, "default", f"window{idx}")

        task = PerPixelRegressionTask()

        # Create the data module.
        data_module = RslearnDataModule(
            task=task,
            inputs=dict(
                image=DataInput(
                    data_type="raster",
                    layers=["image"],
                    bands=["B1"],
                    dtype=DType.FLOAT32,
                    passthrough=True,
                ),
                targets=DataInput(
                    data_type="raster",
                    layers=["label"],
                    bands=["B1"],
                    dtype=DType.INT32,
                    is_target=True,
                    # Here we set the resolution factor so the target is 2x2.
                    resolution_factor=ResolutionFactor(denominator=2),
                ),
            ),
            path=str(ds_path),
        )

        # Create the model architecture. It is just two convs, one to downsample and
        # another to make the prediction.
        model = SingleTaskModel(
            encoder=[
                EncoderModuleWrapper(
                    module=Conv(
                        in_channels=1,
                        out_channels=32,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                ),
            ],
            decoder=[
                Conv(
                    in_channels=32,
                    out_channels=1,
                    kernel_size=3,
                    activation=torch.nn.Identity(),
                ),
                PerPixelRegressionHead(),
            ],
        )

        # Perform fit.
        lm = RslearnLightningModule(
            model,
            task=task,
            optimizer=AdamW(lr=0.001),
        )
        trainer = pl.Trainer(max_epochs=10)
        trainer.fit(lm, datamodule=data_module)

        # Make sure model produces the right output.
        model.eval()
        output = (
            model(
                ModelContext(
                    inputs=[
                        {
                            "image": RasterImage(
                                torch.ones((1, 1, 4, 4), dtype=torch.float32)
                            ),
                        }
                    ],
                    metadatas=[],
                )
            )
            .outputs.detach()
            .numpy()
        )
        print(output)
        # Index into BCHW tensor.
        assert output[0, 0, 0] == pytest.approx(2, abs=0.01)
        assert output[0, 0, 1] == pytest.approx(2, abs=0.01)
        assert output[0, 1, 0] == pytest.approx(2, abs=0.01)
        assert output[0, 1, 1] == pytest.approx(2, abs=0.01)
