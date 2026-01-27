"""Integration tests for single_file_materialization with training pipeline."""

import json
import pathlib
from typing import Any

import numpy as np
import pytest
from shapely.geometry import Polygon
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import Item
from rslearn.dataset import Dataset, Window
from rslearn.dataset.materialize import RasterMaterializer
from rslearn.tile_stores.default import DefaultTileStore
from rslearn.tile_stores.tile_store import TileStoreWithLayer
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.dataset import DataInput
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.utils.geometry import STGeometry


def make_item(name: str, projection: Any, bounds: tuple) -> Item:
    """Create a simple item with the given name."""
    return Item(
        name=name,
        geometry=STGeometry(
            projection=projection,
            shp=Polygon(
                [
                    (bounds[0], bounds[1]),
                    (bounds[2], bounds[1]),
                    (bounds[2], bounds[3]),
                    (bounds[0], bounds[3]),
                ]
            ),
            time_range=None,
        ),
    )


@pytest.fixture
def single_file_materialization_dataset(tmp_path: pathlib.Path) -> Dataset:
    """Create a dataset with single_file_materialization enabled.

    The dataset has:
    - An "image" layer with single_file_materialization enabled, containing multiple
      item groups that will be stacked into a single file

    This fixture also materializes the data from a tile store.
    """
    ds_path = UPath(tmp_path) / "dataset"
    tile_path = UPath(tmp_path) / "tiles"

    # Dataset config with single_file_materialization enabled for the image layer
    dataset_config = {
        "layers": {
            "image": {
                "type": "raster",
                "band_sets": [
                    {
                        "dtype": "uint8",
                        "bands": ["band"],
                    }
                ],
                "single_file_materialization": True,
            },
        },
    }

    ds_path.mkdir(parents=True, exist_ok=True)
    with (ds_path / "config.json").open("w") as f:
        json.dump(dataset_config, f)
    dataset = Dataset(ds_path)

    # Create a tile store and write data for multiple item groups
    tile_store = DefaultTileStore()
    tile_store.set_dataset_path(tile_path)

    bounds = (0, 0, 32, 32)
    projection = WGS84_PROJECTION

    # Create window
    window = Window(
        storage=dataset.storage,
        group="default",
        name="default",
        projection=projection,
        bounds=bounds,
        time_range=None,
    )
    window.save()

    # Create 3 item groups with different data for the "image" layer
    # Each group has a distinct pattern that we can verify during data loading
    item_groups = []
    for group_idx in range(3):
        item = make_item(f"item_{group_idx}", projection, bounds)
        # Create image with unique values per group: 50, 100, 150
        image_data = np.full((1, 32, 32), (group_idx + 1) * 50, dtype=np.uint8)
        tile_store.write_raster(
            "image",
            item.name,
            ["band"],
            projection,
            bounds,
            image_data,
        )
        item_groups.append([item])

    # Materialize the image layer using RasterMaterializer
    image_layer_cfg = dataset.layers["image"]
    materializer = RasterMaterializer()
    materializer.materialize(
        TileStoreWithLayer(tile_store, "image"),
        window,
        "image",
        image_layer_cfg,
        item_groups,
    )

    return dataset


def test_single_file_materialization_data_module(
    single_file_materialization_dataset: Dataset,
) -> None:
    """Test that RslearnDataModule can load data with single_file_materialization.

    This test creates an RslearnDataModule and verifies that the train dataloader
    correctly loads data from a single-file materialized layer.
    """
    from rslearn.train.dataset import SplitConfig

    dataset = single_file_materialization_dataset

    # Create DataInput with required settings for single_file_materialization
    image_data_input = DataInput(
        data_type="raster",
        layers=["image"],
        bands=["band"],
        passthrough=True,
        load_all_item_groups=True,
        load_all_layers=True,
    )

    # Create a simple task (we won't actually train, just verify data loading)
    task = ClassificationTask(
        property_name="dummy",
        classes=["a", "b"],
    )

    # Use skip_targets=True since we don't have targets in the dataset
    train_config = SplitConfig(skip_targets=True)

    # Create data module
    data_module = RslearnDataModule(
        path=dataset.path,
        inputs={"image": image_data_input},
        task=task,
        batch_size=1,
        num_workers=0,  # Use 0 workers for simpler debugging
        train_config=train_config,
        val_config=train_config,
    )

    # Setup the data module for training
    data_module.setup("fit")

    # Get one batch from the train dataloader
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    # batch is (input_dicts, target_dicts, metadata)
    input_dicts, _, _ = batch

    # Verify the image was loaded correctly
    # input_dicts is a list of dicts, one per batch item
    assert len(input_dicts) == 1
    assert "image" in input_dicts[0]

    raster_image = input_dicts[0]["image"]
    # Shape should be (C, T, H, W) = (1, 3, 32, 32)
    assert raster_image.image.shape == (1, 3, 32, 32), (
        f"Expected shape (1, 3, 32, 32), got {raster_image.image.shape}"
    )
