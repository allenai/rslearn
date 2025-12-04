"""Tests for all storages."""

import pathlib

import pytest
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import Item
from rslearn.dataset.storage.file import FileDatasetStorageFactory
from rslearn.dataset.storage.storage import DatasetStorageFactory
from rslearn.dataset.window import Window, WindowLayerData

STORAGE_FACTORIES: list[DatasetStorageFactory] = [
    FileDatasetStorageFactory(),
]


@pytest.mark.parametrize("storage_factory", STORAGE_FACTORIES)
def test_empty_dataset(
    storage_factory: DatasetStorageFactory, tmp_path: pathlib.Path
) -> None:
    """Make sure there are no windows in a new dataset."""
    storage = storage_factory.get_storage(UPath(tmp_path))
    assert storage.get_windows() == []


@pytest.mark.parametrize("storage_factory", STORAGE_FACTORIES)
def test_create_and_update_window(
    storage_factory: DatasetStorageFactory, tmp_path: pathlib.Path
) -> None:
    """Create one window and update it and make sure everything works."""
    storage = storage_factory.get_storage(UPath(tmp_path))

    # Create a window and make sure it is returned.
    window = Window(
        storage=storage,
        group="group",
        name="name",
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 4, 4),
        time_range=None,
    )
    metadata = window.get_metadata()
    storage.create_or_update_window(window)

    get_result = storage.get_windows()
    assert len(get_result) == 1
    assert get_result[0].get_metadata() == metadata

    # Now update the window.
    window.bounds = (0, 0, 5, 5)
    storage.create_or_update_window(window)

    get_result = storage.get_windows()
    assert len(get_result) == 1
    assert get_result[0].bounds == (0, 0, 5, 5)


@pytest.mark.parametrize("storage_factory", STORAGE_FACTORIES)
def test_mark_one_layer_completed(
    storage_factory: DatasetStorageFactory, tmp_path: pathlib.Path
) -> None:
    """Create two windows and mark one layer completed in one of them."""
    storage = storage_factory.get_storage(UPath(tmp_path))

    group = "group"
    window_names = ["window1", "window2"]
    windows = []
    for window_name in window_names:
        window = Window(
            storage=storage,
            group=group,
            name=window_name,
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 4, 4),
            time_range=None,
        )
        windows.append(window)
        storage.create_or_update_window(window)

    # Make layer directory since it is expected for the data to be materialized there
    # before marking completed.
    windows[0].get_layer_dir("layer_name").mkdir(parents=True, exist_ok=True)

    storage.mark_layer_completed(group, window_names[0], "layer_name")
    assert storage.is_layer_completed(group, window_names[0], "layer_name")
    assert not storage.is_layer_completed(group, window_names[1], "layer_name")


@pytest.mark.parametrize("storage_factory", STORAGE_FACTORIES)
def test_mark_two_item_groups_completed(
    storage_factory: DatasetStorageFactory, tmp_path: pathlib.Path
) -> None:
    """Mark two item groups completed, make sure other item groups are not completed."""
    storage = storage_factory.get_storage(UPath(tmp_path))
    window = Window(
        storage=storage,
        group="group",
        name="name",
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 4, 4),
        time_range=None,
    )
    storage.create_or_update_window(window)

    # Make layer directory since it is expected for the data to be materialized there
    # before marking completed.
    window.get_layer_dir("layer_name", group_idx=0).mkdir(parents=True, exist_ok=True)
    window.get_layer_dir("layer_name", group_idx=1).mkdir(parents=True, exist_ok=True)

    storage.mark_layer_completed("group", "name", "layer_name", group_idx=0)
    storage.mark_layer_completed("group", "name", "layer_name", group_idx=1)
    assert storage.is_layer_completed("group", "name", "layer_name", group_idx=0)
    assert storage.is_layer_completed("group", "name", "layer_name", group_idx=1)
    assert not storage.is_layer_completed("group", "name", "layer_name", group_idx=2)


@pytest.mark.parametrize("storage_factory", STORAGE_FACTORIES)
def test_save_layer_datas(
    storage_factory: DatasetStorageFactory, tmp_path: pathlib.Path
) -> None:
    """Save some layer datas and then load them again."""
    storage = storage_factory.get_storage(UPath(tmp_path))
    window = Window(
        storage=storage,
        group="group",
        name="name",
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 4, 4),
        time_range=None,
    )
    storage.create_or_update_window(window)
    assert storage.get_layer_datas("group", "name") == {}
    item = Item("item", window.get_geometry())
    storage.save_layer_datas(
        "group",
        "name",
        {
            "layer_name": WindowLayerData(
                "layer_name", serialized_item_groups=[[item.serialize()]]
            ),
        },
    )

    layer_datas = storage.get_layer_datas("group", "name")
    assert len(layer_datas) == 1
    assert len(layer_datas["layer_name"].serialized_item_groups) == 1
    deserialized_item = Item.deserialize(
        layer_datas["layer_name"].serialized_item_groups[0][0]
    )
    assert deserialized_item.name == "item"
