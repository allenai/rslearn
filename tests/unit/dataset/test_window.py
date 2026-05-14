import pathlib

import pytest
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.dataset.storage.file import FileWindowStorage
from rslearn.dataset.window_data_storage.per_item_group import PerItemGroupStorage


@pytest.fixture
def empty_window(tmp_path: pathlib.Path) -> Window:
    window = Window(
        storage=FileWindowStorage(UPath(tmp_path)),
        group="default",
        name="default",
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 1, 1),
        time_range=None,
        data_storage=PerItemGroupStorage(),
    )
    window.save()
    return window


def test_completed_layer(empty_window: Window) -> None:
    """Mark a layer completed and verify is_layer_completed flips."""
    layer_name = "layer"
    assert not empty_window.is_layer_completed(layer_name)
    empty_window.mark_layer_completed(layer_name)
    assert empty_window.is_layer_completed(layer_name)


def test_window_location(tmp_path: pathlib.Path) -> None:
    # Make sure window directory is in the expected location.
    # This ensures compatibility with existing datasets.
    ds_path = UPath(tmp_path)
    group_name = "group"
    window_name = "window"
    window_dir = Window.get_window_root(ds_path, group_name, window_name)
    assert window_dir == ds_path / "windows" / group_name / window_name
