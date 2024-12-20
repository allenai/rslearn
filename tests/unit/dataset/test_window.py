import pathlib

import pytest
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window


@pytest.fixture
def empty_window(tmp_path: pathlib.Path) -> Window:
    window = Window(
        path=UPath(tmp_path),
        group="default",
        name="default",
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 1, 1),
        time_range=None,
    )
    window.save()
    return window


def test_completed_layer(empty_window: Window) -> None:
    # Mark a layer completed and make sure is_layer_completed is True.
    # We need to write some files there, mark_layer_completed assumes the directory
    # exists already (otherwise it couldn't possible be completed).
    layer_name = "layer"
    layer_dir = empty_window.get_layer_dir(layer_name)
    layer_dir.mkdir(parents=True)
    (layer_dir / "somefiles").touch()

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


def test_layer_dir_location(empty_window: Window) -> None:
    # Make sure layer directory is in the expected location.
    # This ensures compatibility with existing datasets.
    layer_name = "layer"
    layer_dir = empty_window.get_layer_dir(layer_name)
    assert layer_dir == empty_window.path / "layers" / layer_name
