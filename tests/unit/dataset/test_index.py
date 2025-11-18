import json
import pathlib

import pytest
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset.dataset import Dataset
from rslearn.dataset.index import DatasetIndex
from rslearn.dataset.window import Window, WindowLayerData


@pytest.fixture
def dummy_dataset(tmp_path: pathlib.Path) -> Dataset:
    """An empty dataset with one layer called layer."""
    ds_path = UPath(tmp_path)
    with (ds_path / "config.json").open("w") as f:
        json.dump({"layers": {"layer": {"type": "vector"}}}, f)
    return Dataset(ds_path)


def make_window(dataset: Dataset, name: str) -> Window:
    """Add a window to the dataset with the given name."""
    window = Window(
        path=Window.get_window_root(dataset.path, "group", name),
        group="group",
        name=name,
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 1, 1),
        time_range=None,
    )
    window.save()
    return window


def test_build_index_one_window(dummy_dataset: Dataset) -> None:
    """Ensure that the index is built correctly with one window.

    We check the window, but also the layer datas and completed layers.
    """
    window = make_window(dummy_dataset, "name")
    layer_data = WindowLayerData("layer", [[{"name": "x"}]])
    window.save_layer_datas({"layer": layer_data})
    layer_dir = window.get_layer_dir("layer")
    layer_dir.mkdir(parents=True)
    window.mark_layer_completed("layer")

    index = DatasetIndex.build_index(dummy_dataset, workers=1)

    assert len(index.windows) == 1
    assert index.windows[0].get_metadata() == window.get_metadata()
    assert len(index.layer_datas[window.name]) == 1
    assert index.layer_datas[window.name][0].serialize() == layer_data.serialize()
    assert index.completed_layers[window.name] == [("layer", 0)]


def test_index_used_for_loading_windows(dummy_dataset: Dataset) -> None:
    """Verify that the index is used to load windows, when it exists.

    To do so, we create a window, build the index, and then create another window, but
    verify only the first window is returned in a call to
    """
    window1 = make_window(dummy_dataset, "window1")
    DatasetIndex.build_index(dummy_dataset, workers=1).save_index(dummy_dataset.path)
    make_window(dummy_dataset, "window2")
    windows = dummy_dataset.load_windows()
    assert len(windows) == 1
    assert windows[0].get_metadata() == window1.get_metadata()


def test_no_index_when_loading_windows(dummy_dataset: Dataset) -> None:
    """Verify that the no_index argument to loading windows works.

    It should not use the outdated index.
    """
    DatasetIndex.build_index(dummy_dataset, workers=1).save_index(dummy_dataset.path)
    make_window(dummy_dataset, "window")
    assert len(dummy_dataset.load_windows()) == 0
    assert len(dummy_dataset.load_windows(no_index=True)) == 1


def test_index_used_for_completed_layers(dummy_dataset: Dataset) -> None:
    """Verify that the index is used for checking completed layers."""
    window = make_window(dummy_dataset, "name")
    DatasetIndex.build_index(dummy_dataset, workers=1).save_index(dummy_dataset.path)
    layer_dir = window.get_layer_dir("layer")
    layer_dir.mkdir(parents=True)
    window.mark_layer_completed("layer")

    windows = dummy_dataset.load_windows()
    assert not windows[0].is_layer_completed("layer")
