"""Tests for FileWindowStorage directory scanning behavior."""

import json
from pathlib import Path

from upath import UPath

from rslearn.dataset.storage.file import FileWindowStorage
from rslearn.utils.geometry import WGS84_PROJECTION


def _write_window_metadata(dataset_path: UPath, group: str, name: str) -> None:
    window_dir = dataset_path / "windows" / group / name
    window_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "projection": WGS84_PROJECTION.serialize(),
        "bounds": [0, 0, 1, 1],
        "time_range": None,
        "options": {},
    }
    with (window_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f)


def test_get_windows_ignores_ds_store_files(tmp_path: Path) -> None:
    dataset_path = UPath(str(tmp_path / "dataset"))
    (dataset_path / "windows").mkdir(parents=True, exist_ok=True)
    with (dataset_path / "windows" / ".DS_Store").open("w") as f:
        f.write("junk")
    (dataset_path / "windows" / "group1").mkdir(parents=True, exist_ok=True)
    with (dataset_path / "windows" / "group1" / ".DS_Store").open("w") as f:
        f.write("junk")

    _write_window_metadata(dataset_path, group="group1", name="w1")

    storage = FileWindowStorage(dataset_path)
    windows = storage.get_windows()

    assert [(w.group, w.name) for w in windows] == [("group1", "w1")]


def test_get_windows_skips_non_directory_group(tmp_path: Path) -> None:
    dataset_path = UPath(str(tmp_path / "dataset"))
    (dataset_path / "windows").mkdir(parents=True, exist_ok=True)
    with (dataset_path / "windows" / ".DS_Store").open("w") as f:
        f.write("junk")

    storage = FileWindowStorage(dataset_path)
    windows = storage.get_windows(groups=[".DS_Store"])

    assert windows == []


def test_window_survives_move_to_different_group(tmp_path: Path) -> None:
    """Moving a window directory to a different group should just work."""
    dataset_path = UPath(str(tmp_path / "dataset"))
    _write_window_metadata(dataset_path, group="group1", name="w1")

    # Move the window from group1 to group2
    dst_group = dataset_path / "windows" / "group2"
    dst_group.mkdir(parents=True, exist_ok=True)
    src = dataset_path / "windows" / "group1" / "w1"
    dst = dst_group / "w1"
    src.rename(dst)

    storage = FileWindowStorage(dataset_path)
    windows = storage.get_windows()
    assert len(windows) == 1
    assert windows[0].group == "group2"
    assert windows[0].name == "w1"


def test_list_completed_layers_ignores_non_directories(tmp_path: Path) -> None:
    dataset_path = UPath(str(tmp_path / "dataset"))
    _write_window_metadata(dataset_path, group="group1", name="w1")

    layers_dir = dataset_path / "windows" / "group1" / "w1" / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)
    with (layers_dir / ".DS_Store").open("w") as f:
        f.write("junk")

    (layers_dir / "sentinel2").mkdir(parents=True, exist_ok=True)
    with (layers_dir / "sentinel2" / "completed").open("w") as f:
        f.write("")

    storage = FileWindowStorage(dataset_path)
    completed = storage.list_completed_layers(group="group1", name="w1")

    assert ("sentinel2", 0) in completed
