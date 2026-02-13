import json
import pathlib
import sys
from typing import Any

import shapely
from upath import UPath

import rslearn.main
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import Item
from rslearn.dataset import Dataset, Window, WindowLayerData
from rslearn.dataset.storage.sqlite import SQLiteWindowStorageFactory
from rslearn.utils.geometry import STGeometry


class TestMigrate:
    def test_dataset_migrate_to_sqlite(
        self, tmp_path: pathlib.Path, monkeypatch: Any
    ) -> None:
        ds_path = UPath(tmp_path) / "dataset"
        ds_path.mkdir(parents=True, exist_ok=True)
        with (ds_path / "config.json").open("w") as f:
            json.dump({"layers": {}}, f)

        dataset = Dataset(ds_path)
        window = Window(
            storage=dataset.storage,
            group="group",
            name="window",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 4, 4),
            time_range=None,
        )
        window.save()
        item = Item(
            name="item",
            geometry=STGeometry(WGS84_PROJECTION, shapely.box(*window.bounds), None),
        )
        window.save_layer_datas(
            {
                "layer_name": WindowLayerData(
                    "layer_name", serialized_item_groups=[[item.serialize()]]
                )
            }
        )
        window.get_layer_dir("layer_name").mkdir(parents=True, exist_ok=True)
        window.mark_layer_completed("layer_name", group_idx=0)

        storage_config = json.dumps(
            {
                "class_path": "rslearn.dataset.storage.sqlite.SQLiteWindowStorageFactory",
                "init_args": {},
            }
        )
        args = [
            "rslearn",
            "dataset",
            "migrate",
            "--root",
            str(ds_path),
            "--storage-config",
            storage_config,
        ]
        monkeypatch.setattr(sys, "argv", args)
        rslearn.main.main()

        sqlite_storage = SQLiteWindowStorageFactory().get_storage(ds_path)
        migrated_windows = sqlite_storage.get_windows()
        assert len(migrated_windows) == 1
        assert migrated_windows[0].group == "group"
        assert migrated_windows[0].name == "window"
        source_layer_datas = window.load_layer_datas()
        migrated_layer_datas = sqlite_storage.get_layer_datas("group", "window")
        assert set(migrated_layer_datas.keys()) == set(source_layer_datas.keys())
        assert sqlite_storage.is_layer_completed("group", "window", "layer_name", 0)
