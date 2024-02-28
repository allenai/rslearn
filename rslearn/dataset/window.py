import json
import os
from datetime import datetime
from typing import Any, Optional

import shapely

from rslearn.utils import Projection, STGeometry, open_atomic


class WindowLayerData:
    def __init__(
        self,
        layer_name: str,
        serialized_item_groups: list[list[Any]],
        materialized: bool = False,
    ) -> None:
        self.layer_name = layer_name
        self.serialized_item_groups = serialized_item_groups
        self.materialized = materialized

    def serialize(self) -> dict:
        return {
            "layer_name": self.layer_name,
            "serialized_item_groups": self.serialized_item_groups,
            "materialized": self.materialized,
        }

    @staticmethod
    def deserialize(d: dict) -> "WindowLayerData":
        return WindowLayerData(
            layer_name=d["layer_name"],
            serialized_item_groups=d["serialized_item_groups"],
            materialized=d["materialized"],
        )


class Window:
    def __init__(
        self,
        window_root: str,
        group: str,
        name: str,
        projection: Projection,
        bounds: tuple[int, int, int, int],
        time_range: Optional[tuple[datetime, datetime]],
        options: dict[str, Any] = {},
    ) -> None:
        """Creates a new Window instance.

        A window stores metadata about one spatiotemporal window in a dataset, that is
        stored in metadata.json.

        Args:
            window_root: the root directory of this window
            group: the group the window belongs to
            name: the unique name for this window
            projection: the projection of the window
            bounds: the bounds of the window in pixel coordinates
            time_range: optional time range of the window
            options: additional options (?)
        """
        self.window_root = window_root
        self.group = group
        self.name = name
        self.projection = projection
        self.bounds = bounds
        self.time_range = time_range
        self.options = options

    def save(self) -> None:
        os.makedirs(self.window_root, exist_ok=True)
        metadata = {
            "group": self.group,
            "name": self.name,
            "projection": self.projection.serialize(),
            "bounds": self.bounds,
            "time_range": (
                [self.time_range[0].isoformat(), self.time_range[1].isoformat()]
                if self.time_range
                else None
            ),
            "options": self.options,
        }
        with open_atomic(os.path.join(self.window_root, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    def get_geometry(self) -> STGeometry:
        """Computes the STGeometry corresponding to this window."""
        return STGeometry(
            projection=self.projection,
            shp=shapely.geometry.box(*self.bounds),
            time_range=self.time_range,
        )

    def load_layer_datas(self) -> dict[str, WindowLayerData]:
        """Load layer datas describing items in retrieved layers from items.json."""
        items_fname = os.path.join(self.window_root, "items.json")
        if not os.path.exists(items_fname):
            return {}
        with open(items_fname) as f:
            layer_datas = [
                WindowLayerData.deserialize(layer_data) for layer_data in json.load(f)
            ]
        return {layer_data.layer_name: layer_data for layer_data in layer_datas}

    def save_layer_datas(self, layer_datas: dict[str, WindowLayerData]) -> None:
        """Save layer datas to items.json."""
        json_data = [layer_data.serialize() for layer_data in layer_datas.values()]
        with open_atomic(os.path.join(self.window_root, "items.json"), "w") as f:
            json.dump(json_data, f)

    @staticmethod
    def load(window_root: str) -> "Window":
        with open(os.path.join(window_root, "metadata.json"), "r") as f:
            metadata = json.load(f)
        return Window(
            window_root=window_root,
            group=metadata["group"],
            name=metadata["name"],
            projection=Projection.deserialize(metadata["projection"]),
            bounds=metadata["bounds"],
            time_range=(
                (
                    datetime.fromisoformat(metadata["time_range"][0]),
                    datetime.fromisoformat(metadata["time_range"][1]),
                )
                if metadata["time_range"]
                else None
            ),
            options=metadata["options"],
        )

    @staticmethod
    def get_window_root(ds_root: str, group: str, name: str) -> str:
        """Gets the root directory of a window.

        Args:
            ds_root: the dataset root directory
            group: the group of the window
            name: the name of the window
        Returns:
            the window root directory
        """
        return os.path.join(ds_root, "windows", group, name)
