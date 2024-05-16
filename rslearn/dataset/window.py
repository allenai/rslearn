"""rslearn windows."""

import json
import os
from datetime import datetime
from typing import Any, Optional

import shapely

from rslearn.utils import Projection, STGeometry, open_atomic


class WindowLayerData:
    """Layer data for retrieved layers specifying relevant items in the data source.

    This stores the outputs from dataset prepare for a given layer.
    """

    def __init__(
        self,
        layer_name: str,
        serialized_item_groups: list[list[Any]],
        materialized: bool = False,
    ) -> None:
        """Initialize a new WindowLayerData.

        Args:
            layer_name: the layer name
            serialized_item_groups: the groups of items, where each item is serialized
                so it is JSON-encodable.
            materialized: whether the items have been materialized
        """
        self.layer_name = layer_name
        self.serialized_item_groups = serialized_item_groups
        self.materialized = materialized

    def serialize(self) -> dict:
        """Serialize a WindowLayerData is a JSON-encodable dict.

        Returns:
            the JSON-encodable dict
        """
        return {
            "layer_name": self.layer_name,
            "serialized_item_groups": self.serialized_item_groups,
            "materialized": self.materialized,
        }

    @staticmethod
    def deserialize(d: dict) -> "WindowLayerData":
        """Deserialize a WindowLayerData.

        Args:
            d: a JSON dict

        Returns:
            the WindowLayerData
        """
        return WindowLayerData(
            layer_name=d["layer_name"],
            serialized_item_groups=d["serialized_item_groups"],
            materialized=d["materialized"],
        )


class Window:
    """A spatiotemporal window in an rslearn dataset."""

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
        """Save the window metadata to its root directory."""
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
        """Load a Window from its root directory.

        Args:
            window_root: the root directory of the window

        Returns:
            the Window
        """
        with open(os.path.join(window_root, "metadata.json")) as f:
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
