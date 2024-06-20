"""rslearn windows."""

import json
from datetime import datetime
from typing import Any, Optional

import shapely

from rslearn.utils import Projection, STGeometry, FileAPI


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
        file_api: FileAPI,
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
            file_api: the FileAPI rooted at this window
            group: the group the window belongs to
            name: the unique name for this window
            projection: the projection of the window
            bounds: the bounds of the window in pixel coordinates
            time_range: optional time range of the window
            options: additional options (?)
        """
        self.file_api = file_api
        self.group = group
        self.name = name
        self.projection = projection
        self.bounds = bounds
        self.time_range = time_range
        self.options = options

    def save(self) -> None:
        """Save the window metadata to its root directory."""
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
        with self.file_api.open_atomic("metadata.json", "w") as f:
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
        if not self.file_api.exists("items.json"):
            return {}
        with self.file_api.open("items.json", "r") as f:
            layer_datas = [
                WindowLayerData.deserialize(layer_data) for layer_data in json.load(f)
            ]
        return {layer_data.layer_name: layer_data for layer_data in layer_datas}

    def save_layer_datas(self, layer_datas: dict[str, WindowLayerData]) -> None:
        """Save layer datas to items.json."""
        json_data = [layer_data.serialize() for layer_data in layer_datas.values()]
        with self.file_api.open_atomic("items.json", "w") as f:
            json.dump(json_data, f)

    @staticmethod
    def load(file_api: FileAPI) -> "Window":
        """Load a Window from a FileAPI.

        Args:
            file_api: the FileAPI

        Returns:
            the Window
        """
        with file_api.open("metadata.json", "r") as f:
            metadata = json.load(f)
        return Window(
            file_api=file_api,
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
    def get_window_root(ds_file_api: FileAPI, group: str, name: str) -> FileAPI:
        """Gets the root directory of a window.

        Args:
            ds_file_api: the dataset FileAPI
            group: the group of the window
            name: the name of the window
        Returns:
            the FileAPI for the window
        """
        return ds_file_api.get_folder(ds_file_api.join("windows", group, name))
