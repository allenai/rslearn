import json
import os
from datetime import datetime
from typing import Any, Optional

import shapely
from rasterio.crs import CRS

from rslearn.utils import STGeometry


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
        group: str,
        name: str,
        crs: CRS,
        resolution: float,
        bounds: tuple[float, float, float, float],
        time_range: Optional[tuple[datetime, datetime]],
        layer_datas: dict[str, WindowLayerData] = {},
        options: dict[str, Any] = {},
    ) -> None:
        """Creates a new Window instance.

        A window stores metadata about one spatiotemporal window in a dataset, that is
        stored in metadata.json.

        Args:
            group: the group the window belongs to
            name: the unique name for this window
            crs: the projection
            resolution: resolution in projection units per pixel
            bounds: the bounds of the window in pixel coordinates
            time_range: optional time range of the window
            layer_datas: a dictionary of layer names to WindowLayerData objects
                specifying items for each layer computed via prepare
            options: additional options (?)
        """
        self.group = group
        self.name = name
        self.crs = crs
        self.resolution = resolution
        self.bounds = bounds
        self.time_range = time_range
        self.layer_datas = layer_datas
        self.options = options

    def get_root(self, ds_root: str) -> str:
        return os.path.join(ds_root, "windows", self.group, self.name)

    def save(self, ds_root: str) -> None:
        window_root = self.get_root(ds_root)
        os.makedirs(window_root, exist_ok=True)
        metadata = {
            "group": self.group,
            "name": self.name,
            "crs": self.crs.to_string(),
            "resolution": self.resolution,
            "bounds": self.bounds,
            "time_range": (
                [self.time_range[0].isoformat(), self.time_range[1].isoformat()]
                if self.time_range
                else None
            ),
            "layer_datas": [
                layer_data.serialize() for layer_data in self.layer_datas.values()
            ],
            "options": self.options,
        }
        with open(os.path.join(window_root, "metadata.json.tmp"), "w") as f:
            json.dump(metadata, f)
        os.rename(
            os.path.join(window_root, "metadata.json.tmp"),
            os.path.join(window_root, "metadata.json"),
        )

    def get_geometry(self) -> STGeometry:
        """Computes the STGeometry corresponding to this window."""
        return STGeometry(
            crs=self.crs,
            resolution=self.resolution,
            shp=shapely.geometry.box(*self.bounds),
            time_range=self.time_range,
        )

    @staticmethod
    def load(window_root: str) -> "Window":
        with open(os.path.join(window_root, "metadata.json"), "r") as f:
            metadata = json.load(f)
        layer_datas = [
            WindowLayerData.deserialize(layer_data)
            for layer_data in metadata["layer_datas"]
        ]
        return Window(
            group=metadata["group"],
            name=metadata["name"],
            crs=CRS.from_string(metadata["crs"]),
            resolution=metadata["resolution"],
            bounds=metadata["bounds"],
            time_range=(
                (
                    datetime.fromisoformat(metadata["time_range"][0]),
                    datetime.fromisoformat(metadata["time_range"][1]),
                )
                if metadata["time_range"]
                else None
            ),
            layer_datas={
                layer_data.layer_name: layer_data for layer_data in layer_datas
            },
            options=metadata["options"],
        )
