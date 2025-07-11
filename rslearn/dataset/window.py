"""rslearn windows."""

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

import shapely
from upath import UPath

from rslearn.log_utils import get_logger
from rslearn.utils import Projection, STGeometry
from rslearn.utils.fsspec import open_atomic

if TYPE_CHECKING:
    from .index import DatasetIndex

logger = get_logger(__name__)

LAYERS_DIRECTORY_NAME = "layers"


def get_window_layer_dir(
    window_path: UPath, layer_name: str, group_idx: int = 0
) -> UPath:
    """Get the directory containing materialized data for the specified layer.

    Args:
        window_path: the window directory.
        layer_name: the layer name.
        group_idx: the index of the group within the layer to get the directory
            for (default 0).

    Returns:
        the path where data is or should be materialized.
    """
    if group_idx == 0:
        folder_name = layer_name
    else:
        folder_name = f"{layer_name}.{group_idx}"
    return window_path / LAYERS_DIRECTORY_NAME / folder_name


def get_layer_and_group_from_dir_name(layer_dir_name: str) -> tuple[str, int]:
    """Get the layer name and group index from the layer directory name.

    Args:
        layer_dir_name: the name of the layer folder.

    Returns:
        a tuple (layer_name, group_idx)
    """
    if "." in layer_dir_name:
        parts = layer_dir_name.split(".")
        if len(parts) != 2:
            raise ValueError(
                f"expected layer directory name {layer_dir_name} to only contain one '.'"
            )
        return (parts[0], int(parts[1]))
    else:
        return (layer_dir_name, 0)


def get_window_raster_dir(
    window_path: UPath, layer_name: str, bands: list[str], group_idx: int = 0
) -> UPath:
    """Get the directory where the raster is materialized.

    Args:
        window_path: the window directory.
        layer_name: the layer name
        bands: the bands in the raster. It should match a band set defined for this
            layer.
        group_idx: the index of the group within the layer.

    Returns:
        the directory containing the raster.
    """
    return get_window_layer_dir(window_path, layer_name, group_idx) / "_".join(bands)


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
        path: UPath,
        group: str,
        name: str,
        projection: Projection,
        bounds: tuple[int, int, int, int],
        time_range: tuple[datetime, datetime] | None,
        options: dict[str, Any] = {},
        index: "DatasetIndex | None" = None,
    ) -> None:
        """Creates a new Window instance.

        A window stores metadata about one spatiotemporal window in a dataset, that is
        stored in metadata.json.

        Args:
            path: the directory of this window
            group: the group the window belongs to
            name: the unique name for this window
            projection: the projection of the window
            bounds: the bounds of the window in pixel coordinates
            time_range: optional time range of the window
            options: additional options (?)
            index: DatasetIndex if it is available
        """
        self.path = path
        self.group = group
        self.name = name
        self.projection = projection
        self.bounds = bounds
        self.time_range = time_range
        self.options = options
        self.index = index

    def get_geometry(self) -> STGeometry:
        """Computes the STGeometry corresponding to this window."""
        return STGeometry(
            projection=self.projection,
            shp=shapely.geometry.box(*self.bounds),
            time_range=self.time_range,
        )

    def load_layer_datas(self) -> dict[str, WindowLayerData]:
        """Load layer datas describing items in retrieved layers from items.json."""
        # Load from index if it is available.
        if self.index is not None:
            layer_datas = self.index.layer_datas.get(self.name, [])

        else:
            items_fname = self.path / "items.json"
            if not items_fname.exists():
                return {}
            with items_fname.open("r") as f:
                layer_datas = [
                    WindowLayerData.deserialize(layer_data)
                    for layer_data in json.load(f)
                ]

        return {layer_data.layer_name: layer_data for layer_data in layer_datas}

    def save_layer_datas(self, layer_datas: dict[str, WindowLayerData]) -> None:
        """Save layer datas to items.json."""
        json_data = [layer_data.serialize() for layer_data in layer_datas.values()]
        items_fname = self.path / "items.json"
        logger.info(f"Saving window items to {items_fname}")
        with open_atomic(items_fname, "w") as f:
            json.dump(json_data, f)

    def get_layer_dir(self, layer_name: str, group_idx: int = 0) -> UPath:
        """Get the directory containing materialized data for the specified layer.

        Args:
            layer_name: the layer name.
            group_idx: the index of the group within the layer to get the directory
                for (default 0).

        Returns:
            the path where data is or should be materialized.
        """
        return get_window_layer_dir(self.path, layer_name, group_idx)

    def is_layer_completed(self, layer_name: str, group_idx: int = 0) -> bool:
        """Check whether the specified layer is completed.

        Completed means there is data in the layer and the data has been written
        (materialized).

        Args:
            layer_name: the layer name.
            group_idx: the index of the group within the layer.

        Returns:
            whether the layer is completed
        """
        # Use the index to speed up the completed check if it is available.
        if self.index is not None:
            return (layer_name, group_idx) in self.index.completed_layers.get(
                self.name, []
            )

        layer_dir = self.get_layer_dir(layer_name, group_idx)
        return (layer_dir / "completed").exists()

    def mark_layer_completed(self, layer_name: str, group_idx: int = 0) -> None:
        """Mark the specified layer completed.

        This must be done after the contents of the layer have been written. If a layer
        has multiple groups, the caller should wait until the contents of all groups
        have been written before marking them completed; this is because, when
        materializing a window, we skip materialization if the first group
        (group_idx=0) is marked completed.

        Args:
            layer_name: the layer name.
            group_idx: the index of the group within the layer.
        """
        layer_dir = self.get_layer_dir(layer_name, group_idx)
        (layer_dir / "completed").touch()

    def get_raster_dir(
        self, layer_name: str, bands: list[str], group_idx: int = 0
    ) -> UPath:
        """Get the directory where the raster is materialized.

        Args:
            layer_name: the layer name
            bands: the bands in the raster. It should match a band set defined for this
                layer.
            group_idx: the index of the group within the layer.

        Returns:
            the directory containing the raster.
        """
        return get_window_raster_dir(self.path, layer_name, bands, group_idx)

    def get_metadata(self) -> dict[str, Any]:
        """Returns the window metadata dictionary."""
        return {
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

    def save(self) -> None:
        """Save the window metadata to its root directory."""
        self.path.mkdir(parents=True, exist_ok=True)
        metadata_path = self.path / "metadata.json"
        logger.debug(f"Saving window metadata to {metadata_path}")
        with open_atomic(metadata_path, "w") as f:
            json.dump(self.get_metadata(), f)

    @staticmethod
    def from_metadata(path: UPath, metadata: dict[str, Any]) -> "Window":
        """Create a Window from its path and metadata dictionary.

        Args:
            path: the root directory of the window.
            metadata: the window metadata.

        Returns:
            the Window
        """
        # Ensure bounds is converted from list to tuple.
        bounds = (
            metadata["bounds"][0],
            metadata["bounds"][1],
            metadata["bounds"][2],
            metadata["bounds"][3],
        )
        return Window(
            path=path,
            group=metadata["group"],
            name=metadata["name"],
            projection=Projection.deserialize(metadata["projection"]),
            bounds=bounds,
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
    def load(path: UPath) -> "Window":
        """Load a Window from a UPath.

        Args:
            path: the root directory of the window

        Returns:
            the Window
        """
        metadata_fname = path / "metadata.json"
        with metadata_fname.open("r") as f:
            metadata = json.load(f)
        return Window.from_metadata(path, metadata)

    @staticmethod
    def get_window_root(ds_path: UPath, group: str, name: str) -> UPath:
        """Gets the root directory of a window.

        Args:
            ds_path: the dataset root directory
            group: the group of the window
            name: the name of the window
        Returns:
            the path for the window
        """
        return ds_path / "windows" / group / name
