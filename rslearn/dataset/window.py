"""rslearn windows."""

import warnings
from datetime import datetime
from typing import Any

import shapely
from upath import UPath

from rslearn.dataset.storage.storage import WindowStorage
from rslearn.log_utils import get_logger
from rslearn.utils import Projection, STGeometry
from rslearn.utils.raster_format import get_bandset_dirname

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
    dirname = get_bandset_dirname(bands)
    return get_window_layer_dir(window_path, layer_name, group_idx) / dirname


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
        storage: WindowStorage,
        group: str,
        name: str,
        projection: Projection,
        bounds: tuple[int, int, int, int],
        time_range: tuple[datetime, datetime] | None,
        options: dict[str, Any] = {},
    ) -> None:
        """Creates a new Window instance.

        A window stores metadata about one spatiotemporal window in a dataset, that is
        stored in metadata.json.

        Args:
            storage: the dataset storage for the underlying rslearn dataset.
            group: the group the window belongs to
            name: the unique name for this window
            projection: the projection of the window
            bounds: the bounds of the window in pixel coordinates
            time_range: optional time range of the window
            options: additional options (?)
        """
        self.storage = storage
        self.group = group
        self.name = name
        self.projection = projection
        self.bounds = bounds
        self.time_range = time_range
        self.options = options

    def get_geometry(self) -> STGeometry:
        """Computes the STGeometry corresponding to this window."""
        return STGeometry(
            projection=self.projection,
            shp=shapely.geometry.box(*self.bounds),
            time_range=self.time_range,
        )

    @property
    def window_root(self) -> UPath:
        """Get the root directory of this window.

        Returns:
            the path for this window's root directory.
        """
        return self.storage.get_window_root(self.group, self.name)

    def load_layer_datas(self) -> dict[str, WindowLayerData]:
        """Load layer datas describing items in retrieved layers from items.json."""
        return self.storage.get_layer_datas(self.group, self.name)

    def save_layer_datas(self, layer_datas: dict[str, WindowLayerData]) -> None:
        """Save layer datas to items.json."""
        self.storage.save_layer_datas(self.group, self.name, layer_datas)

    def list_completed_layers(self) -> list[tuple[str, int]]:
        """List the layers available for this window that are completed.

        Returns:
            a list of (layer_name, group_idx) completed layers.
        """
        return self.storage.list_completed_layers(self.group, self.name)

    def get_vector_layer_dir(self, layer_name: str, group_idx: int = 0) -> UPath:
        """Get the directory for vector layer data.

        For raster data, use the RasterDataStorage abstraction instead.

        Args:
            layer_name: the layer name.
            group_idx: the index of the group within the layer to get the directory
                for (default 0).

        Returns:
            the path where vector data is or should be materialized.
        """
        return get_window_layer_dir(self.window_root, layer_name, group_idx)

    def get_layer_dir(self, layer_name: str, group_idx: int = 0) -> UPath:
        """Get the directory containing materialized data for the specified layer.

        .. deprecated::
            Use get_vector_layer_dir for vectors, or RasterDataStorage for raster data.

        Args:
            layer_name: the layer name.
            group_idx: the index of the group within the layer to get the directory
                for (default 0).

        Returns:
            the path where data is or should be materialized.
        """
        warnings.warn(
            "get_layer_dir is deprecated. Use get_vector_layer_dir for vectors, "
            "or RasterDataStorage for raster data.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_vector_layer_dir(layer_name, group_idx)

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
        return self.storage.is_layer_completed(
            self.group, self.name, layer_name, group_idx
        )

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
        self.storage.mark_layer_completed(self.group, self.name, layer_name, group_idx)

    def get_raster_dir(
        self, layer_name: str, bands: list[str], group_idx: int = 0
    ) -> UPath:
        """Get the directory where the raster is materialized.

        .. deprecated::
            Use RasterDataStorage for reading/writing raster data. This method
            remains functional for programmatic use (e.g., custom scripts that
            create label rasters).

        Args:
            layer_name: the layer name
            bands: the bands in the raster. It should match a band set defined for this
                layer.
            group_idx: the index of the group within the layer.

        Returns:
            the directory containing the raster.
        """
        warnings.warn(
            "get_raster_dir is deprecated. Use RasterDataStorage for reading/writing "
            "raster data. This method remains functional for programmatic use.",
            DeprecationWarning,
            stacklevel=2,
        )
        return get_window_raster_dir(
            self.window_root,
            layer_name,
            bands,
            group_idx,
        )

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
        self.storage.create_or_update_window(self)

    @staticmethod
    def from_metadata(storage: WindowStorage, metadata: dict[str, Any]) -> "Window":
        """Create a Window from the WindowStorage and the window's metadata dictionary.

        Args:
            storage: the WindowStorage for the underlying dataset.
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
            storage=storage,
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
