"""Abstract classes for dataset metadata storage."""

import abc
from typing import TYPE_CHECKING

from upath import UPath

if TYPE_CHECKING:
    from rslearn.dataset.window import Window, WindowLayerData


class DatasetStorage(abc.ABC):
    """An abstract class for the storage backend for a particular rslearn dataset.

    This is instantiated by a DatasetStorageFactory.
    """

    @abc.abstractmethod
    def get_window_root(self, group: str, name: str) -> UPath:
        """Get the path where the window should be stored."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_windows(
        self,
        groups: list[str] | None = None,
        names: list[str] | None = None,
        show_progress: bool = False,
        workers: int = 0,
    ) -> list["Window"]:
        """Load the windows in the dataset.

        Args:
            groups: an optional list of groups to filter loading
            names: an optional list of window names to filter loading
            show_progress: whether to show tqdm progress bar
            workers: number of parallel workers, default 0 (use main thread only to load windows)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def create_or_update_window(self, window: "Window") -> None:
        """Create or update the window.

        An existing window is only updated if there is one with the same name and group.

        If there is a window with the same name but a different group, the behavior is
        undefined.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_layer_datas(self, group: str, name: str) -> dict[str, "WindowLayerData"]:
        """Get the window layer datas for the specified window.

        Args:
            group: the window group.
            name: the window name.

        Returns:
            a dict mapping from the layer name to the layer data for that layer, if one
                was previously saved.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_layer_datas(
        self, group: str, name: str, layer_datas: dict[str, "WindowLayerData"]
    ) -> None:
        """Set the window layer datas for the specified window."""
        raise NotImplementedError

    @abc.abstractmethod
    def list_completed_layers(self, group: str, name: str) -> list[tuple[str, int]]:
        """List the layers available for this window that are completed.

        Args:
            group: the window group.
            name: the window name.

        Returns:
            a list of (layer_name, group_idx) completed layers.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_layer_completed(
        self, group: str, name: str, layer_name: str, group_idx: int = 0
    ) -> bool:
        """Check whether the specified layer is completed in the given window.

        Completed means there is data in the layer and the data has been written
        (materialized).

        Args:
            group: the window group.
            name: the window name.
            layer_name: the layer name.
            group_idx: the index of the group within the layer.

        Returns:
            whether the layer is completed.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def mark_layer_completed(
        self, group: str, name: str, layer_name: str, group_idx: int = 0
    ) -> None:
        """Mark the specified layer completed for the given window.

        This must be done after the contents of the layer have been written. If a layer
        has multiple groups, the caller should wait until the contents of all groups
        have been written before marking them completed; this is because, when
        materializing a window, we skip materialization if the first group
        (group_idx=0) is marked completed.

        Args:
            group: the window group.
            name: the window name.
            layer_name: the layer name.
            group_idx: the index of the group within the layer.
        """
        raise NotImplementedError


class DatasetStorageFactory(abc.ABC):
    """An abstract class for a configurable storage backend for rslearn dataset metadata.

    The dataset config configures a DatasetStorageFactory, whihc creates a
    DatasetStorage given a dataset path.
    """

    @abc.abstractmethod
    def get_storage(self, ds_path: UPath) -> DatasetStorage:
        """Get a DatasetStorage for the given dataset path."""
        raise NotImplementedError
