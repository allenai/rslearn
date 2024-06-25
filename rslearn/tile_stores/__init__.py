"""Tile stores that store ingested raster and vector data before materialization."""

from rslearn.config import TileStoreConfig
from rslearn.utils import FileAPI

from .file import FileTileStore
from .tile_store import LayerMetadata, PrefixedTileStore, TileStore, TileStoreLayer

registry = {"file": FileTileStore}


def load_tile_store(config: TileStoreConfig, ds_file_api: FileAPI) -> TileStore:
    """Load a tile store from a configuration.

    Args:
        config: the tile store configuration.
        ds_file_api: the FileAPI of the dataset.
    """
    return registry[config.name].from_config(config, ds_file_api)


__all__ = (
    "FileTileStore",
    "LayerMetadata",
    "PrefixedTileStore",
    "TileStore",
    "TileStoreLayer",
    "load_tile_store",
)
