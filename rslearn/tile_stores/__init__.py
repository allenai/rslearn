"""Tile stores that store ingested raster and vector data before materialization."""

from upath import UPath

from rslearn.config import TileStoreConfig

from .file import FileTileStore
from .tile_store import (
    LayerMetadata,
    PrefixedTileStore,
    TileStore,
    TileStoreLayer,
    get_tile_store_for_layer,
)

registry = {"file": FileTileStore}


def load_tile_store(config: TileStoreConfig, ds_path: UPath) -> TileStore:
    """Load a tile store from a configuration.

    Args:
        config: the tile store configuration.
        ds_path: the dataset root path.
    """
    return registry[config.name].from_config(config, ds_path)


__all__ = (
    "FileTileStore",
    "LayerMetadata",
    "PrefixedTileStore",
    "TileStore",
    "TileStoreLayer",
    "load_tile_store",
    "get_tile_store_for_layer",
)
