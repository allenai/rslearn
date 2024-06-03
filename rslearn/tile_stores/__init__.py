"""Tile stores that store ingested raster and vector data before materialization."""

from rslearn.config import TileStoreConfig

from .file import FileTileStore
from .tile_store import LayerMetadata, PrefixedTileStore, TileStore, TileStoreLayer

registry = {"file": FileTileStore}


def load_tile_store(config: TileStoreConfig, root_dir: str) -> TileStore:
    """Load a tile store from a configuration.

    Args:
        config: the tile store configuration.
        root_dir: root directory from which paths in the config should be computed
            (usually the dataset root).
    """
    return registry[config.name].from_config(config, root_dir)


__all__ = (
    "FileTileStore",
    "LayerMetadata",
    "PrefixedTileStore",
    "TileStore",
    "TileStoreLayer",
    "load_tile_store",
)
