from rslearn.config import TileStoreConfig

from .file import FileTileStore
from .tile_store import LayerMetadata, PrefixedTileStore, TileStore, TileStoreLayer

registry = {
    "file": FileTileStore,
}


def load_tile_store(config: TileStoreConfig) -> TileStore:
    """Load a tile store from a configuration."""
    return registry[config.name].from_config(config)


__all__ = (
    "FileTileStore",
    "LayerMetadata",
    "PrefixedTileStore",
    "TileStore",
    "TileStoreLayer",
    "load_tile_store",
)
