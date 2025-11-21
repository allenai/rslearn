"""Tile stores that store ingested raster and vector data before materialization."""

from typing import Any

import jsonargparse
from upath import UPath

from rslearn.config import LayerConfig
from rslearn.utils.jsonargparse import init_jsonargparse

from .default import DefaultTileStore
from .tile_store import TileStore, TileStoreWithLayer


def load_tile_store(config: dict[str, Any], ds_path: UPath) -> TileStore:
    """Load a tile store from a configuration.

    Args:
        config: the tile store configuration.
        ds_path: the dataset root path.

    Returns:
        the TileStore
    """
    init_jsonargparse()
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--tile_store", type=TileStore)
    cfg = parser.parse_object({"tile_store": config})
    tile_store = parser.instantiate_classes(cfg).tile_store
    tile_store.set_dataset_path(ds_path)
    return tile_store


def get_tile_store_with_layer(
    tile_store: TileStore, layer_name: str, layer_cfg: LayerConfig
) -> TileStoreWithLayer:
    """Get the TileStoreWithLayer for the specified layer.

    Uses alias of the layer if it is set, otherwise just the layer name.

    Args:
        tile_store: the tile store.
        layer_name: the layer name.
        layer_cfg: the layer configuration which can specify an alias.

    Returns:
        corresponding TileStoreWithLayer
    """
    if layer_cfg.alias is not None:
        return TileStoreWithLayer(tile_store, layer_cfg.alias)
    return TileStoreWithLayer(tile_store, layer_name)


__all__ = (
    "DefaultTileStore",
    "TileStore",
    "TileStoreWithLayer",
    "load_tile_store",
    "get_tile_store_with_layer",
)
