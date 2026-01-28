"""Data sources.

A DataSource represents a source from which raster and vector data corresponding to
spatiotemporal windows can be retrieved.

A DataSource consists of items that can be ingested, like Sentinel-2 scenes or
OpenStreetMap PBF files.

Each source supports operations to lookup items that match with spatiotemporal
geometries, and ingest those items.
"""

from .data_source import (
    DataSource,
    DataSourceContext,
    Item,
    ItemLookupDataSource,
    RetrieveItemDataSource,
)
from .tile_store_data_source import TileStoreDataSource

__all__ = (
    "DataSource",
    "DataSourceContext",
    "Item",
    "ItemLookupDataSource",
    "RetrieveItemDataSource",
    "TileStoreDataSource",
    "data_source_from_config",
)
