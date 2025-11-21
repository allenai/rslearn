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

__all__ = (
    "DataSource",
    "DataSourceContext",
    "Item",
    "ItemLookupDataSource",
    "RetrieveItemDataSource",
    "data_source_from_config",
)
