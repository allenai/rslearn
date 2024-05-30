"""Data sources.

A DataSource represents a source from which raster and vector data corresponding to
spatiotemporal windows can be retrieved.

A DataSource consists of items that can be ingested, like Sentinel-2 scenes or
OpenStreetMap PBF files.

Each source supports operations to lookup items that match with spatiotemporal
geometries, and ingest those items.
"""

import importlib

from rslearn.config import LayerConfig

from .data_source import DataSource, Item, ItemLookupDataSource, RetrieveItemDataSource


def data_source_from_config(config: LayerConfig, root_dir: str) -> DataSource:
    """Loads a data source from config dict.

    Args:
        config: the LayerConfig containing this data source.
        root_dir: dataset root directory.
    """
    name = config.data_source.name
    module_name = ".".join(name.split(".")[:-1])
    class_name = name.split(".")[-1]
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_.from_config(config, root_dir=root_dir)


__all__ = (
    "DataSource",
    "Item",
    "ItemLookupDataSource",
    "RetrieveItemDataSource",
    "data_source_from_config",
)
