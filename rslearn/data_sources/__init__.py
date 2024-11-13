"""Data sources.

A DataSource represents a source from which raster and vector data corresponding to
spatiotemporal windows can be retrieved.

A DataSource consists of items that can be ingested, like Sentinel-2 scenes or
OpenStreetMap PBF files.

Each source supports operations to lookup items that match with spatiotemporal
geometries, and ingest those items.
"""

import functools
import importlib

from upath import UPath

from rslearn.config import LayerConfig
from rslearn.log_utils import get_logger

from .data_source import DataSource, Item, ItemLookupDataSource, RetrieveItemDataSource

logger = get_logger(__name__)


@functools.cache
def data_source_from_config(config: LayerConfig, ds_path: UPath) -> DataSource:
    """Loads a data source from config dict.

    Args:
        config: the LayerConfig containing this data source.
        ds_path: the dataset root directory.
    """
    logger.debug("getting a data source for dataset at %s", ds_path)
    if config.data_source is None:
        raise ValueError("No data source specified")
    name = config.data_source.name
    module_name = ".".join(name.split(".")[:-1])
    class_name = name.split(".")[-1]
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_.from_config(config, ds_path)


__all__ = (
    "DataSource",
    "Item",
    "ItemLookupDataSource",
    "RetrieveItemDataSource",
    "data_source_from_config",
)
