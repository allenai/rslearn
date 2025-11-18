"""Data sources.

A DataSource represents a source from which raster and vector data corresponding to
spatiotemporal windows can be retrieved.

A DataSource consists of items that can be ingested, like Sentinel-2 scenes or
OpenStreetMap PBF files.

Each source supports operations to lookup items that match with spatiotemporal
geometries, and ingest those items.
"""

import copy
import functools
from typing import Any

import jsonargparse

from rslearn.config import LayerConfig
from rslearn.dataset import Dataset
from rslearn.log_utils import get_logger
from rslearn.utils.jsonargparse import data_source_context_serializer, init_jsonargparse

from .data_source import (
    DataSource,
    DataSourceContext,
    Item,
    ItemLookupDataSource,
    RetrieveItemDataSource,
)

logger = get_logger(__name__)


@functools.cache
def data_source_from_config(config: LayerConfig, dataset: Dataset) -> DataSource:
    """Loads a data source from config dict.

    Args:
        config: the LayerConfig containing this data source.
        dataset: the underlying dataset.
    """
    logger.debug("getting a data source for dataset at %s", dataset.path)
    if config.data_source is None:
        raise ValueError("The layer does not specify a data source")

    # Inject the DataSourceContext into the args.
    context = DataSourceContext(
        dataset=dataset,
        layer_config=config,
    )
    ds_config: dict[str, Any] = {
        "class_path": config.data_source.class_path,
        "init_args": copy.deepcopy(config.data_source.init_args),
    }
    ds_config["init_args"]["context"] = data_source_context_serializer(context)

    # Now we can parse with jsonargparse.
    init_jsonargparse()
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--data_source", type=DataSource)
    cfg = parser.parse_object({"data_source": ds_config})
    data_source = parser.instantiate_classes(cfg).data_source
    return data_source


__all__ = (
    "DataSource",
    "DataSourceContext",
    "Item",
    "ItemLookupDataSource",
    "RetrieveItemDataSource",
    "data_source_from_config",
)
