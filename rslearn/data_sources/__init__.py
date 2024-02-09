import importlib
from typing import Any

from .data_source import DataSource, Item, QueryConfig, SpaceMode, TimeMode
from .raster_source import DType, RasterFormat, RasterOptions


def data_source_from_config(config: dict[str, Any]) -> DataSource:
    """Loads a data source from config dict.

    Args:
        name: the class name of the data source
        kwargs: the arguments to pass to the data
    """
    module_name = ".".join(config["name"].split(".")[:-1])
    class_name = config["name"].split(".")[-1]
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_.from_config(config)


__all__ = (
    "DataSource",
    "Item",
    "QueryConfig",
    "SpaceMode",
    "TimeMode",
    "DType",
    "RasterFormat",
    "RasterOptions",
    "data_source_from_config",
)
