import importlib

from .data_source import DataSource, Item, QueryConfig, SpaceMode, TimeMode
from .raster_source import DType, RasterFormat, RasterOptions


def load_raster_options(kwargs: dict) -> RasterOptions:
    """Loads a RasterOptions instance by arguments.

    Args:
        kwargs: the arguments to pass to RasterOptions.
    """
    for k, v in kwargs.items():
        if k == "format":
            kwargs[k] = RasterFormat(v)
        if k == "dtype":
            kwargs[k] = DType(v)
    return RasterOptions(**kwargs)


def load_data_source(name: str, kwargs: dict) -> DataSource:
    """Loads a data source by name and arguments.

    Args:
        name: the class name of the data source
        kwargs: the arguments to pass to the data
    """
    module_name = ".".join(name.split(".")[:-1])
    class_name = name.split(".")[-1]
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    for k, v in kwargs.items():
        if k == "raster_options":
            kwargs[k] = load_raster_options(v)
    return class_(**kwargs)


__all__ = (
    "DataSource",
    "Item",
    "QueryConfig",
    "SpaceMode",
    "TimeMode",
    "DType",
    "RasterFormat",
    "RasterOptions",
    "load_data_source",
    "load_raster_options",
)
