import importlib

from rslearn.config import LayerConfig

from .data_source import DataSource, Item


def data_source_from_config(config: LayerConfig) -> DataSource:
    """Loads a data source from config dict.

    Args:
        name: the class name of the data source
        kwargs: the arguments to pass to the data
    """
    name = config.data_source.name
    module_name = ".".join(name.split(".")[:-1])
    class_name = name.split(".")[-1]
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_.from_config(config)


__all__ = (
    "Item",
    "DataSource",
    "data_source_from_config",
)
