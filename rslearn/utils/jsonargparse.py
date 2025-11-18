"""Custom serialization for jsonargparse."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

import jsonargparse
from rasterio.crs import CRS
from upath import UPath

from rslearn.config.dataset import LayerConfig

if TYPE_CHECKING:
    from rslearn.data_sources.data_source import DataSourceContext

INITIALIZED = False


def crs_serializer(v: CRS) -> str:
    """Serialize CRS for jsonargparse.

    Args:
        v: the CRS object.

    Returns:
        the CRS encoded to string
    """
    return v.to_string()


def crs_deserializer(v: str) -> CRS:
    """Deserialize CRS for jsonargparse.

    Args:
        v: the encoded CRS.

    Returns:
        the decoded CRS object
    """
    return CRS.from_string(v)


def datetime_serializer(v: datetime) -> str:
    """Serialize datetime for jsonargparse.

    Args:
        v: the datetime object.

    Returns:
        the datetime encoded to string
    """
    return v.isoformat()


def datetime_deserializer(v: str) -> datetime:
    """Deserialize datetime for jsonargparse.

    Args:
        v: the encoded datetime.

    Returns:
        the decoded datetime object
    """
    return datetime.fromisoformat(v)


def data_source_context_serializer(v: "DataSourceContext") -> dict[str, Any]:
    """Serialize DataSourceContext for jsonargparse."""
    return {
        "dataset_path": str(v.dataset.path) if v.dataset is not None else None,
        "layer_config": v.layer_config.model_dump(mode="json")
        if v.layer_config is not None
        else None,
    }


def data_source_context_deserializer(v: dict[str, Any]) -> "DataSourceContext":
    """Deserialize DataSourceContext for jsonargparse."""
    # We lazily import these to avoid cyclic dependency.
    from rslearn.data_sources.data_source import DataSourceContext
    from rslearn.dataset.dataset import Dataset

    return DataSourceContext(
        dataset=Dataset(UPath(v["dataset_path"]))
        if v["dataset_path"] is not None
        else None,
        layer_config=LayerConfig.model_validate(v["layer_config"])
        if v["layer_config"] is not None
        else None,
    )


def init_jsonargparse() -> None:
    """Initialize custom jsonargparse serializers."""
    global INITIALIZED
    if INITIALIZED:
        return
    jsonargparse.typing.register_type(CRS, crs_serializer, crs_deserializer)
    jsonargparse.typing.register_type(
        datetime, datetime_serializer, datetime_deserializer
    )

    from rslearn.data_sources.data_source import DataSourceContext

    jsonargparse.typing.register_type(
        DataSourceContext,
        data_source_context_serializer,
        data_source_context_deserializer,
    )

    INITIALIZED = True
