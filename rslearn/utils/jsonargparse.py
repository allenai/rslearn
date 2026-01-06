"""Custom serialization for jsonargparse."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

import jsonargparse
from rasterio.crs import CRS
from upath import UPath

from rslearn.config.dataset import LayerConfig
from rslearn.utils.geometry import ResolutionFactor

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
    x = {
        "ds_path": (str(v.ds_path) if v.ds_path is not None else None),
        "layer_config": (
            v.layer_config.model_dump(mode="json")
            if v.layer_config is not None
            else None
        ),
    }
    return x


def data_source_context_deserializer(v: dict[str, Any]) -> "DataSourceContext":
    """Deserialize DataSourceContext for jsonargparse."""
    # We lazily import these to avoid cyclic dependency.
    from rslearn.data_sources.data_source import DataSourceContext

    return DataSourceContext(
        ds_path=(UPath(v["ds_path"]) if v["ds_path"] is not None else None),
        layer_config=(
            LayerConfig.model_validate(v["layer_config"])
            if v["layer_config"] is not None
            else None
        ),
    )


def resolution_factor_serializer(v: ResolutionFactor) -> str:
    """Serialize ResolutionFactor for jsonargparse.

    Args:
        v: the ResolutionFactor object.

    Returns:
        the ResolutionFactor encoded to string
    """
    if hasattr(v, "init_args"):
        init_args = v.init_args
        return f"{init_args.numerator}/{init_args.denominator}"

    return f"{v.numerator}/{v.denominator}"


def resolution_factor_deserializer(v: int | str | dict) -> ResolutionFactor:
    """Deserialize ResolutionFactor for jsonargparse.

    Args:
        v: the encoded ResolutionFactor.

    Returns:
        the decoded ResolutionFactor object
    """
    # Handle already-instantiated ResolutionFactor
    if isinstance(v, ResolutionFactor):
        return v

    # Handle Namespace from class_path syntax (used during config save/validation)
    if hasattr(v, "init_args"):
        init_args = v.init_args
        return ResolutionFactor(
            numerator=init_args.numerator,
            denominator=init_args.denominator,
        )

    # Handle dict from class_path syntax in YAML config
    if isinstance(v, dict) and "init_args" in v:
        init_args = v["init_args"]
        return ResolutionFactor(
            numerator=init_args.get("numerator", 1),
            denominator=init_args.get("denominator", 1),
        )

    if isinstance(v, int):
        return ResolutionFactor(numerator=v)
    elif isinstance(v, str):
        parts = v.split("/")
        if len(parts) == 1:
            return ResolutionFactor(numerator=int(parts[0]))
        elif len(parts) == 2:
            return ResolutionFactor(
                numerator=int(parts[0]),
                denominator=int(parts[1]),
            )
        else:
            raise ValueError("expected resolution factor to be of the form x or 1/x")
    else:
        raise ValueError("expected resolution factor to be str or int")


def init_jsonargparse() -> None:
    """Initialize custom jsonargparse serializers."""
    global INITIALIZED
    if INITIALIZED:
        return
    jsonargparse.typing.register_type(CRS, crs_serializer, crs_deserializer)
    jsonargparse.typing.register_type(
        datetime, datetime_serializer, datetime_deserializer
    )
    jsonargparse.typing.register_type(
        ResolutionFactor, resolution_factor_serializer, resolution_factor_deserializer
    )

    from rslearn.data_sources.data_source import DataSourceContext

    jsonargparse.typing.register_type(
        DataSourceContext,
        data_source_context_serializer,
        data_source_context_deserializer,
    )

    INITIALIZED = True
