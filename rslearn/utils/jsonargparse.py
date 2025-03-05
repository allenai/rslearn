"""Custom serialization for jsonargparse."""

import jsonargparse
from rasterio.crs import CRS


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


def init_jsonargparse() -> None:
    """Initialize custom jsonargparse serializers."""
    jsonargparse.typing.register_type(CRS, crs_serializer, crs_deserializer)
