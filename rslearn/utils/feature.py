"""GeoJSON-like feature class."""

import json
from typing import Any

import shapely

from .geometry import Projection, STGeometry


class Feature:
    """A GeoJSON-like feature that contains one vector geometry."""

    def __init__(self, geometry: STGeometry, properties: dict[str, Any] | None = {}):
        """Initialize a new Feature.

        Args:
            geometry: the STGeometry
            properties: properties of the feature
        """
        self.geometry = geometry
        self.properties = properties

    def to_geojson(self) -> dict[str, Any]:
        """Returns a GeoJSON dict corresponding to this feature."""
        return {
            "type": "Feature",
            "properties": self.properties,
            "geometry": json.loads(shapely.to_geojson(self.geometry.shp)),
        }

    def to_projection(self, projection: Projection) -> "Feature":
        """Converts this Feature to the target projection.

        Args:
            projection: the target projection

        Returns:
            a new Feature in the target projection
        """
        return Feature(self.geometry.to_projection(projection), self.properties)

    @staticmethod
    def from_geojson(projection: Projection, d: dict[str, Any]) -> "Feature":
        """Construct a Feature from a GeoJSON encoding.

        Args:
            projection: the projection of the GeoJSON feature
            d: the GeoJSON feature dict

        Returns:
            a Feature representing the specified geometry
        """
        shp = shapely.geometry.shape(d["geometry"])
        return Feature(STGeometry(projection, shp, None), d.get("properties", {}))
