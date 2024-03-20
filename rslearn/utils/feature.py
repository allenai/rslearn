import json
from typing import Any, Optional

import shapely

from .geometry import Projection, STGeometry


class Feature:
    def __init__(self, geometry: STGeometry, properties: Optional[dict[str, Any]] = {}):
        self.geometry = geometry
        self.properties = properties

    def to_geojson(self) -> dict[str, Any]:
        return {
            "type": "Feature",
            "properties": self.properties,
            "geometry": json.loads(shapely.to_geojson(self.geometry.shp)),
        }

    def to_projection(self, projection: Projection) -> "Feature":
        return Feature(
            self.geometry.to_projection(projection),
            self.properties,
        )

    @staticmethod
    def from_geojson(projection: Projection, d: dict[str, Any]):
        shp = shapely.geometry.shape(d["geometry"])
        return Feature(STGeometry(projection, shp, None), d["properties"])
