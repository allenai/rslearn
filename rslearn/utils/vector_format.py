import json
from typing import Any

import numpy as np
import shapely
from class_registry import ClassRegistry

from rslearn.config import VectorFormatConfig

from .feature import Feature
from .file_api import FileAPI
from .geometry import PixelBounds, Projection, STGeometry

VectorFormats = ClassRegistry()


class VectorFormat:
    def encode_vector(
        self,
        file_api: FileAPI,
        projection: Projection,
        features: list[Feature],
    ) -> None:
        """Encodes vector data.

        Args:
            file_api: the file API to write to
            projection: the projection of the raster data
            features: the vector data
        """
        raise NotImplementedError

    def decode_vector(self, file_api: FileAPI, bounds: PixelBounds) -> list[Feature]:
        """Decodes vector data.

        Args:
            file_api: the file API to read from
            bounds: the bounds of the vector data to read

        Returns:
            the vector data
        """
        raise NotImplementedError


@VectorFormats.register("tile")
class TileVectorFormat(VectorFormat):
    def __init__(self, tile_size: int = 512):
        """Initialize a new TileVectorFormat instance.

        Args:
            tile_size: the tile size (grid size in pixels), default 512
        """
        self.tile_size = tile_size

    def encode_vector(
        self,
        file_api: FileAPI,
        projection: Projection,
        features: list[Feature],
    ) -> None:
        """Encodes vector data.

        Args:
            file_api: the file API to write to
            projection: the projection of the raster data
            features: the vector data
        """
        tile_data = {}
        for feat in features:
            if not feat.geometry.shp.is_valid:
                continue
            bounds = feat.geometry.shp.bounds
            start_tile = (
                int(bounds[0]) // self.tile_size,
                int(bounds[1]) // self.tile_size,
            )
            end_tile = (
                int(bounds[2]) // self.tile_size + 1,
                int(bounds[3]) // self.tile_size + 1,
            )
            for col in range(start_tile[0], end_tile[0]):
                for row in range(start_tile[1], end_tile[1]):
                    cur_shp = feat.geometry.shp.intersection(
                        shapely.box(
                            col * self.tile_size,
                            row * self.tile_size,
                            (col + 1) * self.tile_size,
                            (row + 1) * self.tile_size,
                        )
                    )
                    cur_shp = shapely.transform(
                        cur_shp,
                        lambda array: array
                        - np.array([[col * self.tile_size, row * self.tile_size]]),
                    )
                    cur_feat = Feature(
                        STGeometry(projection, cur_shp, None), feat.properties
                    )
                    try:
                        cur_geojson = cur_feat.to_geojson()
                    except Exception as e:
                        print(e)
                        continue
                    tile = (col, row)
                    if tile not in tile_data:
                        tile_data[tile] = []
                    tile_data[tile].append(cur_geojson)

        for (col, row), geojson_features in tile_data.items():
            fc = {
                "type": "FeatureCollection",
                "features": [geojson_feat for geojson_feat in geojson_features],
            }
            with file_api.open(f"{col}_{row}.geojson", "w") as f:
                json.dump(fc, f)

    def decode_vector(self, file_api: FileAPI, bounds: PixelBounds) -> list[Feature]:
        """Decodes vector data.

        Args:
            file_api: the file API to read from
            bounds: the bounds of the vector data to read

        Returns:
            the vector data
        """
        start_tile = (bounds[0] // self.tile_size, bounds[1] // self.tile_size)
        end_tile = (
            (bounds[2] - 1) // self.tile_size + 1,
            (bounds[3] - 1) // self.tile_size + 1,
        )
        features = []
        for col in range(start_tile[0], end_tile[0]):
            for row in range(start_tile[1], end_tile[1]):
                fname = f"{col}_{row}.geojson"
                if not file_api.exists(fname):
                    continue
                with file_api.open(fname, "r") as f:
                    fc = json.load(f)
                features.extend([Feature.from_geojson(feat) for feat in fc["features"]])
        return features

    @staticmethod
    def from_config(name: str, config: dict[str, Any]) -> "TileVectorFormat":
        return TileVectorFormat(
            tile_size=config.get("tile_size", 512),
        )


@VectorFormats.register("geojson")
class GeojsonVectorFormat(VectorFormat):
    """A vector format that uses one big GeoJSON."""

    fname = "data.geojson"

    def encode_vector(
        self,
        file_api: FileAPI,
        projection: Projection,
        features: list[Feature],
    ) -> None:
        """Encodes vector data.

        Args:
            file_api: the file API to write to
            projection: the projection of the raster data
            features: the vector data
        """
        with file_api.open(self.fname, "w") as f:
            json.dump(
                {
                    "type": "FeatureCollection",
                    "features": [feat.to_geojson for feat in features],
                },
                f,
            )

    def decode_vector(self, file_api: FileAPI, bounds: PixelBounds) -> list[Feature]:
        """Decodes vector data.

        Args:
            file_api: the file API to read from
            bounds: the bounds of the vector data to read

        Returns:
            the vector data
        """
        with file_api.open(self.fname, "r") as f:
            fc = json.load(f)
        return [Feature.from_geojson(feat) for feat in fc["features"]]

    @staticmethod
    def from_config(name: str, config: dict[str, Any]) -> "GeojsonVectorFormat":
        return GeojsonVectorFormat()


def load_vector_format(config: VectorFormatConfig) -> VectorFormat:
    cls = VectorFormats.get_class(config.name)
    return cls.from_config(config.name, config.config_dict)
