"""Classes for writing vector data to a UPath."""

import json
from enum import Enum
from typing import Any

import numpy as np
import shapely
from class_registry import ClassRegistry
from rasterio.crs import CRS
from upath import UPath

from rslearn.config import VectorFormatConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.log_utils import get_logger

from .feature import Feature
from .geometry import PixelBounds, Projection, STGeometry

logger = get_logger(__name__)
VectorFormats = ClassRegistry()


class VectorFormat:
    """An abstract class for writing vector data.

    Implementations of VectorFormat should support reading and writing vector data in
    a UPath. Vector data is a list of GeoJSON-like features.
    """

    def encode_vector(
        self, path: UPath, projection: Projection, features: list[Feature]
    ) -> None:
        """Encodes vector data.

        Args:
            path: the directory to write to
            projection: the projection of the raster data
            features: the vector data
        """
        raise NotImplementedError

    def decode_vector(self, path: UPath, bounds: PixelBounds) -> list[Feature]:
        """Decodes vector data.

        Args:
            path: the directory to read from
            bounds: the bounds of the vector data to read

        Returns:
            the vector data
        """
        raise NotImplementedError


@VectorFormats.register("tile")
class TileVectorFormat(VectorFormat):
    """TileVectorFormat stores data in GeoJSON files corresponding to grid cells.

    A tile size defines the grid size in pixels. One file is created for each grid cell
    containing at least one feature. Features are written to all grid cells that they
    intersect.
    """

    def __init__(self, tile_size: int = 512):
        """Initialize a new TileVectorFormat instance.

        Args:
            tile_size: the tile size (grid size in pixels), default 512
        """
        self.tile_size = tile_size

    def encode_vector(
        self, path: UPath, projection: Projection, features: list[Feature]
    ) -> None:
        """Encodes vector data.

        Args:
            path: the directory to write to
            projection: the projection of the raster data
            features: the vector data
        """
        tile_data: dict = {}
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

        path.mkdir(parents=True, exist_ok=True)
        for (col, row), geojson_features in tile_data.items():
            fc = {
                "type": "FeatureCollection",
                "features": [geojson_feat for geojson_feat in geojson_features],
                "properties": projection.serialize(),
            }
            cur_fname = path / f"{col}_{row}.geojson"
            logger.debug("writing tile (%d, %d) to %s", col, row, cur_fname)
            with cur_fname.open("w") as f:
                json.dump(fc, f)

    def decode_vector(self, path: UPath, bounds: PixelBounds) -> list[Feature]:
        """Decodes vector data.

        Args:
            path: the directory to read from
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
                cur_fname = path / f"{col}_{row}.geojson"
                if not cur_fname.exists():
                    continue
                with cur_fname.open("r") as f:
                    fc = json.load(f)
                if "properties" in fc and "crs" in fc["properties"]:
                    projection = Projection.deserialize(fc["properties"])
                else:
                    projection = WGS84_PROJECTION

                for feat in fc["features"]:
                    shp = shapely.geometry.shape(feat["geometry"])
                    shp = shapely.transform(
                        shp,
                        lambda array: array
                        + np.array([[col * self.tile_size, row * self.tile_size]]),
                    )
                    feat["geometry"] = json.loads(shapely.to_geojson(shp))
                    features.append(Feature.from_geojson(projection, feat))
        return features

    @staticmethod
    def from_config(name: str, config: dict[str, Any]) -> "TileVectorFormat":
        """Create a TileVectorFormat from a config dict.

        Args:
            name: the name of this format
            config: the config dict

        Returns:
            the TileVectorFormat
        """
        return TileVectorFormat(tile_size=config.get("tile_size", 512))


class GeojsonCoordinateMode(Enum):
    """The projection to use when writing GeoJSON file."""

    # Write the features as is.
    PIXEL = "pixel"

    # Write the features in CRS coordinates (i.e., a projection with x_resolution=1 and
    # y_resolution=1).
    CRS = "crs"

    # Write in WGS84 (longitude, latitude) coordinates.
    WGS84 = "wgs84"


@VectorFormats.register("geojson")
class GeojsonVectorFormat(VectorFormat):
    """A vector format that uses one big GeoJSON."""

    fname = "data.geojson"

    def __init__(
        self, coordinate_mode: GeojsonCoordinateMode = GeojsonCoordinateMode.PIXEL
    ):
        """Create a new GeojsonVectorFormat.

        Args:
            coordinate_mode: the projection to use for coordinates written to the
                GeoJSON files. PIXEL means we write them as is, CRS means we just undo
                the resolution in the Projection so they are in CRS coordinates, and
                WGS84 means we always write longitude/latitude. When using PIXEL, the
                GeoJSON will not be readable by GIS tools since it relies on a custom
                encoding.
        """
        self.coordinate_mode = coordinate_mode

    def encode_to_file(
        self, fname: UPath, projection: Projection, features: list[Feature]
    ) -> None:
        """Encode vector data to a specific file.

        Args:
            fname: the file to write to
            projection: the projection of the raster data
            features: the vector data
        """
        fc: dict[str, Any] = {"type": "FeatureCollection"}

        # Identify target projection and convert features.
        # Also set the target projection in the FeatureCollection.
        # For PIXEL mode, we need to use a custom encoding so the resolution is stored.
        output_projection: Projection
        if self.coordinate_mode == GeojsonCoordinateMode.PIXEL:
            output_projection = projection
            fc["properties"] = projection.serialize()
        else:
            if self.coordinate_mode == GeojsonCoordinateMode.CRS:
                output_projection = Projection(projection.crs, 1, 1)
            elif self.coordinate_mode == GeojsonCoordinateMode.WGS84:
                output_projection = WGS84_PROJECTION

            fc["crs"] = {
                "type": "name",
                "properties": {
                    "name": output_projection.crs.to_wkt(),
                },
            }

        fc["features"] = []
        for feat in features:
            feat = feat.to_projection(output_projection)
            fc["features"].append(feat.to_geojson())

        logger.debug(
            "writing features to %s with coordinate mode %s",
            fname,
            self.coordinate_mode,
        )
        with fname.open("w") as f:
            json.dump(fc, f)

    def encode_vector(
        self, path: UPath, projection: Projection, features: list[Feature]
    ) -> None:
        """Encodes vector data.

        Args:
            path: the directory to write to
            projection: the projection of the raster data
            features: the vector data
        """
        path.mkdir(parents=True, exist_ok=True)
        self.encode_to_file(path / self.fname, projection, features)

    def decode_from_file(self, fname: UPath) -> list[Feature]:
        """Decodes vector data from a filename.

        Args:
            fname: the filename to read.

        Returns:
            the vector data
        """
        with fname.open("r") as f:
            fc = json.load(f)

        # Detect the projection that the features are stored under.
        if "properties" in fc and "crs" in fc["properties"]:
            # Means it uses our custom Projection encoding.
            projection = Projection.deserialize(fc["properties"])
        elif "crs" in fc:
            # Means it uses standard GeoJSON CRS encoding.
            crs = CRS.from_string(fc["crs"]["properties"]["name"])
            projection = Projection(crs, 1, 1)
        else:
            # Otherwise it should be WGS84 (GeoJSONs created in rslearn should include
            # the "crs" attribute, but maybe it was created externally).
            projection = WGS84_PROJECTION

        return [Feature.from_geojson(projection, feat) for feat in fc["features"]]

    def decode_vector(self, path: UPath, bounds: PixelBounds) -> list[Feature]:
        """Decodes vector data.

        Args:
            path: the directory to read from
            bounds: the bounds of the vector data to read

        Returns:
            the vector data
        """
        return self.decode_from_file(path / self.fname)

    @staticmethod
    def from_config(name: str, config: dict[str, Any]) -> "GeojsonVectorFormat":
        """Create a GeojsonVectorFormat from a config dict.

        Args:
            name: the name of this format
            config: the config dict

        Returns:
            the GeojsonVectorFormat
        """
        kwargs = {}
        if "coordinate_mode" in config:
            kwargs["coordinate_mode"] = GeojsonCoordinateMode(config["coordinate_mode"])
        return GeojsonVectorFormat(**kwargs)


def load_vector_format(config: VectorFormatConfig) -> VectorFormat:
    """Loads a VectorFormat from a VectorFormatConfig.

    Args:
        config: the VectorFormatConfig configuration object specifying the
            VectorFormat.

    Returns:
        the loaded VectorFormat implementation
    """
    cls = VectorFormats.get_class(config.name)
    return cls.from_config(config.name, config.config_dict)
