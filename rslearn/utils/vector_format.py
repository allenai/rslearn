"""Classes for writing vector data to a UPath."""

import json
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

import shapely
from rasterio.crs import CRS
from upath import UPath

from rslearn.config import VectorFormatConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.log_utils import get_logger
from rslearn.utils.fsspec import open_atomic

from .feature import Feature
from .geometry import PixelBounds, Projection, STGeometry, safely_reproject_and_clip

logger = get_logger(__name__)
_VectorFormatT = TypeVar("_VectorFormatT", bound="VectorFormat")


class _VectorFormatRegistry(dict[str, type["VectorFormat"]]):
    """Registry for VectorFormat classes."""

    def register(
        self, name: str
    ) -> Callable[[type[_VectorFormatT]], type[_VectorFormatT]]:
        """Decorator to register a vector format class."""

        def decorator(cls: type[_VectorFormatT]) -> type[_VectorFormatT]:
            self[name] = cls
            return cls

        return decorator


VectorFormats = _VectorFormatRegistry()


class VectorFormat:
    """An abstract class for writing vector data.

    Implementations of VectorFormat should support reading and writing vector data in
    a UPath. Vector data is a list of GeoJSON-like features.
    """

    def encode_vector(self, path: UPath, features: list[Feature]) -> None:
        """Encodes vector data.

        Args:
            path: the directory to write to
            features: the vector data
        """
        raise NotImplementedError

    def decode_vector(
        self, path: UPath, projection: Projection, bounds: PixelBounds
    ) -> list[Feature]:
        """Decodes vector data.

        Args:
            path: the directory to read from
            projection: the projection to read the data in
            bounds: the bounds to read under the given projection. Only features that
                intersect the bounds should be returned.

        Returns:
            the vector data
        """
        raise NotImplementedError

    @staticmethod
    def from_config(name: str, config: dict[str, Any]) -> "VectorFormat":
        """Create a VectorFormat from a config dict.

        Args:
            name: the name of this format
            config: the config dict

        Returns:
            the VectorFormat instance
        """
        raise NotImplementedError


@VectorFormats.register("tile")
class TileVectorFormat(VectorFormat):
    """TileVectorFormat stores data in GeoJSON files corresponding to grid cells.

    A tile size defines the grid size in pixels. One file is created for each grid cell
    containing at least one feature. Features are written to all grid cells that they
    intersect.
    """

    def __init__(
        self,
        tile_size: int = 512,
        projection: Projection | None = None,
        index_property_name: str = "tvf_index",
    ):
        """Initialize a new TileVectorFormat instance.

        Args:
            tile_size: the tile size (grid size in pixels), default 512
            projection: if set, store features under this projection. Otherwise, the
                output projection is taken from the first feature in an encode_vector
                call.
            index_property_name: property name used to store an index integer that
                identifies the same feature across different tiles.
        """
        self.tile_size = tile_size
        self.projection = projection
        self.index_property_name = index_property_name

    def encode_vector(self, path: UPath, features: list[Feature]) -> None:
        """Encodes vector data.

        Args:
            path: the directory to write to
            features: the vector data
        """
        # Determine the output projection to write in.
        if len(features) == 0:
            # We won't actually write any features but still setting output_projection
            # to write to projection.json.
            # We just fallback to WGS84 here.
            output_projection = WGS84_PROJECTION
        elif self.projection is not None:
            output_projection = self.projection
        else:
            output_projection = features[0].geometry.projection

        # Save metadata file containing the serialized projection so we can load it
        # when decoding.
        with open_atomic(path / "projection.json", "w") as f:
            json.dump(output_projection.serialize(), f)

        # Dictionary from tile (col, row) to the list of features intersecting that
        # tile. We iterate over the features to populate tile_data, then write each
        # tile as a separate file.
        tile_data: dict[tuple[int, int], list[dict]] = {}

        for feat_idx, feat in enumerate(features):
            # Skip invalid features since they can cause errors.
            if not feat.geometry.shp.is_valid:
                continue

            # Identify each grid cell that this feature intersects.
            geometry = feat.geometry.to_projection(output_projection)
            bounds = geometry.shp.bounds
            start_tile = (
                int(bounds[0]) // self.tile_size,
                int(bounds[1]) // self.tile_size,
            )
            end_tile = (
                int(bounds[2]) // self.tile_size + 1,
                int(bounds[3]) // self.tile_size + 1,
            )

            # We add an index property to the features so when reading we can
            # de-duplicate (in case we read multiple tiles that contain the same
            # feature).
            properties = {self.index_property_name: feat_idx}
            properties.update(feat.properties)
            # Use the re-projected geometry here.
            output_feat = Feature(geometry, properties)
            output_geojson = output_feat.to_geojson()

            # Now we add the feature to each tile that it intersects.
            for col in range(start_tile[0], end_tile[0]):
                for row in range(start_tile[1], end_tile[1]):
                    tile_box = shapely.box(
                        col * self.tile_size,
                        row * self.tile_size,
                        (col + 1) * self.tile_size,
                        (row + 1) * self.tile_size,
                    )
                    if not geometry.shp.intersects(tile_box):
                        continue
                    tile = (col, row)
                    if tile not in tile_data:
                        tile_data[tile] = []
                    tile_data[tile].append(output_geojson)

        path.mkdir(parents=True, exist_ok=True)

        # Now save each tile.
        for (col, row), geojson_features in tile_data.items():
            fc = {
                "type": "FeatureCollection",
                "features": [geojson_feat for geojson_feat in geojson_features],
                "properties": output_projection.serialize(),
            }
            cur_fname = path / f"{col}_{row}.geojson"
            logger.debug("writing tile (%d, %d) to %s", col, row, cur_fname)
            with open_atomic(cur_fname, "w") as f:
                json.dump(fc, f)

    def decode_vector(
        self, path: UPath, projection: Projection, bounds: PixelBounds
    ) -> list[Feature]:
        """Decodes vector data.

        Args:
            path: the directory to read from
            projection: the projection to read the data in
            bounds: the bounds to read under the given projection. Only features that
                intersect the bounds should be returned.

        Returns:
            the vector data
        """
        # Convert the bounds to the projection of the stored data.
        with (path / "projection.json").open() as f:
            storage_projection = Projection.deserialize(json.load(f))
        bounds_geom = STGeometry(projection, shapely.box(*bounds), None)
        storage_bounds = bounds_geom.to_projection(storage_projection).shp.bounds

        start_tile = (
            int(storage_bounds[0]) // self.tile_size,
            int(storage_bounds[1]) // self.tile_size,
        )
        end_tile = (
            (int(storage_bounds[2]) - 1) // self.tile_size + 1,
            (int(storage_bounds[3]) - 1) // self.tile_size + 1,
        )
        features = []
        for col in range(start_tile[0], end_tile[0]):
            for row in range(start_tile[1], end_tile[1]):
                cur_fname = path / f"{col}_{row}.geojson"
                if not cur_fname.exists():
                    continue
                with cur_fname.open() as f:
                    fc = json.load(f)

                for geojson_feat in fc["features"]:
                    feat = Feature.from_geojson(storage_projection, geojson_feat)
                    features.append(feat.to_projection(projection))
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
        kwargs = {}
        if "tile_size" in config:
            kwargs["tile_size"] = config["tile_size"]
        if "projection" in config:
            kwargs["projection"] = Projection.deserialize(config["projection"])
        if "index_property_name" in config:
            kwargs["index_property_name"] = config["index_property_name"]
        return TileVectorFormat(**kwargs)


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

    def encode_to_file(self, fname: UPath, features: list[Feature]) -> None:
        """Encode vector data to a specific file.

        Args:
            fname: the file to write to
            features: the vector data
        """
        fc: dict[str, Any] = {"type": "FeatureCollection"}

        # Identify target projection and convert features.
        # Also set the target projection in the FeatureCollection.
        # For PIXEL mode, we need to use a custom encoding so the resolution is stored.
        output_projection: Projection
        if len(features) > 0 and self.coordinate_mode != GeojsonCoordinateMode.WGS84:
            if self.coordinate_mode == GeojsonCoordinateMode.PIXEL:
                output_projection = features[0].geometry.projection
                fc["properties"] = output_projection.serialize()
            elif self.coordinate_mode == GeojsonCoordinateMode.CRS:
                output_projection = Projection(
                    features[0].geometry.projection.crs, 1, 1
                )
                fc["crs"] = {
                    "type": "name",
                    "properties": {
                        "name": output_projection.crs.to_wkt(),
                    },
                }
        else:
            # Either there are no features so we need to fallback to WGS84, or the
            # coordinate mode is WGS84.
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
        with open_atomic(fname, "w") as f:
            json.dump(fc, f)

    def encode_vector(self, path: UPath, features: list[Feature]) -> None:
        """Encodes vector data.

        Args:
            path: the directory to write to
            features: the vector data
        """
        path.mkdir(parents=True, exist_ok=True)
        self.encode_to_file(path / self.fname, features)

    def decode_from_file(self, fname: UPath) -> list[Feature]:
        """Decodes vector data from a filename.

        Args:
            fname: the filename to read.

        Returns:
            the vector data
        """
        with fname.open() as f:
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

    def decode_vector(
        self, path: UPath, projection: Projection, bounds: PixelBounds
    ) -> list[Feature]:
        """Decodes vector data.

        Args:
            path: the directory to read from
            projection: the projection to read the data in
            bounds: the bounds to read under the given projection. Only features that
                intersect the bounds should be returned.

        Returns:
            the vector data
        """
        features = self.decode_from_file(path / self.fname)

        # Re-project to the desired projection and clip to bounds.
        dst_geom = STGeometry(projection, shapely.box(*bounds), None)
        reprojected_geoms = safely_reproject_and_clip(
            [feat.geometry for feat in features], dst_geom
        )
        reprojected_features = []
        for feat, geom in zip(features, reprojected_geoms):
            if geom is None:
                # None value means that it did not intersect the provided bounds.
                continue
            reprojected_features.append(Feature(geom, feat.properties))

        return reprojected_features

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
    cls = VectorFormats[config.name]
    return cls.from_config(config.name, config.config_dict)
