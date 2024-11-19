"""Spatiotemporal geometry utilities."""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import numpy.typing as npt
import rasterio.warp
import shapely
import shapely.wkt
from rasterio.crs import CRS

from rslearn.log_utils import get_logger

logger = get_logger(__name__)

RESOLUTION_EPSILON = 1e-6


PixelBounds = tuple[int, int, int, int]


def is_same_resolution(res1: float, res2: float) -> bool:
    """Returns whether the two resolutions are the same."""
    return (max(res1, res2) / min(res1, res2) - 1) < RESOLUTION_EPSILON


def shp_intersects(shp1: shapely.Geometry, shp2: shapely.Geometry) -> bool:
    """Returns whether the two shapes intersect.

    Tries shp.intersects but falls back to shp.intersection which can be more
    reliable.
    """
    try:
        return shp1.intersects(shp2)
    except shapely.GEOSException:
        return shp1.intersection(shp2).area > 0


class Projection:
    """A projection specifies a CRS, x resolution, and y resolution.

    The coordinate reference system (CRS) defines the meaning of the coordinates. The
    resolutions specify the pixels per projection unit, and are used to map pixel
    coordinates to CRS coordinates.
    """

    def __init__(self, crs: CRS, x_resolution: float, y_resolution: float) -> None:
        """Initialize a new Projection.

        Args:
            crs: the CRS
            x_resolution: the x resolution
            y_resolution: the y resolution
        """
        self.crs = crs
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution

    def __eq__(self, other: Any) -> bool:
        """Returns whether this projection is the same as the other projection."""
        if not isinstance(other, Projection):
            return False
        if self.crs != other.crs:
            return False
        if not is_same_resolution(self.x_resolution, other.x_resolution):
            return False
        if not is_same_resolution(self.y_resolution, other.y_resolution):
            return False
        return True

    def __repr__(self) -> str:
        """Returns a string representation of this projection."""
        return (
            f"Projection(crs={self.crs}, "
            + f"x_resolution={self.x_resolution}, "
            + f"y_resolution={self.y_resolution})"
        )

    def __str__(self) -> str:
        """Returns a human-readable string summary of this projection."""
        return f"{self.crs}_{self.x_resolution}_{self.y_resolution}"

    def __hash__(self) -> int:
        """Returns a hash of this projection."""
        return hash((self.crs, self.x_resolution, self.y_resolution))

    def serialize(self) -> dict:
        """Serializes the projection to a JSON-encodable dictionary."""
        return {
            "crs": self.crs.to_string(),
            "x_resolution": self.x_resolution,
            "y_resolution": self.y_resolution,
        }

    @staticmethod
    def deserialize(d: dict) -> "Projection":
        """Deserializes a projection from a JSON-decoded dictionary."""
        return Projection(
            crs=CRS.from_string(d["crs"]),
            x_resolution=d["x_resolution"],
            y_resolution=d["y_resolution"],
        )


class STGeometry:
    """A spatiotemporal geometry.

    Specifiec crs and resolution and corresponding shape in pixel coordinates. Also
    specifies an optional time range (time range is unlimited if unset).
    """

    def __init__(
        self,
        projection: Projection,
        shp: shapely.Geometry,
        time_range: tuple[datetime, datetime] | None,
    ):
        """Creates a new spatiotemporal geometry.

        Args:
            projection: the projection
            shp: the shape in pixel coordinates
            time_range: optional start and end time (default unlimited)
        """
        self.projection = projection
        self.shp = shp
        self.time_range = time_range

    def contains_time(self, time: datetime) -> bool:
        """Returns whether this box contains the time."""
        if self.time_range is None:
            return True
        return time >= self.time_range[0] and time < self.time_range[1]

    def distance_to_time(self, time: datetime) -> timedelta:
        """Returns the distance from this box to the specified time.

        Args:
            time: the time to compute distance from

        Returns:
            the distance, which is 0 if the box contains the time
        """
        if self.time_range is None:
            return timedelta()
        if time < self.time_range[0]:
            return self.time_range[0] - time
        if time > self.time_range[1]:
            return time - self.time_range[1]
        return timedelta()

    def distance_to_time_range(
        self, time_range: tuple[datetime, datetime] | None
    ) -> timedelta:
        """Returns the distance from this geometry to the specified time range.

        Args:
            time_range: the time range to compute distance from

        Returns:
            the distance, which is 0 if the time ranges intersect
        """
        if self.time_range is None or time_range is None:
            return timedelta()
        if time_range[1] < self.time_range[0]:
            return self.time_range[0] - time_range[1]
        if self.time_range[1] < time_range[0]:
            return time_range[0] - self.time_range[1]
        return timedelta()

    def intersects_time_range(
        self, time_range: tuple[datetime, datetime] | None
    ) -> bool:
        """Returns whether this geometry intersects the other time range."""
        if self.time_range is None or time_range is None:
            return True
        if self.time_range[1] <= time_range[0]:
            return False
        if time_range[1] <= self.time_range[0]:
            return False
        return True

    def intersects(self, other: "STGeometry") -> bool:
        """Returns whether this box intersects the other box."""
        # Check temporal.
        if not self.intersects_time_range(other.time_range):
            return False

        # Check spatial.
        # Need to reproject if projections don't match.
        if other.projection != self.projection:
            other = other.to_projection(self.projection)
        if not self.shp.intersects(other.shp):
            return False

        return True

    def to_projection(self, projection: Projection) -> "STGeometry":
        """Transforms this geometry to the specified projection."""

        def apply_resolution(
            array: np.ndarray,
            x_resolution: float,
            y_resolution: float,
            forward: bool = True,
        ) -> np.ndarray:
            if forward:
                return np.stack(
                    [array[:, 0] / x_resolution, array[:, 1] / y_resolution], axis=1
                )
            else:
                return np.stack(
                    [array[:, 0] * x_resolution, array[:, 1] * y_resolution], axis=1
                )

        # Undo resolution.
        shp = shapely.transform(
            self.shp,
            lambda array: apply_resolution(
                array,
                self.projection.x_resolution,
                self.projection.y_resolution,
                forward=False,
            ),
        )
        # Change crs.
        shp = rasterio.warp.transform_geom(self.projection.crs, projection.crs, shp)
        shp = shapely.geometry.shape(shp)
        # Apply new resolution.
        shp = shapely.transform(
            shp,
            lambda array: apply_resolution(
                array, projection.x_resolution, projection.y_resolution, forward=True
            ),
        )
        return STGeometry(projection, shp, self.time_range)

    def __repr__(self) -> str:
        """Returns a string representation of this STGeometry."""
        return (
            f"STGeometry(projection={self.projection}, shp={self.shp}, "
            + f"time_range={self.time_range})"
        )

    def serialize(self) -> dict:
        """Serializes the geometry to a JSON-encodable dictionary."""
        return {
            "projection": self.projection.serialize(),
            "shp": self.shp.wkt,
            "time_range": (
                [self.time_range[0].isoformat(), self.time_range[1].isoformat()]
                if self.time_range
                else None
            ),
        }

    @staticmethod
    def deserialize(d: dict) -> "STGeometry":
        """Deserializes a geometry from a JSON-decoded dictionary."""
        return STGeometry(
            projection=Projection.deserialize(d["projection"]),
            shp=shapely.wkt.loads(d["shp"]),
            time_range=(
                (
                    datetime.fromisoformat(d["time_range"][0]),
                    datetime.fromisoformat(d["time_range"][1]),
                )
                if d["time_range"]
                else None
            ),
        )


def flatten_shape(shp: shapely.Geometry) -> list[shapely.Geometry]:
    """Flatten the shape into a list of primitive shapes (Point, LineString, and Polygon).

    Args:
        shp: the shape, which could be a primitive shape like polygon or a collection.

    Returns:
        list of primitive shapes.
    """
    if isinstance(
        shp,
        shapely.MultiPoint
        | shapely.MultiLineString
        | shapely.MultiPolygon
        | shapely.GeometryCollection,
    ):
        flat_list: list[shapely.Geometry] = []
        for component in shp.geoms:
            flat_list.extend(flatten_shape(component))
        return flat_list

    else:
        return [shp]


def _collect_shapes(shapes: list[shapely.Geometry]) -> shapely.Geometry:
    # Collect the shapes into an appropriate container.
    flat_list: list[shapely.Geometry] = []
    for shp in shapes:
        flat_list.extend(flatten_shape(shp))

    if len(flat_list) == 1:
        return flat_list[0]

    if all(isinstance(shp, shapely.Point) for shp in flat_list):
        return shapely.MultiPoint(flat_list)

    if all(isinstance(shp, shapely.LineString) for shp in flat_list):
        return shapely.MultiLineString(flat_list)

    if all(isinstance(shp, shapely.Polygon) for shp in flat_list):
        return shapely.MultiPolygon(flat_list)

    return shapely.GeometryCollection(flat_list)


def split_shape_at_prime_meridian(
    shp: shapely.Geometry, epsilon: float = 1e-6
) -> shapely.Geometry:
    """Split the given shape at the prime meridian.

    The shape must be in WGS84 coordinates.

    See split_at_prime_meridian for details.

    Args:
        shp: the shape to split.
        epsilon: the padding in degrees.

    Returns:
        the split shape, in WGS84 projection.
    """
    # We assume the shape is fine if:
    # 1. It doesn't need padding (no coordinates close to +/- 180).
    # 2. And all coordinates are either less than 90 or more than -90 (meaning the
    #    shape approaches the prime meridian on at most one side).
    bounds = shp.bounds
    if bounds[0] > -180 + epsilon and bounds[2] < 90:
        return shp
    if bounds[0] > -90 and bounds[2] < 180 - epsilon:
        return shp

    if isinstance(
        shp,
        shapely.MultiPoint
        | shapely.MultiLineString
        | shapely.MultiPolygon
        | shapely.GeometryCollection,
    ):
        return _collect_shapes(
            [split_shape_at_prime_meridian(component) for component in shp]
        )

    if isinstance(shp, shapely.Point):
        # Points only need padding.
        lon = shp.x
        if lon < -180 + epsilon:
            lon = -180 + epsilon
        if lon > 180 - epsilon:
            lon = 180 - epsilon
        return shapely.Point(lon, shp.y)

    if isinstance(shp, shapely.LineString | shapely.Polygon):
        # We add 360 to the negative coordinates and then separate the parts above and
        # below 180.
        def add360(array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            new_array = array.copy()
            new_array[new_array[:, 0] < 0, 0] += 360
            return new_array

        shp = shapely.transform(shp, add360)

        positive_part = shapely.box(0, -90, 180 - epsilon, 90)
        negative_part = shapely.box(180 + epsilon, -90, 360, 90)
        positive_shp = shp.intersection(positive_part)
        negative_shp = shp.intersection(negative_part)
        negative_shp = shapely.transform(negative_shp, lambda coords: coords - [360, 0])
        return _collect_shapes([positive_shp, negative_shp])

    raise TypeError("Unsupported shape type")


def split_at_prime_meridian(geometry: STGeometry, epsilon: float = 1e-6) -> STGeometry:
    """Split lines and polygons in the given geometry at the prime meridian.

    The returned geometry will always be in WGS84 projection.

    Small padding is also introduced to ensure coordinates are a bit more than -180 or
    a bit less than 180.

    For example, if the input is a polygon:

        Polygon([[-180, 10], [180, 11], [-179, 11], [-179, 10]])

    Then it would be converted to:

        Polygon([[-179.999999, 10], [-179,999999, 11], [-179, 11], [-179, 10]])

    Args:
        geometry: the geometry to split.
        epsilon: the padding in degrees. It is equivalent to about 1 m at the equator.
            We ensure no longitude coordinates are within this padding of +/- 180.

    Returns:
        the padded geometry, in WGS84 projection.
    """
    # Convert to WGS84.
    geometry = geometry.to_projection(Projection(CRS.from_epsg(4326), 1, 1))
    new_shp = split_shape_at_prime_meridian(geometry.shp, epsilon=epsilon)
    return STGeometry(geometry.projection, new_shp, geometry.time_range)
