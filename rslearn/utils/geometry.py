from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import rasterio.warp
import shapely
import shapely.wkt
from rasterio.crs import CRS

RESOLUTION_EPSILON = 1e-6


def is_same_resolution(res1: float, res2: float) -> bool:
    """Returns whether the two resolutions are the same."""
    return (max(res1, res2) / min(res1, res2) - 1) < RESOLUTION_EPSILON


class Projection:
    def __init__(self, crs: CRS, x_resolution: float, y_resolution: float) -> None:
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
        return (
            f"Projection(crs={self.crs}, "
            + f"x_resolution={self.x_resolution}, "
            + f"y_resolution={self.y_resolution})"
        )

    def __str__(self) -> str:
        return f"{self.crs}_{self.x_resolution}_{self.y_resolution}"

    def __hash__(self) -> int:
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
        time_range: Optional[tuple[datetime, datetime]],
    ):
        """Creates a new spatiotemporal geometry.

        Args:
            crs: the CRS of the coordinate system
            resolution: projection units per pixel
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
        self, time_range: Optional[tuple[datetime, datetime]]
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
        self, time_range: Optional[tuple[datetime, datetime]]
    ) -> timedelta:
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
        if not self.intersects_time_range(other.time_range):
            return False
        if not self.shp.intersects(other.shp):
            return False
        return True

    def to_projection(self, projection: Projection) -> "STGeometry":
        """Transforms this geometry to the specified projection."""

        def apply_resolution(array, x_resolution, y_resolution, forward=True):
            if forward:
                return np.stack(
                    [
                        array[:, 0] / x_resolution,
                        array[:, 1] / y_resolution,
                    ],
                    axis=1,
                )
            else:
                return np.stack(
                    [
                        array[:, 0] * x_resolution,
                        array[:, 1] * y_resolution,
                    ],
                    axis=1,
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
