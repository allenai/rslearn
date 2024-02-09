from datetime import datetime, timedelta
from typing import Optional

import rasterio.warp
import shapely
import shapely.wkt
from rasterio.crs import CRS


class STGeometry:
    """A spatiotemporal geometry.

    Specifiec crs and resolution and corresponding shape in pixel coordinates. Also
    specifies an optional time range (time range is unlimited if unset).
    """

    def __init__(
        self,
        crs: CRS,
        resolution: float,
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
        self.crs = crs
        self.resolution = resolution
        self.shp = shp
        self.time_range = time_range

    def contains_time(self, time: datetime) -> bool:
        """Returns whether this box contains the time."""
        if self.time_range is None:
            return True
        return time >= self.time_range[0] and time < self.time_range[1]

    def time_distance(self, time: datetime) -> timedelta:
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

    def intersects(self, other: "STGeometry") -> bool:
        """Returns whether this box intersects the other box."""
        if self.time_range and other.time_range:
            if (
                self.time_range[1] <= other.time_range[0]
                or self.time_range[0] >= other.time_range[1]
            ):
                return False
        return self.shp.intersects(other.shp)

    def to_crs(self, crs: CRS, resolution: float) -> "STGeometry":
        """Transforms this geometry to the specified CRS and resolution."""
        # Undo resolution.
        shp = shapely.transform(self.shp, lambda arr: arr * self.resolution)
        # Change crs.
        shp = rasterio.warp.transform_geom(self.crs, crs, shp)
        shp = shapely.geometry.shape(shp)
        # Apply new resolution.
        shp = shapely.transform(shp, lambda arr: arr / resolution)
        return STGeometry(crs, resolution, shp, self.time_range)

    def __repr__(self) -> str:
        return (
            f"STGeometry(crs={self.crs}, resolution={self.resolution}, "
            + f"shp={self.shp}, time_range={self.time_range})"
        )

    def serialize(self) -> dict:
        """Serializes the geometry to a JSON-encodable dictionary."""
        return {
            "crs": self.crs.to_string(),
            "resolution": self.resolution,
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
            crs=CRS.from_string(d["crs"]),
            resolution=d["resolution"],
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
