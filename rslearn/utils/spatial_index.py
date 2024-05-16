"""Abstract SpatialIndex class."""

from typing import Any


class SpatialIndex:
    """An abstract class for spatial indices."""

    def insert(self, box: tuple[float, float, float, float], data: Any) -> None:
        """Insert a box into the index.

        Args:
            box: the bounding box of this item (minx, miny, maxx, maxy)
            data: arbitrary object
        """
        raise NotImplementedError

    def query(self, box: tuple[float, float, float, float]) -> list[Any]:
        """Query the index for objects intersecting a box.

        Args:
            box: the bounding box query (minx, miny, maxx, maxy)

        Returns:
            a list of objects in the index intersecting the box
        """
        raise NotImplementedError
