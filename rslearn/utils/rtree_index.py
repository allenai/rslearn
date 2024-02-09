from typing import Any, Optional

from rtree import index

from rslearn.utils.spatial_index import SpatialIndex


class RtreeIndex(SpatialIndex):
    """An index of temporal geometries using a grid.

    Each cell in the grid contains a list of geometries that intersect it.
    """

    def __init__(self, fname: Optional[str] = None):
        """Initialize a new RtreeIndex.

        If fname is set, the index is persisted on disk, otherwise it is in-memory.

        Args:
            fname: the filename to store the index in, or None to create an in-memory
                index
        """
        self.index = index.Index(fname)
        self.counter = 0

    def insert(self, box: tuple[float, float, float, float], data: Any) -> None:
        """Insert a box into the index.

        Args:
            box: the bounding box of this item (minx, miny, maxx, maxy)
            data: arbitrary object
        """
        self.counter += 1
        self.index.insert(id=self.counter, coordinates=box, obj=data)

    def query(self, box: tuple[float, float, float, float]) -> list[Any]:
        """Query the index for objects intersecting a box.

        Args:
            box: the bounding box query (minx, miny, maxx, maxy)

        Returns:
            a list of objects in the index intersecting the box
        """
        results = self.index.intersection(box, objects=True)
        return [r.object for r in results]
