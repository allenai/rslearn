"""RtreeIndex spatial index implementation."""

import os
from typing import Any

from rtree import index

from rslearn.utils.spatial_index import SpatialIndex


class RtreeIndex(SpatialIndex):
    """An index of spatiotemporal geometries backed by an rtree index.

    Both in-memory and on-disk options are supported.
    """

    def __init__(self, fname: str | None = None):
        """Initialize a new RtreeIndex.

        If fname is set, the index is persisted on disk, otherwise it is in-memory.

        For on-disk index, optionally use is_done and mark_done to reuse an existing
        index at the same filename. For reuse, if mark_done was called previously, then
        the existing index will be reused. Use is_done to check if mark_done was
        previously called, and use mark_done to indicate the index is done writing (a
        .done marker file will be created).

        Otherwise, if mark_done was never called previously, then the index is always
        overwritten.

        Args:
            fname: the filename to store the index in, or None to create an in-memory
                index
        """
        self.fname = fname
        kwargs = {}
        if self.fname and not self.is_done():
            kwargs["properties"] = index.Property(overwrite=True)
        self.index = index.Index(fname, **kwargs)
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

    def is_done(self) -> bool:
        """Returns whether mark_done was previously called for the on-disk index."""
        assert self.fname is not None
        return os.path.exists(self.fname + ".done")

    def mark_done(self) -> None:
        """Marks the on-disk index as done."""
        assert self.fname is not None
        with open(os.path.join(self.fname + ".done"), "w"):
            pass
