"""GridIndex spatial index implementation."""

import math
from collections.abc import Callable
from typing import Any

from rslearn.utils.spatial_index import SpatialIndex


class GridIndex(SpatialIndex):
    """An index of temporal geometries using a grid.

    Each cell in the grid contains a list of geometries that intersect it.
    """

    def __init__(self, size: float) -> None:
        """Initialize a new GridIndex.

        Args:
            size: the size of the grid cells
        """
        self.size = size
        self.grid: dict = {}
        self.items: list = []

    def insert(self, box: tuple[float, float, float, float], data: Any) -> None:
        """Insert a box into the index.

        Args:
            box: the bounding box of this item (minx, miny, maxx, maxy)
            data: arbitrary object
        """
        item_idx = len(self.items)
        self.items.append(data)

        def f(cell: tuple[int, int]) -> None:
            if cell not in self.grid:
                self.grid[cell] = []
            self.grid[cell].append(item_idx)

        self._each_cell(box, f)

    def _each_cell(
        self,
        box: tuple[float, float, float, float],
        f: Callable[[tuple[int, int]], None],
    ) -> None:
        """Call f for each cell intersecting a box.

        Args:
            box: the box (minx, miny, maxx, maxy)
            f: function to call for each cell
        """
        for i in range(
            int(math.floor(box[0] / self.size)), int(math.floor(box[2] / self.size)) + 1
        ):
            for j in range(
                int(math.floor(box[1] / self.size)),
                int(math.floor(box[3] / self.size)) + 1,
            ):
                f((i, j))

    def query(self, box: tuple[float, float, float, float]) -> list[Any]:
        """Query the index for objects intersecting a box.

        Args:
            box: the bounding box query (minx, miny, maxx, maxy)

        Returns:
            a list of objects in the index intersecting the box
        """
        matches = set()

        def f(cell: tuple[int, int]) -> None:
            if cell not in self.grid:
                return
            for item_idx in self.grid[cell]:
                matches.add(item_idx)

        self._each_cell(box, f)
        return [self.items[item_idx] for item_idx in matches]
