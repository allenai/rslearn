"""Contains a SpatialIndex implementation that uses an sqlite database.

# TODO: This is not yet complete decide to either complete it or remove this file.
"""

# Ignoring Mypy until we determine if we want to keep this file.
# mypy: ignore-errors

import json
import sqlite3
from typing import Any

from rslearn.utils.geometry import STGeometry
from rslearn.utils.spatial_index import SpatialIndex


class SqliteIndex(SpatialIndex):
    """An index of spatiotemporal geometries backed by an sqlite database.

    We do not use geospatial extensions. Actually, it looks like this is only partially
    implemented so we should either complete it or remove this file.
    """

    def __init__(self, fname: str):
        """Initialize a new SqliteIndex.

        The index is persisted on disk, and any pre-existing objects in the index will
        be restored.

        Args:
            fname: the filename to store the sqlite3 database
        """
        self.con = sqlite3.connect(fname)
        self.cur = self.con.cursor()
        self.cur.execute(
            "CREATE TABLE IF NOT EXISTS items "
            + "(id INTEGER PRIMARY KEY, x1 REAL, y1 REAL, x2 REAL, y2 REAL, "
            + "geometry TEXT, data TEXT)"
        )
        self.cur.execute("CREATE INDEX IF NOT EXISTS ")
        self.con.commit()

    def insert(self, geometry: STGeometry, data: Any) -> None:
        """Insert a geometry into the index.

        Args:
            geometry: the :class:`STGeometry` specifying the extent of the object
            data: arbitrary JSON-encodable object
        """
        bounds = geometry.shp.bounds
        self.cur.execute(
            "INSERT INTO items (x1, y1, x2, y2, geometry, data) "
            + "VALUES (?, ?, ?, ?, ?, ?)",
            (
                bounds[0],
                bounds[1],
                bounds[2],
                bounds[3],
                json.dumps(geometry.serialize()),
                json.dumps(data),
            ),
        )
        self.con.commit()

    def query(self, geometry: STGeometry) -> list[Any]:
        """Query the index for objects intersecting a geometry.

        Args:
            geometry: the :class:`STGeometry query

        Returns:
            a list of objects in the index intersecting the geometry
        """
        result = self.cur.execute(
            "SELECT geometry, data FROM items "
            + "WHERE x2 > ? AND y2 > ? AND x1 < ? AND y1 < ?",
            geometry.shp.bounds,
        )
        items = []
        while True:
            db_row = result.fetchone()
            if db_row is None:
                break

            cur_geometry = STGeometry.deserialize(json.loads(db_row[0]))
            if not geometry.intersects(cur_geometry):
                continue
            items.append(json.loads(db_row[1]))
        return items

    def close(self):
        """Closes the index."""
        self.con.close()
