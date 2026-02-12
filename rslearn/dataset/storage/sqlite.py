"""SQLite-based window storage backend."""

import functools
import json
import os
import random
import sqlite3
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from fsspec.implementations.local import LocalFileSystem
from pyproj import CRS
from typing_extensions import override
from upath import UPath

from rslearn.dataset.window import Window, WindowLayerData
from rslearn.log_utils import get_logger
from rslearn.utils import Projection

from .storage import WindowStorage, WindowStorageFactory

logger = get_logger(__name__)

# Retry parameters for handling concurrent access
MAX_RETRIES = 10
RETRY_DELAY = 0.1  # seconds


def _get_local_path(ds_path: UPath) -> Path:
    """Get the local filesystem path from a UPath.

    Args:
        ds_path: the UPath to convert.

    Returns:
        the local Path.

    Raises:
        ValueError: if the path is not a local filesystem path.
    """
    if not isinstance(ds_path.fs, LocalFileSystem):
        raise ValueError(
            f"SQLiteWindowStorage only supports local filesystem paths, "
            f"got filesystem: {type(ds_path.fs).__name__}"
        )
    return Path(ds_path.path)


def _retry_on_locked(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that retries a method on database locked/busy errors."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() or "busy" in str(e).lower():
                    sleep_time = RETRY_DELAY * (2**attempt) * random.uniform(0.5, 1.0)
                    logger.debug(
                        f"Database locked, retrying in {sleep_time:.3f}s "
                        f"(attempt {attempt + 1}/{MAX_RETRIES + 1})"
                    )
                    time.sleep(sleep_time)
                    continue
                raise

        # Call without try/except for final retry.
        return func(*args, **kwargs)

    return wrapper


class SQLiteWindowStorage(WindowStorage):
    """SQLite-backed window storage.

    This storage backend uses a single SQLite database file to store all window
    metadata, layer datas, and completed layer markers.
    """

    DB_FILENAME = "windows.sqlite3"

    def __init__(self, path: UPath):
        """Create a new SQLiteWindowStorage.

        Args:
            path: the path to the dataset (must be a local filesystem path).

        Raises:
            ValueError: if the path is not a local filesystem path.
        """
        self.path = path
        self.db_path = _get_local_path(path) / self.DB_FILENAME
        self._conn: sqlite3.Connection | None = None
        self._conn_pid: int | None = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a cached database connection, creating one if needed.

        The connection is cached per instance and reused across calls.
        After a process fork (e.g. PyTorch DataLoader workers), a new connection
        is created since the parent's connection can't be safely reused.
        """
        pid = os.getpid()
        if self._conn is None or self._conn_pid != pid:
            conn = sqlite3.connect(
                str(self.db_path),
                # Disable implicit transactions, so statements autocommit by default
                # and we explicitly start transactions with BEGIN where needed.
                isolation_level=None,
            )
            # Return sqlite3.Rows instead of tuples.
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys=ON")
            self._conn = conn
            self._conn_pid = pid
        return self._conn

    @_retry_on_locked
    def _init_db(self) -> None:
        """Initialize the database schema if it doesn't exist."""
        conn = self._get_connection()
        # Windows table stores window metadata
        conn.execute("""
            CREATE TABLE IF NOT EXISTS windows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_name TEXT NOT NULL,
                name TEXT NOT NULL,
                crs TEXT NOT NULL,
                x_resolution REAL NOT NULL,
                y_resolution REAL NOT NULL,
                bounds_x1 INTEGER NOT NULL,
                bounds_y1 INTEGER NOT NULL,
                bounds_x2 INTEGER NOT NULL,
                bounds_y2 INTEGER NOT NULL,
                time_start TEXT,
                time_end TEXT,
                options_json TEXT NOT NULL,
                UNIQUE(group_name, name)
            )
        """)
        # Index for faster lookups
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_windows_group
            ON windows(group_name)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_windows_name
            ON windows(name)
        """)

        # Layer datas table stores items.json content per window
        conn.execute("""
            CREATE TABLE IF NOT EXISTS layer_datas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_name TEXT NOT NULL,
                window_name TEXT NOT NULL,
                layer_name TEXT NOT NULL,
                data_json TEXT NOT NULL,
                UNIQUE(group_name, window_name, layer_name)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_layer_datas_window
            ON layer_datas(group_name, window_name)
        """)

        # Completed layers table tracks which layers are completed
        conn.execute("""
            CREATE TABLE IF NOT EXISTS completed_layers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_name TEXT NOT NULL,
                window_name TEXT NOT NULL,
                layer_name TEXT NOT NULL,
                group_idx INTEGER NOT NULL DEFAULT 0,
                UNIQUE(group_name, window_name, layer_name, group_idx)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_completed_layers_window
            ON completed_layers(group_name, window_name)
        """)

    @override
    def get_window_root(self, group: str, name: str) -> UPath:
        """Get the path where the window should be stored."""
        return Window.get_window_root(self.path, group, name)

    @_retry_on_locked
    @override
    def get_windows(
        self,
        groups: list[str] | None = None,
        names: list[str] | None = None,
    ) -> list[Window]:
        """Load the windows in the dataset.

        Args:
            groups: an optional list of groups to filter loading
            names: an optional list of window names to filter loading
        """
        conn = self._get_connection()
        query = """
            SELECT group_name, name, crs, x_resolution, y_resolution,
                   bounds_x1, bounds_y1, bounds_x2, bounds_y2,
                   time_start, time_end, options_json
            FROM windows
        """
        conditions: list[str] = []
        params: list[str] = []

        if groups:
            placeholders = ",".join("?" * len(groups))
            conditions.append(f"group_name IN ({placeholders})")
            params.extend(groups)

        if names:
            placeholders = ",".join("?" * len(names))
            conditions.append(f"name IN ({placeholders})")
            params.extend(names)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor = conn.execute(query, params)
        windows = []
        for row in cursor.fetchall():
            projection = Projection(
                crs=CRS.from_string(row["crs"]),
                x_resolution=row["x_resolution"],
                y_resolution=row["y_resolution"],
            )
            bounds = (
                row["bounds_x1"],
                row["bounds_y1"],
                row["bounds_x2"],
                row["bounds_y2"],
            )
            time_range = None
            if row["time_start"] and row["time_end"]:
                time_range = (
                    datetime.fromisoformat(row["time_start"]),
                    datetime.fromisoformat(row["time_end"]),
                )
            options = json.loads(row["options_json"])

            window = Window(
                storage=self,
                group=row["group_name"],
                name=row["name"],
                projection=projection,
                bounds=bounds,
                time_range=time_range,
                options=options,
            )
            windows.append(window)
        return windows

    @_retry_on_locked
    @override
    def create_or_update_window(self, window: Window) -> None:
        """Create or update the window."""
        time_start = None
        time_end = None
        if window.time_range:
            time_start = window.time_range[0].isoformat()
            time_end = window.time_range[1].isoformat()
        options_json = json.dumps(window.options)

        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO windows (
                    group_name, name, crs, x_resolution, y_resolution,
                    bounds_x1, bounds_y1, bounds_x2, bounds_y2,
                    time_start, time_end, options_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(group_name, name)
                DO UPDATE SET
                    crs = excluded.crs,
                    x_resolution = excluded.x_resolution,
                    y_resolution = excluded.y_resolution,
                    bounds_x1 = excluded.bounds_x1,
                    bounds_y1 = excluded.bounds_y1,
                    bounds_x2 = excluded.bounds_x2,
                    bounds_y2 = excluded.bounds_y2,
                    time_start = excluded.time_start,
                    time_end = excluded.time_end,
                    options_json = excluded.options_json
                """,
                (
                    window.group,
                    window.name,
                    window.projection.crs.to_string(),
                    window.projection.x_resolution,
                    window.projection.y_resolution,
                    window.bounds[0],
                    window.bounds[1],
                    window.bounds[2],
                    window.bounds[3],
                    time_start,
                    time_end,
                    options_json,
                ),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        logger.debug(f"Saved window {window.group}/{window.name} to SQLite")

    @_retry_on_locked
    @override
    def get_layer_datas(self, group: str, name: str) -> dict[str, WindowLayerData]:
        """Get the window layer datas for the specified window."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT layer_name, data_json FROM layer_datas
            WHERE group_name = ? AND window_name = ?
            """,
            (group, name),
        )
        result = {}
        for row in cursor.fetchall():
            data = json.loads(row["data_json"])
            layer_data = WindowLayerData.deserialize(data)
            result[layer_data.layer_name] = layer_data
        return result

    @_retry_on_locked
    @override
    def save_layer_datas(
        self, group: str, name: str, layer_datas: dict[str, WindowLayerData]
    ) -> None:
        """Set the window layer datas for the specified window."""
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            # Delete existing layer datas for this window
            conn.execute(
                "DELETE FROM layer_datas WHERE group_name = ? AND window_name = ?",
                (group, name),
            )
            # Insert new layer datas
            for layer_data in layer_datas.values():
                data_json = json.dumps(layer_data.serialize())
                conn.execute(
                    """
                    INSERT INTO layer_datas (group_name, window_name, layer_name, data_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (group, name, layer_data.layer_name, data_json),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        logger.info(f"Saved layer datas for {group}/{name} to SQLite")

    @_retry_on_locked
    @override
    def list_completed_layers(self, group: str, name: str) -> list[tuple[str, int]]:
        """List the layers available for this window that are completed."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT layer_name, group_idx FROM completed_layers
            WHERE group_name = ? AND window_name = ?
            """,
            (group, name),
        )
        return [(row["layer_name"], row["group_idx"]) for row in cursor.fetchall()]

    @_retry_on_locked
    @override
    def is_layer_completed(
        self, group: str, name: str, layer_name: str, group_idx: int = 0
    ) -> bool:
        """Check whether the specified layer is completed in the given window."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT 1 FROM completed_layers
            WHERE group_name = ? AND window_name = ? AND layer_name = ? AND group_idx = ?
            """,
            (group, name, layer_name, group_idx),
        )
        return cursor.fetchone() is not None

    @_retry_on_locked
    @override
    def mark_layer_completed(
        self, group: str, name: str, layer_name: str, group_idx: int = 0
    ) -> None:
        """Mark the specified layer completed for the given window."""
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT OR IGNORE INTO completed_layers (group_name, window_name, layer_name, group_idx)
                VALUES (?, ?, ?, ?)
                """,
                (group, name, layer_name, group_idx),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


class SQLiteWindowStorageFactory(WindowStorageFactory):
    """Factory class for SQLiteWindowStorage."""

    @override
    def get_storage(self, ds_path: UPath) -> SQLiteWindowStorage:
        """Get a SQLiteWindowStorage for the given dataset path."""
        return SQLiteWindowStorage(ds_path)
