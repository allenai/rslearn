"""Data sources backed by EarthDataHub-hosted datasets."""

from __future__ import annotations

import base64
import json
import math
import os
import tempfile
import time as time_mod
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd
import rasterio
import shapely
import xarray as xr
from rasterio.transform import from_origin
from upath import UPath

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_EPSG, WGS84_PROJECTION
from rslearn.data_sources import DataSource, DataSourceContext, Item
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.geometry import STGeometry

logger = get_logger(__name__)


def _floor_to_utc_day(t: datetime) -> datetime:
    """Return the UTC day boundary (00:00) for a datetime.

    If `t` is naive, it is treated as UTC. If `t` is timezone-aware, it is converted
    to UTC before flooring.
    """
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    else:
        t = t.astimezone(UTC)
    return datetime(t.year, t.month, t.day, tzinfo=UTC)


def _bounds_to_lon_ranges_0_360(
    bounds: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    """Convert lon bounds to one or two non-wrapping ranges in [0, 360].

    Expects non-wrapping bounds where min_lon <= max_lon.
    """
    min_lon, _, max_lon, _ = bounds

    if min_lon >= 0 and max_lon >= 0:
        return [(min_lon, max_lon)]
    if min_lon < 0 and max_lon < 0:
        return [(min_lon + 360.0, max_lon + 360.0)]

    # Bounds cross 0 degrees (e.g. [-5, 5]) which wraps in the 0..360 convention.
    return [(min_lon + 360.0, 360.0), (0.0, max_lon)]


def _snap_bounds_outward(
    bounds: tuple[float, float, float, float],
    step_degrees: float,
) -> tuple[float, float, float, float]:
    """Snap lon/lat bounds outward to a fixed grid.

    ERA5(-Land) datasets are provided on a regular 0.1° lat/lon grid. When a window is
    much smaller than the grid spacing, its geographic bounds may not contain any grid
    *centers*, which leads to empty `xarray.sel(..., slice(...))` selections.

    Snapping outward ensures at least one grid cell is selected when there is any
    overlap.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    if min_lon > max_lon or min_lat > max_lat:
        raise ValueError(f"invalid bounds: {bounds}")

    min_lon = math.floor(min_lon / step_degrees) * step_degrees
    min_lat = math.floor(min_lat / step_degrees) * step_degrees
    max_lon = math.ceil(max_lon / step_degrees) * step_degrees
    max_lat = math.ceil(max_lat / step_degrees) * step_degrees

    # Ensure non-empty after snapping.
    if max_lon == min_lon:
        max_lon = min_lon + step_degrees
    if max_lat == min_lat:
        max_lat = min_lat + step_degrees

    return (min_lon, min_lat, max_lon, max_lat)


class ERA5LandDailyUTCv1(DataSource[Item]):
    """ERA5-Land daily UTC (v1) hosted on EarthDataHub.

    This data source reads from the EarthDataHub Zarr store and writes daily GeoTIFFs
    into the dataset tile store.

    Supported bands:
    - d2m: 2m dewpoint temperature (units: K)
    - e: evaporation (units: m of water equivalent)
    - pev: potential evaporation (units: m)
    - ro: runoff (units: m)
    - sp: surface pressure (units: Pa)
    - ssr: surface net short-wave (solar) radiation (units: J m-2)
    - ssrd: surface short-wave (solar) radiation downwards (units: J m-2)
    - str: surface net long-wave (thermal) radiation (units: J m-2)
    - swvl1: volumetric soil water layer 1 (units: m3 m-3)
    - swvl2: volumetric soil water layer 2 (units: m3 m-3)
    - t2m: 2m temperature (units: K)
    - tp: total precipitation (units: m)
    - u10: 10m U wind component (units: m s-1)
    - v10: 10m V wind component (units: m s-1)

    Authentication:
        EarthDataHub uses token-based auth. There are two ways to authenticate:

        1. Set the ``EARTHDATAHUB_TOKEN`` environment variable.
        2. Configure your netrc file so HTTP clients
        can attach the token automatically and keep ``trust_env=True``.
    """

    DEFAULT_ZARR_URL = (
        "https://data.earthdatahub.destine.eu/era5/era5-land-daily-utc-v1.zarr"
    )
    ALLOWED_BANDS = {
        "d2m",
        "e",
        "pev",
        "ro",
        "sp",
        "ssr",
        "ssrd",
        "str",
        "swvl1",
        "swvl2",
        "t2m",
        "tp",
        "u10",
        "v10",
    }
    PIXEL_SIZE_DEGREES = 0.1
    DEFAULT_TIME_CHUNK_SIZE = 75
    STALE_LOCK_TIMEOUT_SECONDS = 1800  # 30 minutes
    LOCK_POLL_INTERVAL_SECONDS = 5

    def __init__(
        self,
        band_names: list[str] | None = None,
        zarr_url: str = DEFAULT_ZARR_URL,
        bounds: list[float] | None = None,
        temperature_unit: Literal["celsius", "kelvin"] = "kelvin",
        trust_env: bool = True,
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Initialize a new ERA5LandDailyUTCv1 instance.

        Args:
            band_names: list of bands to ingest. If omitted and a LayerConfig is
                provided via context, bands are inferred from that layer's band sets.
            zarr_url: URL/path to the EarthDataHub Zarr store.
            bounds: optional bounding box as [min_lon, min_lat, max_lon, max_lat]
                in degrees (WGS84). For best performance, set bounds to your area of
                interest.
            temperature_unit: units to return for `t2m` ("celsius" or "kelvin").
            trust_env: if True (default), allow the underlying HTTP client to read
                environment configuration (including netrc) for auth/proxies.
            context: rslearn data source context.
        """
        self._ds_path = context.ds_path
        self.zarr_url = zarr_url
        if bounds is not None:
            if len(bounds) != 4:
                raise ValueError(
                    "ERA5LandDailyUTCv1 bounds must be [min_lon, min_lat, max_lon, max_lat] "
                    f"(got {bounds!r})."
                )
            min_lon, min_lat, max_lon, max_lat = bounds
            if min_lon > max_lon:
                raise ValueError(
                    "ERA5LandDailyUTCv1 does not yet support longitude ranges that cross the dateline "
                    f"(got bounds min_lon={min_lon}, max_lon={max_lon})."
                )
            if min_lat > max_lat:
                raise ValueError(
                    "ERA5LandDailyUTCv1 bounds must have min_lat <= max_lat "
                    f"(got bounds min_lat={min_lat}, max_lat={max_lat})."
                )
        self.bounds = bounds
        self.temperature_unit = temperature_unit
        self.trust_env = trust_env

        self.band_names: list[str]
        if context.layer_config is not None:
            self.band_names = []
            for band_set in context.layer_config.band_sets:
                for band in band_set.bands:
                    if band not in self.band_names:
                        self.band_names.append(band)
        elif band_names is not None:
            self.band_names = band_names
        else:
            raise ValueError(
                "band_names must be set if layer_config is not in the context"
            )

        invalid_bands = [b for b in self.band_names if b not in self.ALLOWED_BANDS]
        if invalid_bands:
            raise ValueError(
                f"unsupported ERA5LandDailyUTCv1 band(s): {invalid_bands}; "
                f"supported: {sorted(self.ALLOWED_BANDS)}"
            )

        self._ds: Any | None = None

    def _resolve_token(self) -> str | None:
        """Resolve the EarthDataHub token from the environment.

        Returns:
            The token string, or None if ``EARTHDATAHUB_TOKEN`` is not set.
        """
        return os.environ.get("EARTHDATAHUB_TOKEN")

    def _get_dataset(self) -> xr.Dataset:
        """Open (and memoize) the backing ERA5-Land Zarr dataset."""
        if self._ds is not None:
            return self._ds

        storage_options: dict[str, Any] | None = None
        if self.zarr_url.startswith("http://") or self.zarr_url.startswith("https://"):
            storage_options = {"client_kwargs": {"trust_env": self.trust_env}}

            # If an explicit token is available, inject a Basic auth header so
            # authentication works without a netrc file (e.g. on clusters).
            token = self._resolve_token()
            if token:
                credentials = base64.b64encode(f"edh:{token}".encode()).decode()
                storage_options["headers"] = {
                    "Authorization": f"Basic {credentials}",
                }

        self._ds = xr.open_dataset(
            self.zarr_url,
            engine="zarr",
            chunks=None,  # No dask
            storage_options=storage_options,
        )

        # Sanity-check that the source time chunk size hasn't changed.
        if self.zarr_url == self.DEFAULT_ZARR_URL:
            ref_band = self.band_names[0]
            chunks = self._ds[ref_band].encoding.get("chunks")
            if chunks is not None and len(chunks) > 0:
                assert int(chunks[0]) == self.DEFAULT_TIME_CHUNK_SIZE, (
                    f"Expected time chunk size {self.DEFAULT_TIME_CHUNK_SIZE}, "
                    f"got {chunks[0]}"
                )

        return self._ds

    # ------------------------------------------------------------------
    # Chunk-level lock helpers
    # ------------------------------------------------------------------

    def _chunk_lock_dir(self) -> UPath | None:
        """Return the directory for chunk lock files, creating it if needed.

        Returns ``None`` when ``ds_path`` is unavailable (e.g. external usage).
        """
        if self._ds_path is None:
            return None
        lock_dir = self._ds_path / ".era5_chunk_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        return lock_dir

    def _try_acquire_chunk_lock(self, chunk_id: int) -> bool:
        """Try to atomically claim a chunk for downloading.

        Uses ``os.open`` with ``O_CREAT | O_EXCL`` so that exactly one process
        wins the race.  A timestamp is written inside the lock file to allow
        stale-lock detection (see ``STALE_LOCK_TIMEOUT_SECONDS``).

        Returns ``True`` if the lock was acquired (caller must download the
        chunk and then call ``_release_chunk_lock``).  Returns ``True`` also
        when locking is unavailable (no ``ds_path``, or the filesystem does not
        support ``O_EXCL``), in which case the caller should just proceed.
        """
        lock_dir = self._chunk_lock_dir()
        if lock_dir is None:
            return True  # No coordination possible, just proceed.

        lock_file = lock_dir / f"chunk_{chunk_id}.lock"
        lock_path_str = str(lock_file)
        try:
            fd = os.open(lock_path_str, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(time_mod.time()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            # Another worker holds the lock.  Check for staleness.
            try:
                with open(lock_path_str) as f:
                    ts = float(f.read().strip())
                if time_mod.time() - ts > self.STALE_LOCK_TIMEOUT_SECONDS:
                    logger.warning("Removing stale chunk lock: %s", lock_file)
                    os.unlink(lock_path_str)
                    return self._try_acquire_chunk_lock(chunk_id)
            except (OSError, ValueError):
                pass
            return False
        except OSError:
            # Lock mechanism not supported (e.g. cloud storage).
            logger.debug("Chunk locking not available, proceeding without coordination")
            return True

    def _release_chunk_lock(self, chunk_id: int) -> None:
        """Delete the lock file for *chunk_id*."""
        lock_dir = self._chunk_lock_dir()
        if lock_dir is None:
            return
        lock_file = lock_dir / f"chunk_{chunk_id}.lock"
        try:
            os.unlink(str(lock_file))
        except OSError:
            pass

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[Item]]]:
        """Get daily items intersecting the given geometries.

        Returns one item per UTC day that intersects each requested geometry time
        range.
        """
        if query_config.space_mode != SpaceMode.MOSAIC:
            raise ValueError("expected mosaic space mode in the query configuration")

        # If bounds were not explicitly configured, compute them once from
        # the union of all window geometries and persist to ds_path so that
        # ingest() (which runs on a separate instance) can read them back.
        if self.bounds is None:
            min_lon = 180.0
            min_lat = 90.0
            max_lon = -180.0
            max_lat = -90.0
            for geom in geometries:
                wgs84 = geom.to_projection(WGS84_PROJECTION)
                b = wgs84.shp.bounds
                min_lon = min(min_lon, b[0])
                min_lat = min(min_lat, b[1])
                max_lon = max(max_lon, b[2])
                max_lat = max(max_lat, b[3])
            self.bounds = [min_lon, min_lat, max_lon, max_lat]
            self._persist_bounds()

        min_lon, min_lat, max_lon, max_lat = self.bounds
        item_shp = shapely.box(min_lon, min_lat, max_lon, max_lat)

        all_groups: list[list[list[Item]]] = []
        for geometry in geometries:
            if geometry.time_range is None:
                raise ValueError("expected all geometries to have a time range")

            start, end = geometry.time_range
            cur_day = _floor_to_utc_day(start)
            cur_groups: list[list[Item]] = []
            while cur_day < end:
                next_day = cur_day + timedelta(days=1)
                item_name = f"era5land_dailyutc_v1_{cur_day.year:04d}{cur_day.month:02d}{cur_day.day:02d}"
                item_geom = STGeometry(WGS84_PROJECTION, item_shp, (cur_day, next_day))
                cur_groups.append([Item(item_name, item_geom)])
                cur_day = next_day

            all_groups.append(cur_groups)

        return all_groups

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserialize an `Item` previously produced by this data source."""
        assert isinstance(serialized_item, dict)
        return Item.deserialize(serialized_item)

    # ------------------------------------------------------------------
    # Bounds persistence (across prepare / ingest instances)
    # ------------------------------------------------------------------

    def _bounds_file(self) -> UPath | None:
        """Path to the persisted bounds file, or None if ds_path is unavailable."""
        if self._ds_path is None:
            return None
        return self._ds_path / ".era5_bounds.json"

    def _persist_bounds(self) -> None:
        """Write ``self.bounds`` to disk so a future instance can read them."""
        bounds_file = self._bounds_file()
        if bounds_file is None or self.bounds is None:
            return
        bounds_file.parent.mkdir(parents=True, exist_ok=True)
        bounds_file.write_text(json.dumps(self.bounds))

    def _load_persisted_bounds(self) -> list[float] | None:
        """Read bounds previously saved by ``_persist_bounds``, if available."""
        bounds_file = self._bounds_file()
        if bounds_file is None or not bounds_file.exists():
            return None
        try:
            return json.loads(bounds_file.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _write_geotiff(
        self,
        tif_path: str,
        lat: np.ndarray,
        lon: np.ndarray,
        band_arrays: list[np.ndarray],
    ) -> None:
        """Write a GeoTIFF with WGS84 georeferencing from ERA5(-Land) arrays.

        Args:
            tif_path: destination GeoTIFF path.
            lat: 1D latitude coordinate (expected descending, north-to-south).
            lon: 1D longitude coordinate (0..360 in the source dataset).
            band_arrays: band arrays with shape (lat, lon), one per output band.
        """
        if lat.ndim != 1 or lon.ndim != 1:
            raise ValueError("expected 1D latitude/longitude coordinates")
        if len(band_arrays) == 0:
            raise ValueError("expected at least one band array")

        # Convert longitude to [-180, 180) and reorder so GeoTIFF coordinates match
        # common WGS84 conventions and rslearn windows.
        lon = ((lon + 180) % 360) - 180
        lon_sort_idx = np.argsort(lon)
        lon = lon[lon_sort_idx]
        band_arrays = [a[:, lon_sort_idx] for a in band_arrays]

        if len(lon) > 1:
            dx = float(lon[1] - lon[0])
        else:
            dx = self.PIXEL_SIZE_DEGREES
        if len(lat) > 1:
            dy = float(abs(lat[1] - lat[0]))
        else:
            dy = self.PIXEL_SIZE_DEGREES

        # ERA5-Land latitude is descending (north to south). This matches GeoTIFF row order.
        if len(lat) > 1 and lat[1] > lat[0]:
            raise ValueError("expected latitude coordinate to be descending")

        west = float(lon.min() - dx / 2)
        north = float(lat.max() + dy / 2)
        transform = from_origin(west, north, dx, dy)
        crs = f"EPSG:{WGS84_EPSG}"

        array = np.stack(band_arrays, axis=0).astype(np.float32)
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=array.shape[1],
            width=array.shape[2],
            count=array.shape[0],
            dtype=array.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(array)

    def _fetch_and_write_chunk(
        self,
        tile_store: TileStoreWithLayer,
        ds: xr.Dataset,
        chunk_start_idx: int,
        chunk_end_idx: int,
        bounds: tuple[float, float, float, float],
    ) -> None:
        """Fetch an entire time-chunk from the Zarr store and write daily GeoTIFFs.

        Args:
            tile_store: the tile store to write into.
            ds: the opened xarray Dataset.
            chunk_start_idx: first time index (inclusive) of the chunk.
            chunk_end_idx: last time index (exclusive) of the chunk.
            bounds: WGS84 bounding box ``(min_lon, min_lat, max_lon, max_lat)``.
        """
        time_vals = ds["valid_time"].values

        snapped_bounds = _snap_bounds_outward(
            bounds, step_degrees=self.PIXEL_SIZE_DEGREES
        )
        lon_ranges_0_360 = _bounds_to_lon_ranges_0_360(snapped_bounds)
        min_lat = snapped_bounds[1]
        max_lat = snapped_bounds[3]

        time_slice = slice(chunk_start_idx, chunk_end_idx)

        # Select only the needed bands and subset time + space in one go.
        subset = (
            ds[self.band_names]
            .isel(valid_time=time_slice)
            .sel(
                latitude=slice(max_lat, min_lat),
            )
        )
        if len(lon_ranges_0_360) == 1:
            subset = subset.sel(
                longitude=slice(lon_ranges_0_360[0][0], lon_ranges_0_360[0][1]),
            )
        else:
            subset = xr.concat(
                [subset.sel(longitude=slice(lo, hi)) for lo, hi in lon_ranges_0_360],
                dim="longitude",
            )
        subset = subset.load()

        lat = subset["latitude"].to_numpy()
        lon = subset["longitude"].to_numpy()
        if lat.size == 0 or lon.size == 0:
            raise ValueError(
                "ERA5LandDailyUTCv1 chunk selection returned empty grid "
                f"(bounds={snapped_bounds})"
            )

        # Write one GeoTIFF per day in the chunk.
        n_times = chunk_end_idx - chunk_start_idx
        for t_offset in range(n_times):
            day_ts = pd.Timestamp(time_vals[chunk_start_idx + t_offset])
            item_name = (
                f"era5land_dailyutc_v1_"
                f"{day_ts.year:04d}{day_ts.month:02d}{day_ts.day:02d}"
            )

            if tile_store.is_raster_ready(item_name, self.band_names):
                continue

            band_arrays: list[np.ndarray] = []
            for band in self.band_names:
                arr = subset[band].values[t_offset]  # (n_lat, n_lon)
                if band == "t2m" and self.temperature_unit == "celsius":
                    arr = arr - 273.15
                band_arrays.append(arr)

            with tempfile.TemporaryDirectory() as tmp_dir:
                local_tif_fname = os.path.join(tmp_dir, f"{item_name}.tif")
                self._write_geotiff(local_tif_fname, lat, lon, band_arrays)
                tile_store.write_raster_file(
                    item_name, self.band_names, UPath(local_tif_fname)
                )

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest daily ERA5-Land rasters for the requested items/geometries.

        Instead of fetching one day at a time (which redundantly downloads the
        same Zarr time-chunk for every day in the chunk), this method:

        1. Groups the requested day-items by their underlying Zarr time-chunk.
        2. Uses a lock file so that only one worker downloads a given chunk.
        3. Fetches the entire chunk once and writes *all* daily GeoTIFFs from
           that chunk into the tile store.

        Workers that lose the lock wait for their specific day(s) to appear
        in the tile store.
        """
        ds = self._get_dataset()
        time_vals = ds["valid_time"].values
        time_chunk_size = self.DEFAULT_TIME_CHUNK_SIZE
        n_times = len(time_vals)

        # Resolve spatial bounds.  Priority:
        # 1. Explicitly configured bounds (from dataset config).
        # 2. Bounds persisted by get_items() during prepare (full window union).
        # 3. Fallback: compute from this batch's geometries (partial).
        if self.bounds is None:
            persisted = self._load_persisted_bounds()
            if persisted is not None:
                self.bounds = persisted
        if self.bounds is not None:
            bounds = (
                float(self.bounds[0]),
                float(self.bounds[1]),
                float(self.bounds[2]),
                float(self.bounds[3]),
            )
        else:
            all_geoms = [g for geom_list in geometries for g in geom_list]
            min_lon = 180.0
            min_lat = 90.0
            max_lon = -180.0
            max_lat = -90.0
            for geom in all_geoms:
                wgs84 = geom.to_projection(WGS84_PROJECTION)
                b = wgs84.shp.bounds
                min_lon = min(min_lon, b[0])
                min_lat = min(min_lat, b[1])
                max_lon = max(max_lon, b[2])
                max_lat = max(max_lat, b[3])
            bounds = (min_lon, min_lat, max_lon, max_lat)

        # Group items by chunk ID.
        items_by_chunk: dict[int, list[Item]] = {}

        for item in items:
            if tile_store.is_raster_ready(item.name, self.band_names):
                continue

            if item.geometry.time_range is None:
                raise ValueError("expected item to have a time range")

            day_start = _floor_to_utc_day(item.geometry.time_range[0])
            # Build a timezone-naive datetime64 to avoid numpy timezone warnings.
            day_np = np.datetime64(day_start.replace(tzinfo=None), "ns")
            time_idx = int(np.searchsorted(time_vals, day_np))
            time_idx = min(time_idx, n_times - 1)
            chunk_id = time_idx // time_chunk_size

            if chunk_id not in items_by_chunk:
                items_by_chunk[chunk_id] = []
            items_by_chunk[chunk_id].append(item)

        # Process each chunk.
        for chunk_id in items_by_chunk:
            chunk_start = chunk_id * time_chunk_size
            chunk_end = min((chunk_id + 1) * time_chunk_size, n_times)

            if self._try_acquire_chunk_lock(chunk_id):
                try:
                    logger.info(
                        "Fetching ERA5 chunk %d (time indices %d..%d) "
                        "for %d requested item(s)",
                        chunk_id,
                        chunk_start,
                        chunk_end - 1,
                        len(items_by_chunk[chunk_id]),
                    )
                    self._fetch_and_write_chunk(
                        tile_store, ds, chunk_start, chunk_end, bounds
                    )
                finally:
                    self._release_chunk_lock(chunk_id)
            else:
                # Another worker is handling this chunk — poll until our
                # items are ready.
                logger.info(
                    "Chunk %d is being processed by another worker, "
                    "waiting for %d item(s)...",
                    chunk_id,
                    len(items_by_chunk[chunk_id]),
                )
                for item in items_by_chunk[chunk_id]:
                    while not tile_store.is_raster_ready(item.name, self.band_names):
                        time_mod.sleep(self.LOCK_POLL_INTERVAL_SECONDS)
