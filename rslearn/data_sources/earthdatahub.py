"""Data sources backed by EarthDataHub-hosted datasets."""

from __future__ import annotations

import base64
import math
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import shapely
import xarray as xr
from rasterio.crs import CRS

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_EPSG, WGS84_PROJECTION
from rslearn.data_sources import DataSource, DataSourceContext, Item
from rslearn.data_sources.utils import MatchedItemGroup
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.raster_array import RasterArray

logger = get_logger(__name__)


def _floor_to_utc_day(t: datetime) -> datetime:
    """Return the UTC day boundary (00:00) for a datetime.

    If `t` is naive, it is treated as UTC. If `t` is timezone-aware, it is converted
    to UTC before flooring.

    Args:
        t: Input datetime, either naive or timezone-aware.

    Returns:
        Timezone-aware datetime snapped to 00:00 UTC on the same UTC calendar day.
    """
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    else:
        t = t.astimezone(UTC)
    return datetime(t.year, t.month, t.day, tzinfo=UTC)


def _bounds_to_lon_ranges_0_360(
    snap_min_lon: float,
    snap_max_lon: float,
) -> list[tuple[float, float]]:
    """Convert lon bounds to one or two non-wrapping ranges in [0, 360].

    Expects non-wrapping bounds where snap_min_lon <= snap_max_lon.

    Args:
        snap_min_lon: Snapped minimum longitude bound in [-180, 180].
        snap_max_lon: Snapped maximum longitude bound in [-180, 180].

    Returns:
        One or two longitude intervals in [0, 360] that cover the input bounds.
    """
    min_lon, max_lon = snap_min_lon, snap_max_lon

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
    """Snap lon/lat bounds outward to the enclosing ERA5-Land grid edges.

    A request window may be smaller than the 0.1° ERA5-Land grid spacing and
    fit entirely within a single grid cell. Snapping the bounds outward to the
    enclosing grid-aligned edges ensures the window always covers at least the
    surrounding grid cell(s), so downstream overlap logic can identify the
    correct chunks to fetch.

    Args:
        bounds: Bounding box ``(min_lon, min_lat, max_lon, max_lat)``.
        step_degrees: Grid spacing in degrees for lon/lat snapping.

    Returns:
        Bounding box expanded to enclosing grid-aligned edges.
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


def _np_datetime64_to_utc(ts: np.datetime64) -> datetime:
    """Convert a numpy datetime64 to a timezone-aware Python datetime (UTC).

    Args:
        ts: Numpy datetime value interpreted as UTC.

    Returns:
        Python datetime with ``tzinfo=UTC``.
    """
    return pd.Timestamp(ts).to_pydatetime(warn=False).replace(tzinfo=UTC)


class ERA5LandChunkItem(Item):
    """An item representing a single Zarr chunk in the ERA5-Land dataset."""

    def __init__(
        self,
        name: str,
        geometry: STGeometry,
        time_chunk: int,
        lat_chunk: int,
        lon_chunk: int,
    ) -> None:
        """Create am earthdatahub ERA5L chunk specific Item.

        Args:
            name: unique name of the item.
            geometry: the spatial and temporal extent of the item.
            time_chunk: time-dimension chunk index in the Zarr store.
            lat_chunk: latitude-dimension chunk index in the Zarr store.
            lon_chunk: longitude-dimension chunk index in the Zarr store.
        """
        super().__init__(name, geometry)
        self.time_chunk = time_chunk
        self.lat_chunk = lat_chunk
        self.lon_chunk = lon_chunk

    def serialize(self) -> dict:
        """Serialize the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["time_chunk"] = self.time_chunk
        d["lat_chunk"] = self.lat_chunk
        d["lon_chunk"] = self.lon_chunk
        return d

    @staticmethod
    def deserialize(d: dict) -> ERA5LandChunkItem:
        """Deserialize an item from a JSON-decoded dictionary."""
        item = Item.deserialize(d)
        return ERA5LandChunkItem(
            name=item.name,
            geometry=item.geometry,
            time_chunk=d["time_chunk"],
            lat_chunk=d["lat_chunk"],
            lon_chunk=d["lon_chunk"],
        )


class ERA5LandDailyUTCv1(DataSource[ERA5LandChunkItem]):
    """ERA5-Land daily UTC (v1) hosted on EarthDataHub.

    This data source reads from the EarthDataHub Zarr store and writes
    multi-timestep GeoTIFFs into the dataset tile store.  Each item corresponds
    to exactly one Zarr chunk, currently of size time (75d) x lat (15°) x lon (30°),
    so only the chunks needed for each window are fetched.

    The recommended configuration uses ``SINGLE_COMPOSITE`` space mode with
    ``SPATIAL_MOSAIC_TEMPORAL_STACK`` compositing method.  Materialization reads
    all chunk items in a group, mosaics them spatially, stacks them temporally,
    clips to the request time range, and produces a single ``(C, T, H, W)``
    raster.

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
    ERA5L_PIX_DEG = 0.1
    DEFAULT_TIME_CHUNK_SIZE = 75
    NODATA_VALUE: float = -9999.0

    def __init__(
        self,
        band_names: list[str] | None = None,
        zarr_url: str = DEFAULT_ZARR_URL,
        trust_env: bool = True,
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Initialize a new ERA5LandDailyUTCv1 instance.

        Args:
            band_names: list of bands to ingest. If omitted and a LayerConfig is
                provided via context, bands are inferred from that layer's band sets.
            zarr_url: URL/path to the EarthDataHub Zarr store.
            trust_env: if True (default), allow the underlying HTTP client to read
                environment configuration (including netrc) for auth/proxies.
            context: rslearn data source context.
        """
        self.zarr_url = zarr_url
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

        self._ds: xr.Dataset | None = None

    def _get_dataset(self) -> xr.Dataset:
        """Open (and memoize) the backing ERA5-Land Zarr dataset."""
        if self._ds is not None:
            return self._ds

        storage_options: dict[str, Any] | None = None
        if self.zarr_url.startswith("http://") or self.zarr_url.startswith("https://"):
            storage_options = {"client_kwargs": {"trust_env": self.trust_env}}

            # If an explicit token is available, inject a Basic auth header so
            # authentication works without a netrc file (e.g. on clusters).
            token = os.environ.get("EARTHDATAHUB_TOKEN")
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

        # Warn if the upstream time chunk size has drifted from the value
        # assumed when this data source was written.
        if self.zarr_url == self.DEFAULT_ZARR_URL:
            ref_band = self.band_names[0]
            chunks = self._ds[ref_band].encoding.get("chunks")
            if chunks is not None and len(chunks) > 0:
                if int(chunks[0]) != self.DEFAULT_TIME_CHUNK_SIZE:
                    logger.warning(
                        "EarthDataHub Zarr time chunk size changed from %d to "
                        "%d since this data source was written. The code will "
                        "still run but may not fetch data as efficiently.",
                        self.DEFAULT_TIME_CHUNK_SIZE,
                        int(chunks[0]),
                    )

        # Validate/cache chunk sizes so callers get a clear error at open time
        ref_band = self.band_names[0]
        chunks = self._ds[ref_band].encoding.get("chunks")
        if chunks is None or len(chunks) != 3:
            raise ValueError(
                f"Expected 3D chunk encoding (time, lat, lon) for band "
                f"'{ref_band}', got {chunks!r}. Ensure the Zarr store has "
                f"explicit chunk metadata for all three dimensions."
            )
        self._chunk_sizes = (int(chunks[0]), int(chunks[1]), int(chunks[2]))

        return self._ds

    # ------------------------------------------------------------------
    # Spatial / temporal chunk overlap helpers
    # ------------------------------------------------------------------

    def _find_overlapping_time_chunks(
        self,
        time_vals: np.ndarray,
        n_times: int,
        time_cs: int,
        start: datetime,
        end: datetime,
    ) -> range:
        """Return the range of time-chunk IDs that temporally overlap the query range ``[start, end)``.

        Args:
            time_vals: All dataset available time steps (days)
            n_times: Total number of time steps (days) in dataset.
            time_cs: Time chunk size (in days), should be = DEFAULT_TIME_CHUNK_SIZE = 75
            start: Inclusive query start time.
            end: Exclusive query end time.

        Returns:
            Range of overlapping time chunk indices. May be empty.
        """
        start_floor = _floor_to_utc_day(start)
        start_np = np.datetime64(start_floor.replace(tzinfo=None), "ns")
        end_np = np.datetime64(end.replace(tzinfo=None), "ns")

        start_day_idx = int(np.searchsorted(time_vals, start_np))
        end_day_idx = int(np.searchsorted(time_vals, end_np, side="right"))

        # If the query range falls entirely outside the dataset, return empty.
        if start_day_idx >= n_times or end_day_idx == 0:
            return range(0, 0)

        first_chunk_idx = start_day_idx // time_cs
        last_chunk_idx = (end_day_idx - 1) // time_cs
        return range(first_chunk_idx, last_chunk_idx + 1)

    def _find_overlapping_lat_chunks(
        self,
        lat_vals: np.ndarray,
        lat_cs: int,
        snap_min_lat: float,
        snap_max_lat: float,
    ) -> range | None:
        """Return the range of lat-chunk IDs that overlap the latitude band.

        Returns ``None`` if no grid cells fall within the band.

        Args:
            lat_vals: All dataset available latitude coordinate values.
            lat_cs: Latitude chunk size.
            snap_min_lat: Snapped minimum latitude bound.
            snap_max_lat: Snapped maximum latitude bound.

        Returns:
            Range of overlapping latitude chunk indices, or ``None`` when no overlap.
        """
        lat_mask = (lat_vals >= snap_min_lat) & (lat_vals <= snap_max_lat)
        lat_grid_indices = np.where(lat_mask)[0]
        if len(lat_grid_indices) == 0:
            return None
        first_chunk_lat_idx = int(lat_grid_indices[0]) // lat_cs
        last_chunk_lat_idx = int(lat_grid_indices[-1]) // lat_cs
        return range(first_chunk_lat_idx, last_chunk_lat_idx + 1)

    def _find_overlapping_lon_chunks(
        self,
        lon_vals: np.ndarray,
        lon_cs: int,
        snap_min_lon: float,
        snap_max_lon: float,
    ) -> list[int]:
        """Return sorted list of lon-chunk IDs that overlap the longitude range.

        Handles the [-180, 180) → [0, 360) conversion internally.

        Args:
            lon_vals: Dataset longitude coordinate values in [0, 360).
            lon_cs: Longitude chunk size.
            snap_min_lon: Snapped minimum longitude bound.
            snap_max_lon: Snapped maximum longitude bound.

        Returns:
            Sorted list of overlapping longitude chunk indices.
            Returning a list instead of a range (like _find_overlapping_lat_chunks) because
            longitude can wrap around the antimeridian (180°/-180° boundary),
            which means the overlapping chunks may not be contiguous.
        """
        lon_ranges = _bounds_to_lon_ranges_0_360(snap_min_lon, snap_max_lon)
        chunk_idx_set: set[int] = set()
        for lo, hi in lon_ranges:
            lon_mask = (lon_vals >= lo) & (lon_vals <= hi)
            lon_grid_indices = np.where(lon_mask)[0]
            if len(lon_grid_indices) == 0:
                continue
            first_chunk_lon_idx = int(lon_grid_indices[0]) // lon_cs
            last_chunk_lon_idx = int(lon_grid_indices[-1]) // lon_cs
            for chunk_idx in range(first_chunk_lon_idx, last_chunk_lon_idx + 1):
                chunk_idx_set.add(chunk_idx)
        return sorted(chunk_idx_set)

    # ------------------------------------------------------------------
    # get_items / ingest
    # ------------------------------------------------------------------

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[ERA5LandChunkItem]]]:
        """Get chunk-level items intersecting the given geometries.

        Each item maps 1-to-1 to a single Zarr chunk identified by a
        ``(time, lat, lon)`` chunk index triple.  Only the chunks that
        spatially and temporally overlap each geometry are returned.

        Use with ``SINGLE_COMPOSITE`` space mode and
        ``SPATIAL_MOSAIC_TEMPORAL_STACK`` compositing so that
        materialization can mosaic spatially and stack temporally.

        Args:
            geometries: Query geometries with time ranges.
            query_config: Query configuration controlling grouping/compositing modes.

        Returns:
            Nested item groups as expected by rslearn materialization.
        """
        if query_config.space_mode != SpaceMode.SINGLE_COMPOSITE:
            raise ValueError(
                "ERA5LandDailyUTCv1 requires SINGLE_COMPOSITE space mode "
                f"(got {query_config.space_mode})"
            )
        if query_config.min_matches != 0:
            raise ValueError(
                "min_matches is not supported for ERA5LandDailyUTCv1; set min_matches=0"
            )

        ds = self._get_dataset()
        time_vals = ds["valid_time"].values
        lat_vals = ds["latitude"].values
        lon_vals = ds["longitude"].values
        n_times = len(time_vals)
        n_lat = len(lat_vals)
        n_lon = len(lon_vals)
        time_cs, lat_cs, lon_cs = self._chunk_sizes

        all_groups: list[list[list[ERA5LandChunkItem]]] = []
        for geometry in geometries:
            if geometry.time_range is None:
                raise ValueError("expected all geometries to have a time range")

            # --- temporal overlap ---
            time_chunks_idx = self._find_overlapping_time_chunks(
                time_vals, n_times, time_cs, *geometry.time_range
            )

            # --- spatial overlap ---
            wgs84 = geometry.to_projection(WGS84_PROJECTION)
            bbox = wgs84.shp.bounds  # (min_lon, min_lat, max_lon, max_lat)
            snapped = _snap_bounds_outward(bbox, self.ERA5L_PIX_DEG)

            lat_chunk_indices = self._find_overlapping_lat_chunks(
                lat_vals, lat_cs, snapped[1], snapped[3]
            )
            lon_chunk_indices = self._find_overlapping_lon_chunks(
                lon_vals, lon_cs, snapped[0], snapped[2]
            )

            if lat_chunk_indices is None or not lon_chunk_indices:
                all_groups.append([MatchedItemGroup([], geometry.time_range)])
                continue

            # --- build items for every (t, lat, lon) triple ---
            items: list[ERA5LandChunkItem] = []
            for tc_idx in time_chunks_idx:
                tc_start = tc_idx * time_cs
                tc_end = min((tc_idx + 1) * time_cs, n_times)
                chunk_start_dt = _np_datetime64_to_utc(time_vals[tc_start])
                chunk_last_dt = _np_datetime64_to_utc(time_vals[tc_end - 1])
                chunk_end_dt = chunk_last_dt + timedelta(days=1)

                for latc in lat_chunk_indices:
                    latc_start = latc * lat_cs
                    latc_end = min((latc + 1) * lat_cs, n_lat)
                    chunk_lats = lat_vals[latc_start:latc_end]

                    for lonc in lon_chunk_indices:
                        lonc_start = lonc * lon_cs
                        lonc_end = min((lonc + 1) * lon_cs, n_lon)
                        chunk_lons = lon_vals[lonc_start:lonc_end]

                        # Convert lon to [-180, 180) for the item geometry.
                        lons_180 = ((chunk_lons + 180) % 360) - 180
                        half_px = self.ERA5L_PIX_DEG / 2
                        item_shp = shapely.box(
                            float(lons_180.min()) - half_px,
                            float(chunk_lats.min()) - half_px,
                            float(lons_180.max()) + half_px,
                            float(chunk_lats.max()) + half_px,
                        )

                        item_name = f"era5land_v1_t{tc_idx}_y{latc}_x{lonc}"
                        item_geom = STGeometry(
                            WGS84_PROJECTION,
                            item_shp,
                            (chunk_start_dt, chunk_end_dt),
                        )
                        items.append(
                            ERA5LandChunkItem(
                                item_name,
                                item_geom,
                                tc_idx,
                                latc,
                                lonc,
                            )
                        )

            all_groups.append([MatchedItemGroup(items, geometry.time_range)])

        return all_groups

    def deserialize_item(self, serialized_item: dict) -> ERA5LandChunkItem:
        """Deserialize an `Item` previously produced by this data source.

        Args:
            serialized_item: Serialized item dictionary.

        Returns:
            Deserialized ``ERA5LandChunkItem``.
        """
        return ERA5LandChunkItem.deserialize(serialized_item)

    # ------------------------------------------------------------------
    # Projection / bounds helper (used by ingest)
    # ------------------------------------------------------------------

    def _compute_projection_and_bounds(
        self, lat: np.ndarray, lon: np.ndarray
    ) -> tuple[Projection, PixelBounds]:
        """Compute rslearn Projection and PixelBounds from ERA5 lat/lon grids.

        Args:
            lat: 1-D latitude array (any ordering; min/max are used).
            lon: 1-D longitude array (ascending, in [-180, 180)).

        Returns:
            ``(projection, pixel_bounds)`` suitable for ``tile_store.write_raster``.
        """
        projection = Projection(
            CRS.from_epsg(WGS84_EPSG), self.ERA5L_PIX_DEG, -self.ERA5L_PIX_DEG
        )

        west = float(lon.min()) - self.ERA5L_PIX_DEG / 2
        north = float(lat.max()) + self.ERA5L_PIX_DEG / 2
        col_off = round(west / self.ERA5L_PIX_DEG)
        row_off = round(north / (-self.ERA5L_PIX_DEG))
        pixel_bounds: PixelBounds = (
            col_off,
            row_off,
            col_off + len(lon),
            row_off + len(lat),
        )

        return projection, pixel_bounds

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[ERA5LandChunkItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest ERA5-Land chunk items into the tile store.

        Each item corresponds to exactly one Zarr chunk identified by a
        ``(time, lat, lon)`` index triple.  The chunk is fetched with a
        single ``isel`` call and stored as a multi-timestep ``RasterArray``.

        Args:
            tile_store: Target tile store used to persist rasters.
            items: Chunk-level items to ingest.
            geometries: Grouped request geometries provided by the ingest pipeline.
        """
        ds = self._get_dataset()
        time_vals = ds["valid_time"].values
        n_times = len(time_vals)
        n_lat = len(ds["latitude"])
        n_lon = len(ds["longitude"])
        time_cs, lat_cs, lon_cs = self._chunk_sizes

        for item in items:
            if tile_store.is_raster_ready(item, self.band_names):
                continue

            tc_start = item.time_chunk * time_cs
            tc_end = min((item.time_chunk + 1) * time_cs, n_times)
            latc_start = item.lat_chunk * lat_cs
            latc_end = min((item.lat_chunk + 1) * lat_cs, n_lat)
            lonc_start = item.lon_chunk * lon_cs
            lonc_end = min((item.lon_chunk + 1) * lon_cs, n_lon)

            logger.info(
                "Fetching ERA5 chunk t=%d lat=%d lon=%d "
                "(time %d..%d, lat %d..%d, lon %d..%d)",
                item.time_chunk,
                item.lat_chunk,
                item.lon_chunk,
                tc_start,
                tc_end - 1,
                latc_start,
                latc_end - 1,
                lonc_start,
                lonc_end - 1,
            )

            # Fetch exactly one Zarr chunk by index.
            subset = (
                ds[self.band_names]
                .isel(
                    valid_time=slice(tc_start, tc_end),
                    latitude=slice(latc_start, latc_end),
                    longitude=slice(lonc_start, lonc_end),
                )
                .load()
            )

            lat = subset["latitude"].to_numpy()
            lon = subset["longitude"].to_numpy()

            # Convert longitude to [-180, 180) and sort ascending.
            lon = ((lon + 180) % 360) - 180
            lon_sort_idx = np.argsort(lon)
            lon = lon[lon_sort_idx]

            # Build (C, T, H, W) array with lon reordered.
            band_arrays: list[np.ndarray] = []
            for band in self.band_names:
                arr = subset[band].values  # (T, H, W)
                arr = arr[:, :, lon_sort_idx]
                band_arrays.append(arr)

            array = np.stack(band_arrays, axis=0).astype(np.float32)  # (C, T, H, W)
            np.nan_to_num(array, nan=self.NODATA_VALUE, copy=False)

            # Build timestamps: one (start, end) per day in the chunk.
            n_chunk_times = tc_end - tc_start
            timestamps: list[tuple[datetime, datetime]] = []
            for t_offset in range(n_chunk_times):
                day_dt = _np_datetime64_to_utc(time_vals[tc_start + t_offset])
                timestamps.append((day_dt, day_dt + timedelta(days=1)))

            projection, pixel_bounds = self._compute_projection_and_bounds(lat, lon)

            raster = RasterArray(array=array, timestamps=timestamps)
            tile_store.write_raster(
                item,
                self.band_names,
                projection,
                pixel_bounds,
                raster,
                nodata_val=self.NODATA_VALUE,
            )
