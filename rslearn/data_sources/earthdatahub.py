"""Data sources backed by EarthDataHub-hosted datasets."""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import numpy as np
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
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    else:
        t = t.astimezone(UTC)
    return datetime(t.year, t.month, t.day, tzinfo=UTC)


def _bounds_to_lon_ranges_0_360(
    bounds: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    """Convert lon bounds to one or two non-wrapping ranges in [0, 360]."""
    min_lon, _, max_lon, _ = bounds

    # Dateline-crossing bounds in [-180, 180] would have min_lon > max_lon.
    if min_lon > max_lon:
        raise ValueError(
            "ERA5LandDailyUTCv1 does not yet support longitude ranges that cross the dateline "
            f"(got bounds min_lon={min_lon}, max_lon={max_lon})."
        )

    if min_lon >= 0 and max_lon >= 0:
        return [(min_lon, max_lon)]
    if min_lon < 0 and max_lon < 0:
        return [(min_lon + 360.0, max_lon + 360.0)]

    # Bounds cross 0 degrees (e.g. [-5, 5]) which wraps in the 0..360 convention.
    return [(min_lon + 360.0, 360.0), (0.0, max_lon)]


class ERA5LandDailyUTCv1(DataSource[Item]):
    """ERA5-Land daily UTC (v1) hosted on EarthDataHub.

    This data source reads from the EarthDataHub Zarr store and writes daily GeoTIFFs
    into the dataset tile store.

    Supported bands:
    - t2m: 2m temperature (units: K)
    - tp: total precipitation (units: m)

    Authentication:
        EarthDataHub uses token-based auth. Configure your netrc file so HTTP clients
        can attach the token automatically. On Linux and MacOS the netrc path is
        `~/.netrc`.
    """

    DEFAULT_ZARR_URL = (
        "https://data.earthdatahub.destine.eu/era5/era5-land-daily-utc-v1.zarr"
    )
    ALLOWED_BANDS = {"t2m", "tp"}
    PIXEL_SIZE_DEGREES = 0.1

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
        self.zarr_url = zarr_url
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
            raise ValueError("band_names must be set if layer_config is not in the context")

        invalid_bands = [b for b in self.band_names if b not in self.ALLOWED_BANDS]
        if invalid_bands:
            raise ValueError(
                f"unsupported ERA5LandDailyUTCv1 band(s): {invalid_bands}; "
                f"supported: {sorted(self.ALLOWED_BANDS)}"
            )

        self._ds: Any | None = None

    def _get_dataset(self):
        if self._ds is not None:
            return self._ds

        storage_options: dict[str, Any] | None = None
        if self.zarr_url.startswith("http://") or self.zarr_url.startswith("https://"):
            storage_options = {"client_kwargs": {"trust_env": self.trust_env}}

        self._ds = xr.open_dataset(
            self.zarr_url,
            engine="zarr",
            chunks={},
            storage_options=storage_options,
        )
        return self._ds

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[Item]]]:
        """Get daily items intersecting the given geometries.

        Returns one item per UTC day that intersects each requested geometry time
        range.
        """
        if query_config.space_mode != SpaceMode.MOSAIC:
            raise ValueError("expected mosaic space mode in the query configuration")

        if self.bounds is not None:
            min_lon, min_lat, max_lon, max_lat = self.bounds
            item_shp = shapely.box(min_lon, min_lat, max_lon, max_lat)
        else:
            item_shp = shapely.box(-180, -90, 180, 90)

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
        assert isinstance(serialized_item, dict)
        return Item.deserialize(serialized_item)

    def _get_effective_bounds(
        self, geometries: list[STGeometry]
    ) -> tuple[float, float, float, float]:
        if self.bounds is not None:
            min_lon, min_lat, max_lon, max_lat = self.bounds
            return (min_lon, min_lat, max_lon, max_lat)

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
        return (min_lon, min_lat, max_lon, max_lat)

    def _write_geotiff(
        self,
        tif_path: str,
        lat: np.ndarray,
        lon: np.ndarray,
        band_arrays: list[np.ndarray],
    ) -> None:
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

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        ds = self._get_dataset()

        for item, item_geoms in zip(items, geometries):
            if tile_store.is_raster_ready(item.name, self.band_names):
                continue

            if item.geometry.time_range is None:
                raise ValueError("expected item to have a time range")

            day_start = _floor_to_utc_day(item.geometry.time_range[0])
            day_str = f"{day_start.year:04d}-{day_start.month:02d}-{day_start.day:02d}"

            bounds = self._get_effective_bounds(item_geoms)
            lon_ranges_0_360 = _bounds_to_lon_ranges_0_360(bounds)
            min_lat = bounds[1]
            max_lat = bounds[3]

            # Subset the dataset before computing, for performance.
            # Latitude is descending in the dataset.
            sel_kwargs_base: dict[str, Any] = dict(
                valid_time=day_str,
                latitude=slice(max_lat, min_lat),
            )

            band_arrays: list[np.ndarray] = []
            lat: np.ndarray | None = None
            lon: np.ndarray | None = None
            for band in self.band_names:
                if len(lon_ranges_0_360) == 1:
                    da = ds[band].sel(
                        **sel_kwargs_base,
                        longitude=slice(lon_ranges_0_360[0][0], lon_ranges_0_360[0][1]),
                    )
                else:
                    parts = [
                        ds[band].sel(**sel_kwargs_base, longitude=slice(lo, hi))
                        for (lo, hi) in lon_ranges_0_360
                    ]
                    da = xr.concat(parts, dim="longitude")

                if band == "t2m" and self.temperature_unit == "celsius":
                    da = da - 273.15

                da = da.load()
                if lat is None:
                    lat = da["latitude"].to_numpy()
                    lon = da["longitude"].to_numpy()
                band_arrays.append(da.to_numpy())

            assert lat is not None and lon is not None
            if lat.size == 0 or lon.size == 0:
                raise ValueError(
                    f"ERA5LandDailyUTCv1 selection returned empty grid for item {item.name} "
                    f"(bounds={bounds})"
                )

            with tempfile.TemporaryDirectory() as tmp_dir:
                local_tif_fname = os.path.join(tmp_dir, f"{item.name}.tif")
                self._write_geotiff(local_tif_fname, lat, lon, band_arrays)
                tile_store.write_raster_file(
                    item.name, self.band_names, UPath(local_tif_fname)
                )
