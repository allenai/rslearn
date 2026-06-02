"""Copernicus GLO-30 DEM (30m) from the original public S3 bucket.

This module provides a data source for the Copernicus Global 30m Digital Elevation
Model (GLO-30), reading directly from the public AWS S3 bucket
``copernicus-dem-30m`` (no credentials required).

The dataset is split into 1x1-degree COG tiles, with filenames like::

    Copernicus_DSM_COG_10_N47_00_W123_00_DEM/Copernicus_DSM_COG_10_N47_00_W123_00_DEM.tif

In addition to raw elevation, this data source can compute **slope** and **aspect**
as derived bands during ingest. Configure the desired bands via the layer config's
``band_sets``. Supported band names:

- ``elevation``: raw DEM value in meters
- ``slope``: terrain slope in degrees [0, 90)
- ``aspect``: compass direction of steepest descent in degrees [0, 360), -1 for flat

Slope and aspect are computed from the geographic (EPSG:4326) grid with per-row
latitude correction for proper metric gradients.

Items from this data source do not come with a time range (the DEM is static).
"""

from __future__ import annotations

import math
import os
import tempfile
from datetime import timedelta

import numpy as np
import rasterio
import shapely

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource, DataSourceContext, Item
from rslearn.data_sources.utils import MatchedItemGroup
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.geometry import STGeometry, flatten_shape
from rslearn.utils.retry_session import create_retry_session

logger = get_logger(__name__)

SUPPORTED_BANDS = ("elevation", "slope", "aspect")

GLO30_BUCKET = "copernicus-dem-30m"
GLO30_REGION = "eu-central-1"
GLO30_BASE_URL = f"https://{GLO30_BUCKET}.s3.{GLO30_REGION}.amazonaws.com/"

# Approximate meters per degree of latitude (constant).
METERS_PER_DEG_LAT = 111_320.0


def _tile_name(lat: int, lon: int) -> str:
    """Return the GLO-30 tile directory/file name for a 1x1-degree cell.

    Args:
        lat: integer latitude of the cell's southern edge.
        lon: integer longitude of the cell's western edge.

    Returns:
        tile name like ``Copernicus_DSM_COG_10_N47_00_W123_00_DEM``.
    """
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def _tile_url(lat: int, lon: int, base_url: str = GLO30_BASE_URL) -> str:
    """Return the full HTTPS URL for a GLO-30 COG tile."""
    name = _tile_name(lat, lon)
    return f"{base_url}{name}/{name}.tif"


def compute_terrain(
    elevation: np.ndarray,
    pixel_size_deg: float,
    lat_south: float,
    nodata: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute slope and aspect from a DEM in geographic (EPSG:4326) coordinates.

    Uses per-row latitude correction so that gradients are in proper metric units.

    Args:
        elevation: 2D elevation array (rows x cols) in meters. Rows go north-to-south
            (standard GeoTIFF convention, i.e. y_resolution < 0).
        pixel_size_deg: absolute pixel size in degrees (e.g. 1/3600 for 1 arc-second).
        lat_south: latitude of the southern edge of the tile in degrees.
        nodata: source nodata value; these pixels will be NaN in the output.

    Returns:
        (slope_deg, aspect_deg) arrays of same shape as *elevation*, dtype float32.
        slope_deg is in [0, 90). aspect_deg is compass bearing [0, 360), or -1 for
        flat pixels. Nodata pixels are set to NaN.
    """
    dem = elevation.astype(np.float32)
    height, width = dem.shape

    nodata_mask = None
    if nodata is not None:
        nodata_mask = dem == nodata
        dem[nodata_mask] = np.nan

    # Per-row latitude at pixel centres (row 0 = north, row H-1 = south).
    lat_north = lat_south + height * pixel_size_deg
    row_lats = np.linspace(
        lat_north - pixel_size_deg / 2,
        lat_south + pixel_size_deg / 2,
        height,
    )
    row_lat_rad = np.deg2rad(row_lats)

    # Metric spacing per row.
    res_y_m = pixel_size_deg * METERS_PER_DEG_LAT
    res_x_m = pixel_size_deg * METERS_PER_DEG_LAT * np.cos(row_lat_rad)  # (H,)
    # Avoid division by zero near poles.
    res_x_m = np.maximum(res_x_m, 1.0)

    # Partial derivatives using central differences.
    # np.gradient(Z, spacing_axis0, spacing_axis1) returns [dZ/drow, dZ/dcol].
    dz_drow, dz_dcol = np.gradient(dem)

    # Convert from "per pixel" to "per meter".
    # dz_drow already in units of "elevation per row", divide by metric row spacing.
    dz_north = -(dz_drow / res_y_m)  # negate: rows go south
    dz_east = dz_dcol / res_x_m[:, np.newaxis]  # broadcast over columns

    # Slope.
    slope_rad = np.arctan(np.hypot(dz_east, dz_north))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    # Aspect: compass bearing of steepest descent.
    # atan2(east, north) = clockwise angle from north of steepest ascent;
    # add 180° to get steepest descent direction.
    aspect_ascent = np.degrees(np.arctan2(dz_east, dz_north))
    aspect_deg = ((aspect_ascent + 180.0) % 360.0).astype(np.float32)

    # Mark flat pixels as -1.
    flat = slope_deg < 1e-6
    aspect_deg[flat] = -1.0

    # Restore nodata.
    if nodata_mask is not None:
        slope_deg[nodata_mask] = np.nan
        aspect_deg[nodata_mask] = np.nan

    return slope_deg, aspect_deg


class CopernicusGLO30(DataSource):
    """Data source for Copernicus GLO-30 DEM from the public S3 bucket.

    The data is split into 1x1-degree COG tiles.  Items from this data source do
    not carry a time range (the DEM is static).

    The layer's ``band_sets`` should contain a single band set with one or more of
    the supported band names: ``elevation``, ``slope``, ``aspect``.

    Example config::

        {
            "class_path": "rslearn.data_sources.copernicus_glo30.CopernicusGLO30",
            "init_args": {},
            "query_config": {"space_mode": "MOSAIC", "max_matches": 1},
            "ingest": true
        }
    """

    BASE_URL = GLO30_BASE_URL

    def __init__(
        self,
        timeout: timedelta = timedelta(seconds=30),
        context: DataSourceContext = DataSourceContext(),
    ):
        """Initialize a new CopernicusGLO30 instance.

        Args:
            timeout: timeout for HTTP requests.
            context: the data source context.
        """
        if context.layer_config is not None:
            if len(context.layer_config.band_sets) != 1:
                raise ValueError("expected a single band set")
            bands = context.layer_config.band_sets[0].bands
            for b in bands:
                if b not in SUPPORTED_BANDS:
                    raise ValueError(
                        f"unsupported band '{b}'; choose from {SUPPORTED_BANDS}"
                    )
            self.band_names = list(bands)
        else:
            self.band_names = list(SUPPORTED_BANDS)

        self.timeout = timeout
        self.session = create_retry_session()

        self._needs_slope = "slope" in self.band_names
        self._needs_aspect = "aspect" in self.band_names

    def _build_item(self, lon_min: int, lat_min: int) -> Item:
        """Create an Item for a 1x1-degree GLO-30 tile."""
        name = _tile_name(lat_min, lon_min)
        geometry = STGeometry(
            WGS84_PROJECTION,
            shapely.box(lon_min, lat_min, lon_min + 1, lat_min + 1),
            None,
        )
        return Item(name, geometry)

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def get_item_by_name(self, name: str) -> Item:
        """Gets an item by name.

        Args:
            name: the tile name (e.g. ``Copernicus_DSM_COG_10_N47_00_W123_00_DEM``).

        Returns:
            the Item object.
        """
        lat, lon = self._parse_tile_name(name)
        return self._build_item(lon, lat)

    @staticmethod
    def _parse_tile_name(name: str) -> tuple[int, int]:
        """Parse a tile name to (lat_min, lon_min)."""
        # Copernicus_DSM_COG_10_N47_00_W123_00_DEM
        parts = name.split("_")
        # parts: [Copernicus, DSM, COG, 10, N47, 00, W123, 00, DEM]
        ns_part = parts[4]  # e.g. "N47" or "S03"
        ew_part = parts[6]  # e.g. "W123" or "E010"

        lat = int(ns_part[1:])
        if ns_part[0] == "S":
            lat = -lat

        lon = int(ew_part[1:])
        if ew_part[0] == "W":
            lon = -lon

        return lat, lon

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[MatchedItemGroup[Item]]]:
        """Get a list of items intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries.
            query_config: the query configuration.

        Returns:
            list of groups of items for each geometry.
        """
        if query_config.space_mode != SpaceMode.MOSAIC or query_config.max_matches != 1:
            raise ValueError(
                "expected mosaic with max_matches=1 for the query configuration"
            )
        if query_config.min_matches != 0:
            raise ValueError(
                "min_matches is not supported for CopernicusGLO30; set min_matches=0"
            )

        groups = []
        for geometry in geometries:
            wgs84_geometry = geometry.to_wgs84()
            items: list[Item] = []
            seen: set[tuple[int, int]] = set()
            for shp in flatten_shape(wgs84_geometry.shp):
                shp_bounds = shp.bounds
                cell_bounds = (
                    math.floor(shp_bounds[0]),
                    math.floor(shp_bounds[1]),
                    math.ceil(shp_bounds[2]),
                    math.ceil(shp_bounds[3]),
                )
                for lon_min in range(cell_bounds[0], cell_bounds[2]):
                    for lat_min in range(cell_bounds[1], cell_bounds[3]):
                        key = (lon_min, lat_min)
                        if key in seen:
                            continue
                        seen.add(key)
                        items.append(self._build_item(lon_min, lat_min))

            logger.debug(f"Got {len(items)} GLO-30 items for geometry")
            groups.append([MatchedItemGroup(items, geometry.time_range)])

        return groups

    def deserialize_item(self, serialized_item: dict) -> Item:
        """Deserializes an item from JSON-decoded data."""
        return Item.deserialize(serialized_item)

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Downloads elevation COGs from S3 and optionally computes slope/aspect.

        Args:
            tile_store: the tile store to ingest into.
            items: the items to ingest.
            geometries: a list of geometries needed for each item.
        """
        for item in items:
            if tile_store.is_raster_ready(item, self.band_names):
                continue

            lat, lon = self._parse_tile_name(item.name)
            url = _tile_url(lat, lon, base_url=self.BASE_URL)
            logger.debug(f"Downloading GLO-30 tile {item.name} from {url}")

            response = self.session.get(
                url, stream=True, timeout=self.timeout.total_seconds()
            )

            if response.status_code == 404:
                logger.warning(f"Skipping {item.name}: tile not found (likely ocean)")
                continue
            response.raise_for_status()

            with tempfile.TemporaryDirectory() as tmp_dir:
                raw_path = os.path.join(tmp_dir, "raw.tif")
                with open(raw_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                if self._needs_slope or self._needs_aspect:
                    self._ingest_with_derived_bands(
                        tile_store, item, raw_path, lat, tmp_dir
                    )
                else:
                    # Elevation only — pass through the original file.
                    from upath import UPath

                    tile_store.write_raster_file(
                        item,
                        self.band_names,
                        UPath(raw_path),
                        time_range=item.geometry.time_range,
                    )

    def _ingest_with_derived_bands(
        self,
        tile_store: TileStoreWithLayer,
        item: Item,
        raw_path: str,
        lat_south: int,
        tmp_dir: str,
    ) -> None:
        """Read elevation, compute derivatives, and write a multi-band GeoTIFF."""
        from upath import UPath

        with rasterio.open(raw_path) as src:
            elevation = src.read(1)
            profile = src.profile.copy()
            nodata = src.nodata
            pixel_size_deg = abs(src.transform.a)

        band_arrays: dict[str, np.ndarray] = {}
        band_arrays["elevation"] = elevation.astype(np.float32)

        if self._needs_slope or self._needs_aspect:
            slope, aspect = compute_terrain(
                elevation, pixel_size_deg, lat_south, nodata=nodata
            )
            band_arrays["slope"] = slope
            band_arrays["aspect"] = aspect

        # Stack requested bands in config order.
        stack = np.stack([band_arrays[b] for b in self.band_names], axis=0)

        out_path = os.path.join(tmp_dir, "derived.tif")
        profile.update(
            count=len(self.band_names),
            dtype="float32",
            compress="deflate",
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(stack)

        tile_store.write_raster_file(
            item,
            self.band_names,
            UPath(out_path),
            time_range=item.geometry.time_range,
        )
