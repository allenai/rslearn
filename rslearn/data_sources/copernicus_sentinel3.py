"""Data sources for Sentinel-3 data in the ESA Copernicus API."""

import pathlib
import tempfile
from typing import Any
from urllib.parse import quote
from zipfile import ZipFile

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from rslearn.data_sources.copernicus import Copernicus, CopernicusItem
from rslearn.data_sources.data_source import DataSourceContext
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.geometry import STGeometry
from rslearn.utils.interpolation import interpolate_to_grid
from rslearn.utils.raster_array import RasterArray, RasterMetadata

OLCI_REFLECTANCE_BANDS = [f"Oa{idx:02d}_reflectance" for idx in range(1, 22)]
"""Sentinel-3 OLCI EFR top-of-atmosphere reflectance bands."""

SLSTR_REFLECTANCE_BANDS = [f"S{idx}_reflectance" for idx in range(1, 7)]
"""Sentinel-3 SLSTR RBT nadir-view reflectance bands."""

SLSTR_BT_BANDS = [f"S{idx}_BT" for idx in range(7, 10)]
"""Sentinel-3 SLSTR RBT nadir-view brightness-temperature bands."""


def _sentinel3_query_filter(instrument: str, product_type: str) -> str:
    """Build a CDSE OData filter for one Sentinel-3 product family."""

    def attribute_filter(name: str, value: str) -> str:
        return (
            "Attributes/OData.CSC.StringAttribute/any(att:"
            f"att/Name eq '{quote(name)}' and "
            "att/OData.CSC.StringAttribute/Value eq "
            f"'{quote(value)}')"
        )

    return " and ".join(
        [
            "Collection/Name eq 'SENTINEL-3'",
            attribute_filter("instrumentShortName", instrument),
            attribute_filter("processingLevel", "1"),
            attribute_filter("productType", product_type),
        ]
    )


def _requested_bands(
    context: DataSourceContext,
    band_names: list[str] | None,
    default_bands: list[str],
) -> list[str]:
    """Resolve and validate bands requested by a Sentinel-3 layer."""
    if context.layer_config is not None:
        band_names = [
            band
            for band_set in context.layer_config.band_sets
            for band in band_set.bands
        ]
    elif band_names is None:
        band_names = default_bands

    # Preserve the documented order and silently collapse repeated bands across sets.
    requested = set(band_names)
    if not requested:
        raise ValueError("at least one Sentinel-3 band must be requested")
    unknown = requested.difference(default_bands)
    if unknown:
        raise ValueError(f"unsupported Sentinel-3 bands: {sorted(unknown)}")
    return [band for band in default_bands if band in requested]


def _interpolate_tie_points(
    tie_values: npt.NDArray, target_shape: tuple[int, int]
) -> npt.NDArray[np.float32]:
    """Bilinearly interpolate a tie-point array whose swath aligns with the image.

    This is for tie-point grids that cover the same swath as the image, so the first
    and last tie rows/columns coincide with the first and last image rows/columns
    (e.g. OLCI). Use _interpolate_tie_points_xy when the tie grid extends beyond the
    image swath (e.g. SLSTR).
    """
    values = np.asarray(tie_values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a 2D tie-point array, got shape {values.shape}")

    tie_rows = np.linspace(0, target_shape[0] - 1, values.shape[0])
    tie_cols = np.linspace(0, target_shape[1] - 1, values.shape[1])
    interpolator = RegularGridInterpolator((tie_rows, tie_cols), values)
    rows = np.arange(target_shape[0])[:, None]
    cols = np.arange(target_shape[1])[None, :]
    return interpolator((rows, cols)).astype(np.float32)


def _interpolate_tie_points_xy(
    tie_values: npt.NDArray,
    tie_x: npt.NDArray,
    tie_y: npt.NDArray,
    x: npt.NDArray,
    y: npt.NDArray,
) -> npt.NDArray[np.float32]:
    """Bilinearly interpolate a tie-point array using image-plane coordinates.

    SLSTR tie-point grids span a wider swath than the nadir image, so their edges do
    not coincide with the image edges. Instead, both grids carry across-track (x) and
    along-track (y) coordinates, and the tie grid is separable in those coordinates.

    Args:
        tie_values: values on the tie-point grid (rows, columns).
        tie_x: across-track coordinate of each tie point (rows, columns).
        tie_y: along-track coordinate of each tie point (rows, columns).
        x: across-track coordinate of each image pixel.
        y: along-track coordinate of each image pixel.

    Returns:
        the interpolated values with the shape of x, NaN more than one tie spacing
            outside the tie grid.
    """
    values = np.asarray(tie_values, dtype=np.float64)
    tie_x = np.asarray(tie_x, dtype=np.float64)
    tie_y = np.asarray(tie_y, dtype=np.float64)
    if values.ndim != 2 or tie_x.shape != values.shape or tie_y.shape != values.shape:
        raise ValueError(
            "expected 2D tie-point values with matching x/y coordinates; got "
            f"{values.shape}, {tie_x.shape}, and {tie_y.shape}"
        )
    if np.asarray(x).shape != np.asarray(y).shape:
        raise ValueError("expected image x/y coordinates with matching shapes")

    xs = tie_x[0, :]
    ys = tie_y[:, 0]
    if not (np.allclose(tie_x, xs[None, :]) and np.allclose(tie_y, ys[:, None])):
        raise ValueError("tie-point x/y coordinates are not a separable grid")

    # RegularGridInterpolator needs ascending axes; SLSTR x usually decreases.
    if xs.size > 1 and xs[1] < xs[0]:
        xs = xs[::-1]
        values = values[:, ::-1]
    if ys.size > 1 and ys[1] < ys[0]:
        ys = ys[::-1]
        values = values[::-1, :]

    # The outermost image rows can sit slightly beyond the outermost tie rows, so
    # extrapolate linearly, but only up to one tie spacing past the grid edge. Any
    # pixels beyond that margin are left as NaN instead of being extrapolated.
    interpolator = RegularGridInterpolator(
        (ys, xs), values, method="linear", bounds_error=False, fill_value=None
    )
    x_margin = np.abs(np.diff(xs)).max() if xs.size > 1 else 0.0
    y_margin = np.abs(np.diff(ys)).max() if ys.size > 1 else 0.0
    target_x = np.asarray(x, dtype=np.float64)
    target_y = np.asarray(y, dtype=np.float64)
    result = np.full(target_x.shape, np.nan, dtype=np.float32)
    valid = (
        (target_x >= xs[0] - x_margin)
        & (target_x <= xs[-1] + x_margin)
        & (target_y >= ys[0] - y_margin)
        & (target_y <= ys[-1] + y_margin)
    )
    result[valid] = interpolator(
        np.column_stack([target_y[valid], target_x[valid]])
    ).astype(np.float32)
    return result


def _radiance_to_reflectance(
    radiance: npt.NDArray,
    solar_irradiance: npt.NDArray,
    cos_solar_zenith: npt.NDArray,
) -> npt.NDArray[np.float32]:
    """Convert top-of-atmosphere radiance to unitless reflectance."""
    denominator = np.asarray(solar_irradiance, dtype=np.float32) * np.asarray(
        cos_solar_zenith, dtype=np.float32
    )
    valid = np.isfinite(radiance) & np.isfinite(denominator) & (denominator > 1e-6)
    result = np.full(np.shape(radiance), np.nan, dtype=np.float32)
    result[valid] = (
        np.pi * np.asarray(radiance, dtype=np.float32)[valid] / denominator[valid]
    )
    return result


def _extract_zip_members(
    zipf: ZipFile, basenames: set[str], output_dir: str
) -> dict[str, pathlib.Path]:
    """Extract uniquely named SAFE members and return paths by basename."""
    by_basename: dict[str, list[str]] = {name: [] for name in basenames}
    for member_name in zipf.namelist():
        basename = pathlib.PurePosixPath(member_name).name
        if basename in by_basename and not member_name.endswith("/"):
            by_basename[basename].append(member_name)

    extracted: dict[str, pathlib.Path] = {}
    for basename, member_names in by_basename.items():
        if len(member_names) != 1:
            raise ValueError(
                f"expected one {basename} member in Sentinel-3 product, "
                f"found {len(member_names)}"
            )
        extracted[basename] = pathlib.Path(
            zipf.extract(member_names[0], path=output_dir)
        )
    return extracted


def _read_netcdf_variable(
    path: pathlib.Path, variable_name: str, *, mask_and_scale: bool = True
) -> npt.NDArray:
    """Read one variable from a NetCDF file."""
    with xr.open_dataset(path, mask_and_scale=mask_and_scale) as dataset:
        if variable_name not in dataset:
            raise ValueError(f"variable {variable_name!r} not found in {path.name}")
        return np.asarray(dataset[variable_name].values)


def _crop_swath(
    data: npt.NDArray,
    longitude: npt.NDArray,
    latitude: npt.NDArray,
    geometries: list[STGeometry] | None,
    padding: float,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Crop a swath to the row/column envelope needed by the requested windows."""
    values = np.asarray(data)
    lons = np.asarray(longitude)
    lats = np.asarray(latitude)
    if not geometries:
        return values, lons, lats

    needed = np.zeros(lons.shape, dtype=bool)
    for geometry in geometries:
        min_lon, min_lat, max_lon, max_lat = geometry.to_wgs84().shp.bounds
        # A conventional bounds tuple cannot represent a narrow antimeridian-crossing
        # window. Avoid an incorrect crop in that uncommon case.
        if max_lon - min_lon > 180:
            return values, lons, lats
        needed |= (
            (lons >= min_lon - padding)
            & (lons <= max_lon + padding)
            & (lats >= min_lat - padding)
            & (lats <= max_lat + padding)
        )

    rows, cols = np.nonzero(needed & np.isfinite(lons) & np.isfinite(lats))
    if rows.size == 0:
        raise ValueError(
            "Sentinel-3 swath has no geolocated pixels near requested windows"
        )

    row_start = max(0, int(rows.min()) - 1)
    row_end = min(lons.shape[0], int(rows.max()) + 2)
    col_start = max(0, int(cols.min()) - 1)
    col_end = min(lons.shape[1], int(cols.max()) + 2)
    return (
        values[:, row_start:row_end, col_start:col_end],
        lons[row_start:row_end, col_start:col_end],
        lats[row_start:row_end, col_start:col_end],
    )


def _write_swath(
    tile_store: TileStoreWithLayer,
    item: CopernicusItem,
    band_names: list[str],
    data: npt.NDArray,
    longitude: npt.NDArray,
    latitude: npt.NDArray,
    grid_resolution: float,
    nodata_value: float,
) -> None:
    """Interpolate and write one group of Sentinel-3 swath bands."""
    gridded, projection, bounds = interpolate_to_grid(
        data,
        longitude,
        latitude,
        grid_resolution=grid_resolution,
        nodata_value=nodata_value,
    )
    tile_store.write_raster(
        item,
        band_names,
        projection,
        bounds,
        RasterArray(
            chw_array=gridded,
            time_range=item.geometry.time_range,
            metadata=RasterMetadata(nodata_value=nodata_value),
        ),
    )


class Sentinel3OlciEFR(Copernicus):
    """Sentinel-3 OLCI Level-1 EFR top-of-atmosphere reflectance."""

    PRODUCT_TYPE = "OL_1_EFR___"
    DEFAULT_BANDS = OLCI_REFLECTANCE_BANDS

    def __init__(
        self,
        band_names: list[str] | None = None,
        grid_resolution: float = 0.0027,
        nodata_value: float = -9999.0,
        swath_padding: float = 0.1,
        context: DataSourceContext = DataSourceContext(),
        **kwargs: Any,
    ) -> None:
        """Create an OLCI EFR source.

        Args:
            band_names: bands to ingest when no layer configuration is available.
            grid_resolution: intermediate WGS84 grid resolution in degrees.
            nodata_value: value assigned outside the valid swath.
            swath_padding: padding in degrees around requested windows before gridding.
            context: data source context.
            kwargs: additional options for :class:`Copernicus`.
        """
        if grid_resolution <= 0:
            raise ValueError("grid_resolution must be positive")
        if swath_padding < 0:
            raise ValueError("swath_padding cannot be negative")
        self.band_names = _requested_bands(context, band_names, self.DEFAULT_BANDS)
        self.grid_resolution = grid_resolution
        self.nodata_value = nodata_value
        self.swath_padding = swath_padding
        super().__init__(
            glob_to_bands={"OLCI_EFR": self.band_names},
            query_filter=_sentinel3_query_filter("OLCI", self.PRODUCT_TYPE),
            context=context,
            **kwargs,
        )

    def _process_product_zip(
        self,
        tile_store: TileStoreWithLayer,
        item: CopernicusItem,
        local_zip_fname: str,
        geometries: list[STGeometry] | None = None,
    ) -> None:
        """Convert an OLCI SAFE product and ingest requested reflectance bands."""
        radiance_variables = {
            band: band.replace("_reflectance", "_radiance") for band in self.band_names
        }
        required = {
            "instrument_data.nc",
            "tie_geometries.nc",
            "geo_coordinates.nc",
            *[f"{name}.nc" for name in radiance_variables.values()],
        }
        with tempfile.TemporaryDirectory() as tmp_dir, ZipFile(local_zip_fname) as zipf:
            paths = _extract_zip_members(zipf, required, tmp_dir)

            solar_flux = _read_netcdf_variable(
                paths["instrument_data.nc"], "solar_flux", mask_and_scale=False
            ).astype(np.float32)
            detector_index = _read_netcdf_variable(
                paths["instrument_data.nc"], "detector_index", mask_and_scale=False
            ).astype(np.int64)
            target_shape = detector_index.shape

            solar_zenith = _interpolate_tie_points(
                _read_netcdf_variable(paths["tie_geometries.nc"], "SZA"),
                target_shape,
            )
            cos_solar_zenith = np.clip(
                np.cos(np.deg2rad(solar_zenith)), 0.01, None
            ).astype(np.float32)
            latitude = _read_netcdf_variable(
                paths["geo_coordinates.nc"], "latitude"
            ).astype(np.float32)
            longitude = _read_netcdf_variable(
                paths["geo_coordinates.nc"], "longitude"
            ).astype(np.float32)
            if latitude.shape != target_shape or longitude.shape != target_shape:
                raise ValueError(
                    "OLCI geolocation and detector grids have different shapes: "
                    f"{latitude.shape}, {longitude.shape}, and {target_shape}"
                )

            valid_detector = (detector_index >= 0) & (
                detector_index < solar_flux.shape[1]
            )
            safe_detector = np.clip(detector_index, 0, solar_flux.shape[1] - 1)
            arrays = []
            for band_name, variable_name in radiance_variables.items():
                band_index = int(band_name[2:4]) - 1
                radiance = _read_netcdf_variable(
                    paths[f"{variable_name}.nc"], variable_name
                ).astype(np.float32)
                reflectance = _radiance_to_reflectance(
                    radiance,
                    solar_flux[band_index][safe_detector],
                    cos_solar_zenith,
                )
                arrays.append(np.where(valid_detector, reflectance, np.nan))

        data, longitude, latitude = _crop_swath(
            np.stack(arrays),
            longitude,
            latitude,
            geometries,
            self.swath_padding,
        )
        _write_swath(
            tile_store,
            item,
            self.band_names,
            data,
            longitude,
            latitude,
            self.grid_resolution,
            self.nodata_value,
        )


class Sentinel3SlstrRBT(Copernicus):
    """Sentinel-3 SLSTR Level-1 nadir reflectance and brightness temperature."""

    PRODUCT_TYPE = "SL_1_RBT___"
    DEFAULT_BANDS = SLSTR_REFLECTANCE_BANDS + SLSTR_BT_BANDS

    def __init__(
        self,
        band_names: list[str] | None = None,
        reflectance_grid_resolution: float = 0.0045,
        bt_grid_resolution: float = 0.009,
        nodata_value: float = -9999.0,
        swath_padding: float = 0.1,
        context: DataSourceContext = DataSourceContext(),
        **kwargs: Any,
    ) -> None:
        """Create an SLSTR RBT source.

        Args:
            band_names: bands to ingest when no layer configuration is available.
            reflectance_grid_resolution: intermediate WGS84 resolution for S1-S6.
            bt_grid_resolution: intermediate WGS84 resolution for S7-S9.
            nodata_value: value assigned outside the valid swath.
            swath_padding: padding in degrees around requested windows before gridding.
            context: data source context.
            kwargs: additional options for :class:`Copernicus`.
        """
        if reflectance_grid_resolution <= 0 or bt_grid_resolution <= 0:
            raise ValueError("grid resolutions must be positive")
        if swath_padding < 0:
            raise ValueError("swath_padding cannot be negative")
        self.band_names = _requested_bands(context, band_names, self.DEFAULT_BANDS)
        self.reflectance_bands = [
            band for band in SLSTR_REFLECTANCE_BANDS if band in self.band_names
        ]
        self.bt_bands = [band for band in SLSTR_BT_BANDS if band in self.band_names]
        self.reflectance_grid_resolution = reflectance_grid_resolution
        self.bt_grid_resolution = bt_grid_resolution
        self.nodata_value = nodata_value
        self.swath_padding = swath_padding
        band_groups = {}
        if self.reflectance_bands:
            band_groups["SLSTR_RBT_REFLECTANCE"] = self.reflectance_bands
        if self.bt_bands:
            band_groups["SLSTR_RBT_BT"] = self.bt_bands
        super().__init__(
            glob_to_bands=band_groups,
            query_filter=_sentinel3_query_filter("SLSTR", self.PRODUCT_TYPE),
            context=context,
            **kwargs,
        )

    def _process_product_zip(
        self,
        tile_store: TileStoreWithLayer,
        item: CopernicusItem,
        local_zip_fname: str,
        geometries: list[STGeometry] | None = None,
    ) -> None:
        """Convert and ingest requested SLSTR reflectance and thermal bands."""
        needs_reflectance = bool(
            self.reflectance_bands
        ) and not tile_store.is_raster_ready(item, self.reflectance_bands)
        needs_bt = bool(self.bt_bands) and not tile_store.is_raster_ready(
            item, self.bt_bands
        )
        if not needs_reflectance and not needs_bt:
            return

        required: set[str] = set()
        if needs_reflectance:
            required.update(
                {
                    "indices_an.nc",
                    "geometry_tn.nc",
                    "geodetic_an.nc",
                    "cartesian_tx.nc",
                    "cartesian_an.nc",
                }
            )
            for band_name in self.reflectance_bands:
                band = band_name.split("_")[0]
                required.update({f"{band}_radiance_an.nc", f"{band}_quality_an.nc"})
        if needs_bt:
            required.add("geodetic_in.nc")
            required.update(f"{band}_in.nc" for band in self.bt_bands)

        with tempfile.TemporaryDirectory() as tmp_dir, ZipFile(local_zip_fname) as zipf:
            paths = _extract_zip_members(zipf, required, tmp_dir)

            if needs_reflectance:
                detector = _read_netcdf_variable(
                    paths["indices_an.nc"], "detector_an"
                ).astype(np.float32)
                valid_detector = np.isfinite(detector) & (detector >= 0)
                safe_detector = np.clip(
                    np.nan_to_num(detector, nan=0.0).astype(np.int64), 0, None
                )
                image_x = _read_netcdf_variable(paths["cartesian_an.nc"], "x_an")
                image_y = _read_netcdf_variable(paths["cartesian_an.nc"], "y_an")
                solar_zenith = _interpolate_tie_points_xy(
                    _read_netcdf_variable(paths["geometry_tn.nc"], "solar_zenith_tn"),
                    _read_netcdf_variable(paths["cartesian_tx.nc"], "x_tx"),
                    _read_netcdf_variable(paths["cartesian_tx.nc"], "y_tx"),
                    image_x,
                    image_y,
                )
                cos_solar_zenith = np.clip(
                    np.cos(np.deg2rad(solar_zenith)), 0.01, None
                ).astype(np.float32)
                reflectance_arrays = []
                for band_name in self.reflectance_bands:
                    band = band_name.split("_")[0]
                    radiance_name = f"{band}_radiance_an"
                    irradiance_name = f"{band}_solar_irradiance_an"
                    radiance = _read_netcdf_variable(
                        paths[f"{radiance_name}.nc"], radiance_name
                    ).astype(np.float32)
                    detector_irradiance = _read_netcdf_variable(
                        paths[f"{band}_quality_an.nc"], irradiance_name
                    ).astype(np.float32)
                    band_detector = np.clip(
                        safe_detector, 0, detector_irradiance.size - 1
                    )
                    reflectance = _radiance_to_reflectance(
                        radiance,
                        detector_irradiance.reshape(-1)[band_detector],
                        cos_solar_zenith,
                    )
                    reflectance_arrays.append(
                        np.where(valid_detector, reflectance, np.nan)
                    )
                reflectance_latitude = _read_netcdf_variable(
                    paths["geodetic_an.nc"], "latitude_an"
                ).astype(np.float32)
                reflectance_longitude = _read_netcdf_variable(
                    paths["geodetic_an.nc"], "longitude_an"
                ).astype(np.float32)

            if needs_bt:
                bt_arrays = []
                for band in self.bt_bands:
                    variable_name = f"{band}_in"
                    bt_arrays.append(
                        _read_netcdf_variable(
                            paths[f"{variable_name}.nc"], variable_name
                        ).astype(np.float32)
                    )
                bt_latitude = _read_netcdf_variable(
                    paths["geodetic_in.nc"], "latitude_in"
                ).astype(np.float32)
                bt_longitude = _read_netcdf_variable(
                    paths["geodetic_in.nc"], "longitude_in"
                ).astype(np.float32)

        if needs_reflectance:
            data, longitude, latitude = _crop_swath(
                np.stack(reflectance_arrays),
                reflectance_longitude,
                reflectance_latitude,
                geometries,
                self.swath_padding,
            )
            _write_swath(
                tile_store,
                item,
                self.reflectance_bands,
                data,
                longitude,
                latitude,
                self.reflectance_grid_resolution,
                self.nodata_value,
            )

        if needs_bt:
            data, longitude, latitude = _crop_swath(
                np.stack(bt_arrays),
                bt_longitude,
                bt_latitude,
                geometries,
                self.swath_padding,
            )
            _write_swath(
                tile_store,
                item,
                self.bt_bands,
                data,
                longitude,
                latitude,
                self.bt_grid_resolution,
                self.nodata_value,
            )
