"""Data source for reading spatio-temporal cubes stored in Zarr."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt
import shapely
from rasterio.crs import CRS
from rasterio.enums import Resampling

from rslearn.config import LayerConfig, QueryConfig, RasterLayerConfig
from rslearn.dataset import Window
from rslearn.dataset.materialize import RasterMaterializer
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStore, TileStoreWithLayer
from rslearn.utils.grid_index import GridIndex
from rslearn.utils.geometry import Projection, STGeometry, shp_intersects
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from .data_source import DataSource, Item
from .utils import match_candidate_items_to_window

logger = get_logger(__name__)


def _import_zarr_deps() -> tuple[Any, Any]:
    """Import dependencies required for interacting with Zarr stores."""

    try:
        import xarray as xr  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "ZarrDataSource requires xarray; install rslearn with the 'extra' extra"
        ) from exc

    try:
        import zarr  # noqa: F401  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "ZarrDataSource requires zarr; install rslearn with the 'extra' extra"
        ) from exc

    return xr, None


def _to_datetime(value: Any) -> datetime:
    """Convert various time coordinate values emitted by xarray to datetime."""

    if isinstance(value, datetime):
        return value

    try:
        import cftime  # type: ignore

        if isinstance(value, cftime.datetime):
            return value.to_datetime()
    except ImportError:  # pragma: no cover - optional dependency
        pass

    if isinstance(value, np.datetime64):
        iso = np.datetime_as_string(value, timezone="UTC")
        if iso.endswith("Z"):
            iso = iso[:-1]
        return datetime.fromisoformat(iso)

    if isinstance(value, str):
        return datetime.fromisoformat(value)

    raise TypeError(f"Unsupported time coordinate type: {type(value)!r}")


def _compute_time_boundaries(times: list[datetime], fallback: timedelta) -> list[datetime]:
    """Compute interval boundaries for a monotonic list of timestamps."""

    if len(times) == 0:
        return []

    # Ensure timestamps are sorted to avoid negative durations.
    sorted_times = sorted(times)
    boundaries: list[datetime] = []

    if len(sorted_times) == 1:
        half = fallback / 2
        boundaries.append(sorted_times[0] - half)
        boundaries.append(sorted_times[0] + half)
        return boundaries

    for idx in range(len(sorted_times) + 1):
        if idx == 0:
            delta = sorted_times[1] - sorted_times[0]
            boundaries.append(sorted_times[0] - delta / 2)
        elif idx == len(sorted_times):
            delta = sorted_times[-1] - sorted_times[-2]
            boundaries.append(sorted_times[-1] + delta / 2)
        else:
            prev_time = sorted_times[idx - 1]
            next_time = sorted_times[idx]
            boundaries.append(prev_time + (next_time - prev_time) / 2)

    return boundaries


class ZarrItem(Item):
    """Represents a spatio-temporal chunk inside the Zarr data cube."""

    x_range: tuple[int, int]
    y_range: tuple[int, int]
    time_range_indexes: tuple[int, int] | None
    dim_slices: dict[str, tuple[int, int]]
    x_offset: int
    y_offset: int

    def __init__(
        self,
        name: str,
        geometry: STGeometry,
        x_range: tuple[int, int],
        y_range: tuple[int, int],
        time_range_indexes: tuple[int, int] | None,
        dim_slices: dict[str, tuple[int, int]],
        x_offset: int,
        y_offset: int,
    ) -> None:
        super().__init__(name, geometry)
        self.x_range = x_range
        self.y_range = y_range
        self.time_range_indexes = time_range_indexes
        self.dim_slices = dim_slices
        self.x_offset = x_offset
        self.y_offset = y_offset

    @property
    def pixel_bounds(self) -> tuple[int, int, int, int]:
        """Return bounds in pixel coordinates (x0, y0, x1, y1)."""

        return (
            self.x_offset + self.x_range[0],
            self.y_offset + self.y_range[0],
            self.x_offset + self.x_range[1],
            self.y_offset + self.y_range[1],
        )

    def serialize(self) -> dict:
        """Serialize this item for storing inside a window."""

        data = super().serialize()
        data.update(
            {
                "x_range": list(self.x_range),
                "y_range": list(self.y_range),
                "x_offset": self.x_offset,
                "y_offset": self.y_offset,
                "time_range_indexes": list(self.time_range_indexes)
                if self.time_range_indexes
                else None,
                "dim_slices": {
                    dim: list(range_pair) for dim, range_pair in self.dim_slices.items()
                },
            }
        )
        return data

    @staticmethod
    def deserialize(data: dict) -> "ZarrItem":
        """Deserialize a serialized ZarrItem."""

        base_item = Item.deserialize(data)
        dim_slices = {
            dim: (range_pair[0], range_pair[1]) for dim, range_pair in data["dim_slices"].items()
        }
        time_range_indexes = None
        if data["time_range_indexes"] is not None:
            time_range_indexes = (
                data["time_range_indexes"][0],
                data["time_range_indexes"][1],
            )
        return ZarrItem(
            name=base_item.name,
            geometry=base_item.geometry,
            x_range=(data["x_range"][0], data["x_range"][1]),
            y_range=(data["y_range"][0], data["y_range"][1]),
            time_range_indexes=time_range_indexes,
            dim_slices=dim_slices,
            x_offset=data.get("x_offset", 0),
            y_offset=data.get("y_offset", 0),
        )


class ZarrDataSource(DataSource[ZarrItem], TileStore):
    """DataSource for reading raster cubes stored in a Zarr hierarchy."""

    DEFAULT_SINGLE_TIME_INTERVAL = timedelta(hours=1)

    def __init__(
        self,
        *,
        store_uri: str,
        data_variable: str,
        projection: Projection,
        band_names: list[str],
        dtype: np.dtype,
        nodata: float | int | None,
        axis_names: dict[str, str],
        items: list[ZarrItem],
        time_boundaries: list[datetime] | None,
        storage_options: dict[str, Any],
        consolidated: bool,
    ) -> None:
        self.store_uri = store_uri
        self.data_variable = data_variable
        self.projection = projection
        self.band_names = band_names
        self.dtype = dtype
        self.nodata = nodata
        self.axis_names = axis_names
        self._items = items
        self.storage_options = storage_options
        self.consolidated = consolidated
        self._time_boundaries = time_boundaries

        self._item_by_name = {item.name: item for item in self._items}
        grid_size = 1
        for item in self._items:
            width = item.x_range[1] - item.x_range[0]
            height = item.y_range[1] - item.y_range[0]
            grid_size = max(grid_size, min(width, height))

        self._spatial_index = GridIndex(size=grid_size)
        for item in self._items:
            self._spatial_index.insert(item.geometry.shp.bounds, item)

        self._data_array = None

    @staticmethod
    def from_config(config: LayerConfig, ds_path: Any) -> "ZarrDataSource":
        """Create a ZarrDataSource from a RasterLayer configuration."""

        if not isinstance(config, RasterLayerConfig):
            raise ValueError("ZarrDataSource requires a raster layer")
        if config.data_source is None:
            raise ValueError("data_source configuration required for ZarrDataSource")

        cfg = config.data_source.config_dict

        required_keys = [
            "store_uri",
            "axis_names",
            "pixel_size",
            "origin",
            "bands",
            "crs",
            "dtype",
        ]
        for key in required_keys:
            if key not in cfg:
                raise ValueError(f"Missing required Zarr data source config key: {key}")

        axis_names = cfg["axis_names"]
        x_dim = axis_names.get("x")
        y_dim = axis_names.get("y")
        if not x_dim or not y_dim:
            raise ValueError("axis_names must map 'x' and 'y' dimensions")

        time_dim = axis_names.get("time")
        band_dim = axis_names.get("band")

        pixel_size = cfg["pixel_size"]
        if isinstance(pixel_size, dict):
            x_resolution = float(pixel_size.get("x"))
            y_resolution = -float(pixel_size.get("y"))
        else:
            x_resolution = float(pixel_size)
            y_resolution = -float(pixel_size)

        origin = cfg["origin"]
        if not isinstance(origin, (list, tuple)) or len(origin) != 2:
            raise ValueError("origin must be a two element array [x_min, y_max]")

        projection = Projection(CRS.from_string(cfg["crs"]), x_resolution, y_resolution)

        band_names = cfg["bands"]
        if not isinstance(band_names, list) or len(band_names) == 0:
            raise ValueError("bands must be a non-empty list")

        dtype = np.dtype(cfg["dtype"])
        nodata = cfg.get("nodata")

        data_variable = cfg.get("data_variable")
        storage_options = cfg.get("storage_options")
        consolidated = cfg.get("consolidated", True)

        xr, _ = _import_zarr_deps()
        open_kwargs = dict(consolidated=consolidated)
        if storage_options is not None:
            open_kwargs["storage_options"] = storage_options
        dataset = xr.open_zarr(cfg["store_uri"], **open_kwargs)
        if data_variable is None:
            if len(dataset.data_vars) != 1:
                raise ValueError(
                    "data_variable must be specified when Zarr store has multiple variables"
                )
            data_variable = next(iter(dataset.data_vars))
        data_array = dataset[data_variable]

        time_chunk_size = cfg.get("time_chunk_size", 1)
        chunk_shape_cfg = cfg.get("chunk_shape", {})

        if band_dim and band_dim in data_array.dims:
            band_size = int(data_array.sizes[band_dim])
            if band_size != len(band_names):
                raise ValueError(
                    "Configured bands do not match Zarr band dimension size"
                )
        elif len(band_names) != 1:
            raise ValueError(
                "Zarr data without an explicit band dimension must configure exactly one band"
            )

        def _dimension_chunk(dim: str, default: int) -> int:
            if dim in chunk_shape_cfg:
                return int(chunk_shape_cfg[dim])
            if getattr(data_array, "chunks", None) and data_array.chunksizes.get(dim):
                return int(data_array.chunksizes[dim][0])
            return default

        y_size = int(data_array.sizes[y_dim])
        x_size = int(data_array.sizes[x_dim])
        y_chunk = max(1, min(_dimension_chunk(y_dim, y_size), y_size))
        x_chunk = max(1, min(_dimension_chunk(x_dim, x_size), x_size))

        time_boundaries: list[datetime] | None = None
        time_indexes: Iterable[int] = [0]
        if time_dim in data_array.dims:
            time_size = int(data_array.sizes[time_dim])
            if time_chunk_size != 1:
                logger.warning(
                    "ZarrDataSource currently treats one time-step per item; "
                    "ignoring time_chunk_size=%s", time_chunk_size
                )
            times = [_to_datetime(value) for value in data_array[time_dim].values]
            fallback = ZarrDataSource.DEFAULT_SINGLE_TIME_INTERVAL
            if len(times) >= 2:
                first_delta = times[1] - times[0]
                if first_delta.total_seconds() > 0:
                    fallback = first_delta
            time_boundaries = _compute_time_boundaries(times, fallback)
            time_indexes = range(time_size)
        else:
            time_dim = None

        items: list[ZarrItem] = []
        origin_x, origin_y = float(origin[0]), float(origin[1])
        x_offset = int(round(origin_x / x_resolution))
        y_offset = int(round(origin_y / y_resolution))

        x_ranges = [
            (x_start, min(x_start + x_chunk, x_size))
            for x_start in range(0, x_size, x_chunk)
        ]
        y_ranges = [
            (y_start, min(y_start + y_chunk, y_size))
            for y_start in range(0, y_size, y_chunk)
        ]

        for t_index in time_indexes:
            for y_range in y_ranges:
                for x_range in x_ranges:
                    if time_dim is None:
                        time_range = None
                        time_index_range = None
                    else:
                        assert time_boundaries is not None
                        start = time_boundaries[t_index]
                        end = time_boundaries[t_index + 1]
                        time_range = (start, end)
                        time_index_range = (t_index, t_index + 1)

                    geometry = STGeometry(
                        projection,
                        shapely.box(
                            x_offset + x_range[0],
                            y_offset + y_range[0],
                            x_offset + x_range[1],
                            y_offset + y_range[1],
                        ),
                        time_range,
                    )
                    name = f"t{t_index}_y{y_range[0]}_{y_range[1]}_x{x_range[0]}_{x_range[1]}"
                    dim_slices = {
                        y_dim: y_range,
                        x_dim: x_range,
                    }
                    if time_dim:
                        dim_slices[time_dim] = time_index_range
                    if band_dim and band_dim in data_array.dims:
                        dim_slices[band_dim] = (0, len(band_names))

                    items.append(
                        ZarrItem(
                            name=name,
                            geometry=geometry,
                            x_range=x_range,
                            y_range=y_range,
                            time_range_indexes=time_index_range,
                            dim_slices=dim_slices,
                            x_offset=x_offset,
                            y_offset=y_offset,
                        )
                    )

        return ZarrDataSource(
            store_uri=cfg["store_uri"],
            data_variable=data_variable,
            projection=projection,
            band_names=band_names,
            dtype=dtype,
            nodata=nodata,
            axis_names={
                "x": x_dim,
                "y": y_dim,
                "time": time_dim,
                "band": band_dim,
            },
            items=items,
            time_boundaries=time_boundaries,
            storage_options=storage_options,
            consolidated=consolidated,
        )

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def _load_data_array(self):
        if self._data_array is not None:
            return self._data_array

        xr, _ = _import_zarr_deps()
        dataset = xr.open_zarr(
            self.store_uri,
            consolidated=self.consolidated,
            storage_options=self.storage_options,
        )
        data_array = dataset[self.data_variable]
        self._data_array = data_array
        return data_array

    def _iter_candidate_items(self, geometry: STGeometry) -> list[ZarrItem]:
        geometry_in_projection = geometry.to_projection(self.projection)
        candidates = self._spatial_index.query(geometry_in_projection.shp.bounds)
        filtered: list[ZarrItem] = []
        for item in candidates:
            item_geom = item.geometry
            if not shp_intersects(item_geom.shp, geometry_in_projection.shp):
                continue
            if geometry.time_range and not geometry.intersects_time_range(
                item_geom.time_range
            ):
                continue
            filtered.append(item)
        return filtered

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[ZarrItem]]]:
        groups: list[list[list[ZarrItem]]] = []
        for geometry in geometries:
            candidates = self._iter_candidate_items(geometry)
            geometry_in_projection = geometry.to_projection(self.projection)
            cur_groups = match_candidate_items_to_window(
                geometry_in_projection,
                candidates,
                query_config,
            )
            groups.append(cur_groups)
        return groups

    def deserialize_item(self, serialized_item: Any) -> ZarrItem:
        return ZarrItem.deserialize(serialized_item)

    def _read_item_array(self, item: ZarrItem) -> npt.NDArray[Any]:
        data_array = self._load_data_array()
        indexers: dict[str, slice] = {}
        for dim, range_pair in item.dim_slices.items():
            indexers[dim] = slice(range_pair[0], range_pair[1])

        selected = data_array.isel(**indexers)

        time_dim = self.axis_names.get("time")
        if time_dim and time_dim in selected.dims:
            selected = selected.squeeze(dim=time_dim, drop=True)

        band_dim = self.axis_names.get("band")
        if band_dim and band_dim in selected.dims:
            selected = selected.transpose(band_dim, self.axis_names["y"], self.axis_names["x"])
            array = selected.values
        else:
            selected = selected.transpose(self.axis_names["y"], self.axis_names["x"])
            array = selected.values[None, :, :]

        return np.asarray(array, dtype=self.dtype, order="C")

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[ZarrItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        for item in items:
            if tile_store.is_raster_ready(item.name, self.band_names):
                continue

            array = self._read_item_array(item)
            tile_store.write_raster(
                item.name,
                self.band_names,
                self.projection,
                item.pixel_bounds,
                array,
            )

    def materialize(
        self,
        window: Window,
        item_groups: list[list[ZarrItem]],
        layer_name: str,
        layer_cfg: LayerConfig,
    ) -> None:
        if not isinstance(layer_cfg, RasterLayerConfig):
            raise ValueError("ZarrDataSource only supports raster materialization")
        RasterMaterializer().materialize(
            TileStoreWithLayer(self, layer_name),
            window,
            layer_name,
            layer_cfg,
            item_groups,
        )

    # ------------------------------------------------------------------
    # TileStore interface (read-only)
    # ------------------------------------------------------------------

    def is_raster_ready(
        self, layer_name: str, item_name: str, bands: list[str]
    ) -> bool:
        if bands != self.band_names:
            return False
        return item_name in self._item_by_name

    def get_raster_bands(self, layer_name: str, item_name: str) -> list[list[str]]:
        if item_name not in self._item_by_name:
            return []
        return [self.band_names]

    def get_raster_bounds(
        self, layer_name: str, item_name: str, bands: list[str], projection: Projection
    ) -> tuple[int, int, int, int]:
        if item_name not in self._item_by_name:
            raise ValueError(f"Unknown item {item_name}")
        item = self._item_by_name[item_name]
        geom = item.geometry.to_projection(projection)
        bounds = geom.shp.bounds
        return (
            int(math.floor(bounds[0])),
            int(math.floor(bounds[1])),
            int(math.ceil(bounds[2])),
            int(math.ceil(bounds[3])),
        )

    def read_raster(
        self,
        layer_name: str,
        item_name: str,
        bands: list[str],
        projection: Projection,
        bounds: tuple[int, int, int, int],
        resampling: Resampling = Resampling.bilinear,
    ) -> npt.NDArray[Any]:
        if bands != self.band_names:
            raise ValueError(
                f"ZarrDataSource stores bands {self.band_names}, requested {bands}"
            )

        if item_name not in self._item_by_name:
            raise ValueError(f"Unknown item {item_name}")

        item = self._item_by_name[item_name]

        request_geometry = STGeometry(projection, shapely.box(*bounds), None)
        request_in_native = request_geometry.to_projection(self.projection)

        intersection = request_in_native.shp.intersection(item.geometry.shp)
        if intersection.is_empty:
            height = bounds[3] - bounds[1]
            width = bounds[2] - bounds[0]
            fill_value = self.nodata if self.nodata is not None else 0
            return np.full((len(self.band_names), height, width), fill_value, self.dtype)

        read_bounds = (
            math.floor(intersection.bounds[0]),
            math.floor(intersection.bounds[1]),
            math.ceil(intersection.bounds[2]),
            math.ceil(intersection.bounds[3]),
        )

        crop = self._read_item_array(item)
        x0 = read_bounds[0] - item.pixel_bounds[0]
        x1 = x0 + (read_bounds[2] - read_bounds[0])
        y0 = read_bounds[1] - item.pixel_bounds[1]
        y1 = y0 + (read_bounds[3] - read_bounds[1])
        crop = crop[:, y0:y1, x0:x1]

        if self.projection == projection and read_bounds == bounds:
            return crop

        src_transform = get_transform_from_projection_and_bounds(
            self.projection, read_bounds
        )
        dst_transform = get_transform_from_projection_and_bounds(projection, bounds)
        dst_array = np.full(
            (len(self.band_names), bounds[3] - bounds[1], bounds[2] - bounds[0]),
            self.nodata if self.nodata is not None else 0,
            dtype=self.dtype,
        )

        import rasterio.warp

        rasterio.warp.reproject(
            source=crop,
            src_crs=self.projection.crs,
            src_transform=src_transform,
            destination=dst_array,
            dst_crs=projection.crs,
            dst_transform=dst_transform,
            resampling=resampling,
            src_nodata=self.nodata,
            dst_nodata=self.nodata,
        )

        return dst_array


__all__ = ["ZarrDataSource", "ZarrItem"]
