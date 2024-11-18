"""Classes for storing configuration of a dataset."""

from datetime import timedelta
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
import pytimeparse
import torch
from rasterio.enums import Resampling

from rslearn.utils import PixelBounds, Projection


class DType(Enum):
    """Data type of a raster."""

    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    INT32 = "int32"
    FLOAT32 = "float32"

    def get_numpy_dtype(self) -> npt.DTypeLike:
        """Returns numpy dtype object corresponding to this DType."""
        if self == DType.UINT8:
            return np.uint8
        elif self == DType.UINT16:
            return np.uint16
        elif self == DType.UINT32:
            return np.uint32
        elif self == DType.INT32:
            return np.int32
        elif self == DType.FLOAT32:
            return np.float32
        raise ValueError(f"unable to handle numpy dtype {self}")

    def get_torch_dtype(self) -> torch.dtype:
        """Returns pytorch dtype object corresponding to this DType."""
        if self == DType.INT32:
            return torch.int32
        elif self == DType.FLOAT32:
            return torch.float32
        else:
            raise ValueError(f"unable to handle torch dtype {self}")


RESAMPLING_METHODS = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "cubic_spline": Resampling.cubic_spline,
}


class RasterFormatConfig:
    """A configuration specifying a RasterFormat."""

    def __init__(self, name: str, config_dict: dict[str, Any]) -> None:
        """Initialize a new RasterFormatConfig.

        Args:
            name: the name of the RasterFormat to use.
            config_dict: configuration to pass to the RasterFormat.
        """
        self.name = name
        self.config_dict = config_dict

    @staticmethod
    def from_config(config: dict[str, Any]) -> "RasterFormatConfig":
        """Create a RasterFormatConfig from config dict.

        Args:
            config: the config dict for this RasterFormatConfig
        """
        return RasterFormatConfig(name=config["name"], config_dict=config)


class VectorFormatConfig:
    """A configuration specifying a VectorFormat."""

    def __init__(self, name: str, config_dict: dict[str, Any] = {}) -> None:
        """Initialize a new VectorFormatConfig.

        Args:
            name: the name of the VectorFormat to use.
            config_dict: configuration to pass to the VectorFormat.
        """
        self.name = name
        self.config_dict = config_dict

    @staticmethod
    def from_config(config: dict[str, Any]) -> "VectorFormatConfig":
        """Create a VectorFormatConfig from config dict.

        Args:
            config: the config dict for this VectorFormatConfig
        """
        return VectorFormatConfig(name=config["name"], config_dict=config)


class BandSetConfig:
    """A configuration for a band set in a raster layer.

    Each band set specifies one or more bands that should be stored together.
    It also specifies the storage format and dtype, the zoom offset, etc. for these
    bands.
    """

    def __init__(
        self,
        config_dict: dict[str, Any],
        dtype: DType,
        bands: list[str],
        format: dict[str, Any] | None = None,
        zoom_offset: int = 0,
        remap: dict[str, Any] | None = None,
    ) -> None:
        """Creates a new BandSetConfig instance.

        Args:
            config_dict: the config dict used to configure this BandSetConfig
            dtype: the pixel value type to store tiles in
            bands: list of band names in this BandSetConfig
            format: the format to store tiles in, defaults to geotiff
            zoom_offset: non-negative integer, store images at window resolution
                divided by 2^(zoom_offset).
            remap: config dict for Remapper to remap pixel values
        """
        self.config_dict = config_dict
        self.bands = bands
        self.dtype = dtype
        self.zoom_offset = zoom_offset
        self.remap = remap

        if format is None:
            self.format = {"name": "geotiff"}
        else:
            self.format = format

    def serialize(self) -> dict[str, Any]:
        """Serialize this BandSetConfig to a config dict, currently unused."""
        return {
            "bands": self.bands,
            "format": self.format,
            "dtype": self.dtype,
            "zoom_offset": self.zoom_offset,
            "remap": self.remap,
        }

    @staticmethod
    def from_config(config: dict[str, Any]) -> "BandSetConfig":
        """Create a BandSetConfig from config dict.

        Args:
            config: the config dict for this BandSetConfig
        """
        kwargs = dict(
            config_dict=config,
            dtype=DType(config["dtype"]),
            bands=config["bands"],
        )
        for k in ["format", "zoom_offset", "remap"]:
            if k in config:
                kwargs[k] = config[k]
        return BandSetConfig(**kwargs)  # type: ignore

    def get_final_projection_and_bounds(
        self, projection: Projection, bounds: PixelBounds | None
    ) -> tuple[Projection, PixelBounds | None]:
        """Gets the final projection/bounds based on band set config.

        The band set config may apply a non-zero zoom offset that modifies the window's
        projection.

        Args:
            projection: the window's projection
            bounds: the window's bounds (optional)
            band_set: band set configuration object

        Returns:
            tuple of updated projection and bounds with zoom offset applied
        """
        if self.zoom_offset == 0:
            return projection, bounds
        projection = Projection(
            projection.crs,
            projection.x_resolution / (2**self.zoom_offset),
            projection.y_resolution / (2**self.zoom_offset),
        )
        if bounds is not None:
            if self.zoom_offset > 0:
                zoom_factor = 2**self.zoom_offset
                bounds = tuple(x * zoom_factor for x in bounds)  # type: ignore
            else:
                bounds = tuple(
                    x // (2 ** (-self.zoom_offset))
                    for x in bounds  # type: ignore
                )
        return projection, bounds


class SpaceMode(Enum):
    """Spatial matching mode when looking up items corresponding to a window."""

    CONTAINS = 1
    """Items must contain the entire window."""

    INTERSECTS = 2
    """Items must overlap any portion of the window."""

    MOSAIC = 3
    """Groups of items should be computed that cover the entire window.

    During materialization, items in each group are merged to form a mosaic in the
    dataset.
    """


class TimeMode(Enum):
    """Temporal  matching mode when looking up items corresponding to a window."""

    WITHIN = 1
    """Items must be within the window time range."""

    NEAREST = 2
    """Select items closest to the window time range, up to max_matches."""

    BEFORE = 3
    """Select items before the start of the window time range, up to max_matches."""

    AFTER = 4
    """Select items after the end of the window time range, up to max_matches."""


class QueryConfig:
    """A configuration for querying items in a data source."""

    def __init__(
        self,
        space_mode: SpaceMode = SpaceMode.MOSAIC,
        time_mode: TimeMode = TimeMode.WITHIN,
        max_matches: int = 1,
    ):
        """Creates a new query configuration.

        The provided options determine how a DataSource should lookup items that match a
        spatiotemporal window.

        Args:
            space_mode: specifies how items should be matched with windows spatially
            time_mode: specifies how items should be matched with windows temporally
            max_matches: the maximum number of items (or groups of items, if space_mode
                is MOSAIC) to match
        """
        self.space_mode = space_mode
        self.time_mode = time_mode
        self.max_matches = max_matches

    def serialize(self) -> dict[str, Any]:
        """Serialize this QueryConfig to a config dict, currently unused."""
        return {
            "space_mode": str(self.space_mode),
            "time_mode": str(self.time_mode),
            "max_matches": self.max_matches,
        }

    @staticmethod
    def from_config(config: dict[str, Any]) -> "QueryConfig":
        """Create a QueryConfig from config dict.

        Args:
            config: the config dict for this QueryConfig
        """
        return QueryConfig(
            space_mode=SpaceMode[config.get("space_mode", "MOSAIC")],
            time_mode=TimeMode[config.get("time_mode", "WITHIN")],
            max_matches=config.get("max_matches", 1),
        )


class DataSourceConfig:
    """Configuration for a DataSource in a dataset layer."""

    def __init__(
        self,
        name: str,
        query_config: QueryConfig,
        config_dict: dict[str, Any],
        time_offset: timedelta | None = None,
        duration: timedelta | None = None,
        ingest: bool = True,
    ) -> None:
        """Initializes a new DataSourceConfig.

        Args:
            name: the data source class name
            query_config: the QueryConfig specifying how to match items with windows
            config_dict: additional config passed to initialize the DataSource
            time_offset: optional, add this timedelta to the window's time range before
                matching
            duration: optional, if window's time range is (t0, t1), then update to
                (t0, t0 + duration)
            ingest: whether to ingest this layer or directly materialize it
                (default true)
        """
        self.name = name
        self.query_config = query_config
        self.config_dict = config_dict
        self.time_offset = time_offset
        self.duration = duration
        self.ingest = ingest

    def serialize(self) -> dict[str, Any]:
        """Serialize this DataSourceConfig to a config dict, currently unused."""
        config_dict = self.config_dict.copy()
        config_dict["name"] = self.name
        config_dict["query_config"] = self.query_config.serialize()
        config_dict["ingest"] = self.ingest
        if self.time_offset:
            config_dict["time_offset"] = str(self.time_offset)
        if self.duration:
            config_dict["duration"] = str(self.duration)
        return config_dict

    @staticmethod
    def from_config(config: dict[str, Any]) -> "DataSourceConfig":
        """Create a DataSourceConfig from config dict.

        Args:
            config: the config dict for this DataSourceConfig
        """
        kwargs = dict(
            name=config["name"],
            query_config=QueryConfig.from_config(config.get("query_config", {})),
            config_dict=config,
        )
        if "time_offset" in config:
            kwargs["time_offset"] = timedelta(
                seconds=pytimeparse.parse(config["time_offset"])
            )
        if "duration" in config:
            kwargs["duration"] = timedelta(
                seconds=pytimeparse.parse(config["duration"])
            )
        if "ingest" in config:
            kwargs["ingest"] = config["ingest"]
        return DataSourceConfig(**kwargs)


class LayerType(Enum):
    """The layer type (raster or vector)."""

    RASTER = "raster"
    VECTOR = "vector"


class LayerConfig:
    """Configuration of a layer in a dataset."""

    def __init__(
        self,
        layer_type: LayerType,
        data_source: DataSourceConfig | None = None,
        alias: str | None = None,
    ):
        """Initialize a new LayerConfig.

        Args:
            layer_type: the LayerType (raster or vector)
            data_source: optional DataSourceConfig if this layer is retrievable
            alias: alias for this layer to use in the tile store
        """
        self.layer_type = layer_type
        self.data_source = data_source
        self.alias = alias

    def serialize(self) -> dict[str, Any]:
        """Serialize this LayerConfig to a config dict, currently unused."""
        return {
            "layer_type": str(self.layer_type),
            "data_source": self.data_source,
            "alias": self.alias,
        }


class RasterLayerConfig(LayerConfig):
    """Configuration of a raster layer."""

    def __init__(
        self,
        layer_type: LayerType,
        band_sets: list[BandSetConfig],
        data_source: DataSourceConfig | None = None,
        resampling_method: Resampling = Resampling.bilinear,
        alias: str | None = None,
    ):
        """Initialize a new RasterLayerConfig.

        Args:
            layer_type: the LayerType (must be raster)
            band_sets: the bands to store in this layer
            data_source: optional DataSourceConfig if this layer is retrievable
            resampling_method: how to resample rasters (if needed), default bilinear resampling
            alias: alias for this layer to use in the tile store
        """
        super().__init__(layer_type, data_source, alias)
        self.band_sets = band_sets
        self.resampling_method = resampling_method

    @staticmethod
    def from_config(config: dict[str, Any]) -> "RasterLayerConfig":
        """Create a RasterLayerConfig from config dict.

        Args:
            config: the config dict for this RasterLayerConfig
        """
        kwargs = {
            "layer_type": LayerType(config["type"]),
            "band_sets": [BandSetConfig.from_config(el) for el in config["band_sets"]],
        }
        if "data_source" in config:
            kwargs["data_source"] = DataSourceConfig.from_config(config["data_source"])
        if "resampling_method" in config:
            kwargs["resampling_method"] = RESAMPLING_METHODS[
                config["resampling_method"]
            ]
        if "alias" in config:
            kwargs["alias"] = config["alias"]
        return RasterLayerConfig(**kwargs)  # type: ignore


class VectorLayerConfig(LayerConfig):
    """Configuration of a vector layer."""

    def __init__(
        self,
        layer_type: LayerType,
        data_source: DataSourceConfig | None = None,
        zoom_offset: int = 0,
        format: VectorFormatConfig = VectorFormatConfig("geojson"),
        alias: str | None = None,
    ):
        """Initialize a new VectorLayerConfig.

        Args:
            layer_type: the LayerType (must be vector)
            data_source: optional DataSourceConfig if this layer is retrievable
            zoom_offset: zoom offset at which to store the vector data
            format: the VectorFormatConfig, default storing as GeoJSON
            alias: alias for this layer to use in the tile store
        """
        super().__init__(layer_type, data_source, alias)
        self.zoom_offset = zoom_offset
        self.format = format

    @staticmethod
    def from_config(config: dict[str, Any]) -> "VectorLayerConfig":
        """Create a VectorLayerConfig from config dict.

        Args:
            config: the config dict for this VectorLayerConfig
        """
        kwargs: dict[str, Any] = {"layer_type": LayerType(config["type"])}
        if "data_source" in config:
            kwargs["data_source"] = DataSourceConfig.from_config(config["data_source"])
        if "zoom_offset" in config:
            kwargs["zoom_offset"] = config["zoom_offset"]
        if "format" in config:
            kwargs["format"] = VectorFormatConfig.from_config(config["format"])
        if "alias" in config:
            kwargs["alias"] = config["alias"]
        return VectorLayerConfig(**kwargs)  # type: ignore

    def get_final_projection_and_bounds(
        self, projection: Projection, bounds: PixelBounds | None
    ) -> tuple[Projection, PixelBounds | None]:
        """Gets the final projection/bounds based on zoom offset.

        Args:
            projection: the window's projection
            bounds: the window's bounds (optional)

        Returns:
            tuple of updated projection and bounds with zoom offset applied
        """
        if self.zoom_offset == 0:
            return projection, bounds
        projection = Projection(
            projection.crs,
            projection.x_resolution / (2**self.zoom_offset),
            projection.y_resolution / (2**self.zoom_offset),
        )
        if bounds:
            if self.zoom_offset > 0:
                bounds = tuple(x * (2**self.zoom_offset) for x in bounds)  # type: ignore
            else:
                bounds = tuple(
                    x // (2 ** (-self.zoom_offset))
                    for x in bounds  # type: ignore
                )
        return projection, bounds


def load_layer_config(config: dict[str, Any]) -> LayerConfig:
    """Load a LayerConfig from a config dict."""
    layer_type = LayerType(config.get("type"))
    if layer_type == LayerType.RASTER:
        return RasterLayerConfig.from_config(config)
    elif layer_type == LayerType.VECTOR:
        return VectorLayerConfig.from_config(config)
    raise ValueError(f"Unknown layer type {layer_type}")


class TileStoreConfig:
    """A configuration specifying a TileStore."""

    def __init__(self, name: str, config_dict: dict[str, Any]) -> None:
        """Create a new TileStoreConfig.

        Args:
            name: the tile store implementation name to use
            config_dict: configuration options
        """
        self.name = name
        self.config_dict = config_dict

    @staticmethod
    def from_config(config: dict[str, Any]) -> "TileStoreConfig":
        """Create a TileStoreConfig from config dict.

        Args:
            config: the config dict for this TileStoreConfig
        """
        return TileStoreConfig(name=config["name"], config_dict=config)
