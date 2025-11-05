"""Classes for storing configuration of a dataset."""

import json
from datetime import timedelta
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
import pytimeparse
from rasterio.enums import Resampling

from rslearn.utils import PixelBounds, Projection


class DType(Enum):
    """Data type of a raster."""

    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT32 = "float32"

    def get_numpy_dtype(self) -> npt.DTypeLike:
        """Returns numpy dtype object corresponding to this DType."""
        if self == DType.UINT8:
            return np.uint8
        elif self == DType.UINT16:
            return np.uint16
        elif self == DType.UINT32:
            return np.uint32
        elif self == DType.UINT64:
            return np.uint64
        elif self == DType.INT8:
            return np.int8
        elif self == DType.INT16:
            return np.int16
        elif self == DType.INT32:
            return np.int32
        elif self == DType.INT64:
            return np.int64
        elif self == DType.FLOAT32:
            return np.float32
        raise ValueError(f"unable to handle numpy dtype {self}")


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
        bands: list[str] | None = None,
        num_bands: int | None = None,
        format: dict[str, Any] | None = None,
        zoom_offset: int = 0,
        remap: dict[str, Any] | None = None,
        class_names: list[list[str]] | None = None,
        nodata_vals: list[float] | None = None,
    ) -> None:
        """Creates a new BandSetConfig instance.

        Args:
            config_dict: the config dict used to configure this BandSetConfig
            dtype: the pixel value type to store tiles in
            bands: list of band names in this BandSetConfig. One of bands or num_bands
                must be set.
            num_bands: the number of bands in this band set. The bands will be named
                B00, B01, B02, etc.
            format: the format to store tiles in, defaults to geotiff
            zoom_offset: store images at a resolution higher or lower than the window
                resolution. This enables keeping source data at its native resolution,
                either to save storage space (for lower resolution data) or to retain
                details (for higher resolution data). If positive, store data at the
                window resolution divided by 2^(zoom_offset) (higher resolution). If
                negative, store data at the window resolution multiplied by
                2^(-zoom_offset) (lower resolution).
            remap: config dict for Remapper to remap pixel values
            class_names: optional list of names for the different possible values of
                each band. The length of this list must equal the number of bands. For
                example, [["forest", "desert"]] means that it is a single-band raster
                where values can be 0 (forest) or 1 (desert).
            nodata_vals: the nodata values for this band set. This is used during
                materialization when creating mosaics, to determine which parts of the
                source images should be copied.
        """
        if (bands is None and num_bands is None) or (
            bands is not None and num_bands is not None
        ):
            raise ValueError("exactly one of bands and num_bands must be set")
        if bands is None:
            assert num_bands is not None
            bands = [f"B{idx}" for idx in range(num_bands)]

        if class_names is not None and len(bands) != len(class_names):
            raise ValueError(
                f"the number of class lists ({len(class_names)}) does not match the number of bands ({len(bands)})"
            )

        self.config_dict = config_dict
        self.bands = bands
        self.dtype = dtype
        self.zoom_offset = zoom_offset
        self.remap = remap
        self.class_names = class_names
        self.nodata_vals = nodata_vals

        if format is None:
            self.format = {"name": "geotiff"}
        else:
            self.format = format

    def serialize(self) -> dict[str, Any]:
        """Serialize this BandSetConfig to a config dict."""
        return self.config_dict

    @staticmethod
    def from_config(config: dict[str, Any]) -> "BandSetConfig":
        """Create a BandSetConfig from config dict.

        Args:
            config: the config dict for this BandSetConfig
        """
        kwargs = dict(
            config_dict=config,
            dtype=DType(config["dtype"]),
        )
        for k in [
            "bands",
            "num_bands",
            "format",
            "zoom_offset",
            "remap",
            "class_names",
            "nodata_vals",
        ]:
            if k in config:
                kwargs[k] = config[k]
        return BandSetConfig(**kwargs)  # type: ignore

    def get_final_projection_and_bounds(
        self, projection: Projection, bounds: PixelBounds
    ) -> tuple[Projection, PixelBounds]:
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

    PER_PERIOD_MOSAIC = 4
    """Create one mosaic per sub-period of the time range.

    The duration of the sub-periods is controlled by another option in QueryConfig.
    """

    COMPOSITE = 5
    """Creates one composite covering the entire window.

    During querying all items intersecting the window are placed in one group.
    The compositing_method in the rasterlayer config specifies how these items are reduced
    to a single item (e.g MEAN/MEDIAN/FIRST_VALID) during materialization.
    """

    # TODO add PER_PERIOD_COMPOSITE


class TimeMode(Enum):
    """Temporal  matching mode when looking up items corresponding to a window."""

    WITHIN = 1
    """Items must be within the window time range."""

    NEAREST = 2
    """Select items closest to the window time range, up to max_matches."""

    BEFORE = 3
    """Select items before the end of the window time range, up to max_matches."""

    AFTER = 4
    """Select items after the start of the window time range, up to max_matches."""


class QueryConfig:
    """A configuration for querying items in a data source."""

    def __init__(
        self,
        space_mode: SpaceMode = SpaceMode.MOSAIC,
        time_mode: TimeMode = TimeMode.WITHIN,
        min_matches: int = 0,
        max_matches: int = 1,
        period_duration: timedelta = timedelta(days=30),
    ):
        """Creates a new query configuration.

        The provided options determine how a DataSource should lookup items that match a
        spatiotemporal window.

        Args:
            space_mode: specifies how items should be matched with windows spatially
            time_mode: specifies how items should be matched with windows temporally
            min_matches: the minimum number of item groups. If there are fewer than
                this many matches, then no matches will be returned. This can be used
                to prevent unnecessary data ingestion if the user plans to discard
                windows that do not have a sufficient amount of data.
            max_matches: the maximum number of items (or groups of items, if space_mode
                is MOSAIC) to match
            period_duration: the duration of the periods, if the space mode is
                PER_PERIOD_MOSAIC.
        """
        self.space_mode = space_mode
        self.time_mode = time_mode
        self.min_matches = min_matches
        self.max_matches = max_matches
        self.period_duration = period_duration

    def serialize(self) -> dict[str, Any]:
        """Serialize this QueryConfig to a config dict."""
        return {
            "space_mode": str(self.space_mode),
            "time_mode": str(self.time_mode),
            "min_matches": self.min_matches,
            "max_matches": self.max_matches,
            "period_duration": f"{self.period_duration.total_seconds()}s",
        }

    @staticmethod
    def from_config(config: dict[str, Any]) -> "QueryConfig":
        """Create a QueryConfig from config dict.

        Args:
            config: the config dict for this QueryConfig
        """
        kwargs: dict[str, Any] = dict()
        if "space_mode" in config:
            kwargs["space_mode"] = SpaceMode[config["space_mode"]]
        if "time_mode" in config:
            kwargs["time_mode"] = TimeMode[config["time_mode"]]
        if "period_duration" in config:
            kwargs["period_duration"] = timedelta(
                seconds=pytimeparse.parse(config["period_duration"])
            )
        for k in ["min_matches", "max_matches"]:
            if k not in config:
                continue
            kwargs[k] = config[k]
        return QueryConfig(**kwargs)


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
        """Serialize this DataSourceConfig to a config dict."""
        return self.config_dict

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
        """Serialize this LayerConfig to a config dict."""
        return {
            "layer_type": str(self.layer_type),
            "data_source": self.data_source.serialize() if self.data_source else None,
            "alias": self.alias,
        }

    def __hash__(self) -> int:
        """Return a hash of this LayerConfig."""
        return hash(json.dumps(self.serialize(), sort_keys=True))

    def __eq__(self, other: Any) -> bool:
        """Returns whether other is the same as this LayerConfig.

        Args:
            other: the other object to compare.
        """
        if not isinstance(other, LayerConfig):
            return False
        return self.serialize() == other.serialize()


class CompositingMethod(Enum):
    """Method how to select pixels for the composite from corresponding items of a window."""

    FIRST_VALID = 1
    """Select first valid pixel in order of corresponding items (might be sorted)"""

    MEAN = 2
    """Select per-pixel mean value of corresponding items of a window"""

    MEDIAN = 3
    """Select per-pixel median value of corresponding items of a window"""


class RasterLayerConfig(LayerConfig):
    """Configuration of a raster layer."""

    def __init__(
        self,
        layer_type: LayerType,
        band_sets: list[BandSetConfig],
        data_source: DataSourceConfig | None = None,
        resampling_method: Resampling = Resampling.bilinear,
        alias: str | None = None,
        compositing_method: CompositingMethod = CompositingMethod.FIRST_VALID,
    ):
        """Initialize a new RasterLayerConfig.

        Args:
            layer_type: the LayerType (must be raster)
            band_sets: the bands to store in this layer
            data_source: optional DataSourceConfig if this layer is retrievable
            resampling_method: how to resample rasters (if needed), default bilinear resampling
            alias: alias for this layer to use in the tile store
            compositing_method: how to compute pixel values in the composite of each windows items
        """
        super().__init__(layer_type, data_source, alias)
        self.band_sets = band_sets
        self.resampling_method = resampling_method
        self.compositing_method = compositing_method

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
        if "compositing_method" in config:
            kwargs["compositing_method"] = CompositingMethod[
                config["compositing_method"]
            ]
        return RasterLayerConfig(**kwargs)  # type: ignore


class VectorLayerConfig(LayerConfig):
    """Configuration of a vector layer."""

    def __init__(
        self,
        layer_type: LayerType,
        data_source: DataSourceConfig | None = None,
        format: VectorFormatConfig = VectorFormatConfig("geojson"),
        alias: str | None = None,
        class_property_name: str | None = None,
        class_names: list[str] | None = None,
    ):
        """Initialize a new VectorLayerConfig.

        Args:
            layer_type: the LayerType (must be vector)
            data_source: optional DataSourceConfig if this layer is retrievable
            format: the VectorFormatConfig, default storing as GeoJSON
            alias: alias for this layer to use in the tile store
            class_property_name: optional metadata field indicating that the GeoJSON
                features contain a property that corresponds to a class label, and this
                is the name of that property.
            class_names: the list of classes that the class_property_name property
                could be set to.
        """
        super().__init__(layer_type, data_source, alias)
        self.format = format
        self.class_property_name = class_property_name
        self.class_names = class_names

    @staticmethod
    def from_config(config: dict[str, Any]) -> "VectorLayerConfig":
        """Create a VectorLayerConfig from config dict.

        Args:
            config: the config dict for this VectorLayerConfig
        """
        kwargs: dict[str, Any] = {"layer_type": LayerType(config["type"])}
        if "data_source" in config:
            kwargs["data_source"] = DataSourceConfig.from_config(config["data_source"])
        if "format" in config:
            kwargs["format"] = VectorFormatConfig.from_config(config["format"])

        simple_optionals = [
            "alias",
            "class_property_name",
            "class_names",
        ]
        for k in simple_optionals:
            if k in config:
                kwargs[k] = config[k]

        # The "zoom_offset" option was removed.
        # We should change how we create configuration so we can error on all
        # non-existing config options, but for now we make sure to raise error if
        # zoom_offset is set since it is no longer supported.
        if "zoom_offset" in config:
            raise ValueError("unsupported zoom_offset option in vector layer config")

        return VectorLayerConfig(**kwargs)  # type: ignore


def load_layer_config(config: dict[str, Any]) -> LayerConfig:
    """Load a LayerConfig from a config dict."""
    layer_type = LayerType(config.get("type"))
    if layer_type == LayerType.RASTER:
        return RasterLayerConfig.from_config(config)
    elif layer_type == LayerType.VECTOR:
        return VectorLayerConfig.from_config(config)
    raise ValueError(f"Unknown layer type {layer_type}")
