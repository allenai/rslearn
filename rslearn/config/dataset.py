from datetime import timedelta
from enum import Enum
from typing import Any, Optional

import pytimeparse
from rasterio.enums import Resampling

from rslearn.utils import PixelBounds, Projection


class DType(Enum):
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    FLOAT32 = "float32"


RESAMPLING_METHODS = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "cubic_spline": Resampling.cubic_spline,
}


class RasterFormatConfig:
    """A configuration specifying a RasterFormat."""

    def __init__(self, name: str, config_dict: dict[str, Any]) -> None:
        self.name = name
        self.config_dict = config_dict

    @staticmethod
    def from_config(config: dict[str, Any]) -> "RasterFormatConfig":
        return RasterFormatConfig(
            name=config["name"],
            config_dict=config,
        )


class VectorFormatConfig:
    """A configuration specifying a VectorFormat."""

    def __init__(self, name: str, config_dict: dict[str, Any] = {}) -> None:
        self.name = name
        self.config_dict = config_dict

    @staticmethod
    def from_config(config: dict[str, Any]) -> "VectorFormatConfig":
        return VectorFormatConfig(
            name=config["name"],
            config_dict=config,
        )


class BandSetConfig:
    def __init__(
        self,
        config_dict: dict[str, Any],
        dtype: DType,
        bands: Optional[list[str]] = None,
        format: Optional[dict[str, Any]] = None,
        zoom_offset: int = 0,
        remap_config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Creates a new BandSetConfig instance.

        Args:
            config_dict: the config dict used to configure this BandSetConfig
            band_sets: a list of band sets, each of which is a list of band names that
                should be stored together
            format: the format to store tiles in, defaults to geotiff
            dtype: the pixel value type to store tiles in
            zoom_offset: non-negative integer, store images at window resolution
                divided by 2^(zoom_offset).
        """
        self.config_dict = config_dict
        self.bands = bands
        self.format = format
        self.dtype = dtype
        self.zoom_offset = zoom_offset
        self.remap_config = remap_config

        if not self.format:
            self.format = {
                "name": "geotiff",
            }

    def serialize(self) -> dict[str, Any]:
        return {
            "bands": self.bands,
            "format": self.format,
            "dtype": self.dtype,
            "zoom_offset": self.zoom_offset,
            "remap": self.remap_config,
        }

    @staticmethod
    def from_config(config: dict[str, Any]) -> "BandSetConfig":
        """Creates a new BandSetConfig instance from a configuration dictionary."""
        return BandSetConfig(
            config_dict=config,
            dtype=DType(config["dtype"]),
            bands=config.get("bands"),
            format=config.get("format", "geotiff"),
            zoom_offset=config.get("zoom_offset", 0),
            remap_config=config.get("remap"),
        )

    def get_final_projection_and_bounds(
        self,
        projection: Projection,
        bounds: Optional[PixelBounds],
    ) -> tuple[Projection, Optional[PixelBounds]]:
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
        if bounds:
            if self.zoom_offset > 0:
                bounds = tuple(x * (2**self.zoom_offset) for x in bounds)
            else:
                bounds = tuple(x // (2 ** (-self.zoom_offset)) for x in bounds)
        return projection, bounds


class SpaceMode(Enum):
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
        return {
            "space_mode": str(self.space_mode),
            "time_mode": str(self.time_mode),
            "max_matches": self.max_matches,
        }

    @staticmethod
    def from_config(config: dict[str, Any]) -> "QueryConfig":
        return QueryConfig(
            space_mode=SpaceMode[config.get("space_mode", "MOSAIC")],
            time_mode=TimeMode[config.get("time_mode", "WITHIN")],
            max_matches=config.get("max_matches", 1),
        )


class DataSourceConfig:
    def __init__(
        self,
        name: str,
        query_config: QueryConfig,
        time_offset: Optional[timedelta],
        config_dict: dict[str, Any],
    ) -> None:
        self.name = name
        self.query_config = query_config
        self.time_offset = time_offset
        self.config_dict = config_dict

    def serialize(self) -> dict[str, Any]:
        config_dict = self.config_dict.copy()
        config_dict["name"] = self.__annotations__name
        config_dict["time_offset"] = str(self.time_offset)
        config_dict["query_config"] = self.query_config.serialize()
        return config_dict

    @staticmethod
    def from_config(config: dict[str, Any]) -> "DataSourceConfig":
        time_offset = None
        if "time_offset" in config:
            time_offset = timedelta(seconds=pytimeparse.parse(config["time_offset"]))
        return DataSourceConfig(
            name=config["name"],
            query_config=QueryConfig.from_config(config.get("query_config", {})),
            time_offset=time_offset,
            config_dict=config,
        )


class LayerType(Enum):
    RASTER = "raster"
    VECTOR = "vector"


class LayerConfig:
    def __init__(
        self, layer_type: LayerType, data_source: Optional[DataSourceConfig] = None
    ):
        self.layer_type = layer_type
        self.data_source = data_source

    def serialize(self) -> dict[str, Any]:
        return {
            "layer_type": str(self.layer_type),
            "data_source": self.data_source,
        }


class RasterLayerConfig(LayerConfig):
    def __init__(
        self,
        layer_type: LayerType,
        band_sets: list[BandSetConfig],
        data_source: Optional[DataSourceConfig] = None,
        resampling_method: Resampling = Resampling.bilinear,
    ):
        super().__init__(layer_type, data_source)
        self.band_sets = band_sets
        self.resampling_method = resampling_method

    @staticmethod
    def from_config(config: dict[str, Any]) -> "RasterLayerConfig":
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
        return RasterLayerConfig(**kwargs)


class VectorLayerConfig(LayerConfig):
    def __init__(
        self,
        layer_type: LayerType,
        data_source: Optional[DataSourceConfig] = None,
        zoom_offset: int = 0,
        format: VectorFormatConfig = VectorFormatConfig("geojson"),
    ):
        super().__init__(layer_type, data_source)
        self.zoom_offset = zoom_offset
        self.format = format

    @staticmethod
    def from_config(config: dict[str, Any]) -> "VectorLayerConfig":
        kwargs = {"layer_type": LayerType(config["type"])}
        if "data_source" in config:
            kwargs["data_source"] = DataSourceConfig.from_config(config["data_source"])
        if "zoom_offset" in config:
            kwargs["zoom_offset"] = config["zoom_offset"]
        if "format" in config:
            kwargs["format"] = VectorFormatConfig.from_config(config["format"])
        return VectorLayerConfig(**kwargs)

    def get_final_projection_and_bounds(
        self,
        projection: Projection,
        bounds: Optional[PixelBounds],
    ) -> tuple[Projection, Optional[PixelBounds]]:
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
                bounds = tuple(x * (2**self.zoom_offset) for x in bounds)
            else:
                bounds = tuple(x // (2 ** (-self.zoom_offset)) for x in bounds)
        return projection, bounds


def load_layer_config(config: dict[str, Any]) -> LayerConfig:
    layer_type = LayerType(config.get("type"))
    if layer_type == LayerType.RASTER:
        return RasterLayerConfig.from_config(config)
    elif layer_type == LayerType.VECTOR:
        return VectorLayerConfig.from_config(config)
    raise ValueError(f"Unknown layer type {layer_type}")


class TileStoreConfig:
    """A configuration specifying a TileStore."""

    def __init__(self, name: str, config_dict: dict[str, Any]) -> None:
        self.name = name
        self.config_dict = config_dict

    @staticmethod
    def from_config(config: dict[str, Any]) -> "TileStoreConfig":
        return TileStoreConfig(
            name=config["name"],
            config_dict=config,
        )
