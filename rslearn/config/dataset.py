"""Classes for storing configuration of a dataset."""

import copy
import functools
import json
import warnings
from datetime import timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any

import jsonargparse
import numpy as np
import numpy.typing as npt
import pytimeparse
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    field_validator,
    model_validator,
)
from rasterio.enums import Resampling
from upath import UPath

from rslearn.log_utils import get_logger
from rslearn.utils import PixelBounds, Projection
from rslearn.utils.raster_format import RasterFormat
from rslearn.utils.vector_format import VectorFormat

if TYPE_CHECKING:
    from rslearn.data_sources.data_source import DataSource

logger = get_logger("__name__")


def ensure_timedelta(v: Any) -> Any:
    """Ensure the value is a timedelta.

    If the value is a string, we try to parse it with pytimeparse.

    This function is meant to be used like Annotated[timedelta, BeforeValidator(ensure_timedelta)].
    """
    if isinstance(v, timedelta):
        return v
    if isinstance(v, str):
        return pytimeparse.parse(v)
    raise TypeError(f"Invalid type for timedelta: {type(v).__name__}")


def ensure_optional_timedelta(v: Any) -> Any:
    """Like ensure_timedelta, but allows None as a value."""
    if v is None:
        return None
    if isinstance(v, timedelta):
        return v
    if isinstance(v, str):
        return pytimeparse.parse(v)
    raise TypeError(f"Invalid type for timedelta: {type(v).__name__}")


def serialize_optional_timedelta(v: timedelta | None) -> str | None:
    """Serialize an optional timedelta for compatibility with pytimeparse."""
    if v is None:
        return None
    return str(v.total_seconds()) + "s"


class DType(StrEnum):
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


class ResamplingMethod(StrEnum):
    """An enum representing the rasterio Resampling."""

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    CUBIC = "cubic"
    CUBIC_SPLINE = "cubic_spline"

    def get_rasterio_resampling(self) -> Resampling:
        """Get the rasterio Resampling corresponding to this ResamplingMethod."""
        return RESAMPLING_METHODS[self]


RESAMPLING_METHODS = {
    ResamplingMethod.NEAREST: Resampling.nearest,
    ResamplingMethod.BILINEAR: Resampling.bilinear,
    ResamplingMethod.CUBIC: Resampling.cubic,
    ResamplingMethod.CUBIC_SPLINE: Resampling.cubic_spline,
}


class BandSetConfig(BaseModel):
    """A configuration for a band set in a raster layer.

    Each band set specifies one or more bands that should be stored together.
    It also specifies the storage format and dtype, the zoom offset, etc. for these
    bands.
    """

    dtype: DType = Field(description="Pixel value type to store the data under")
    bands: list[str] = Field(
        default_factory=lambda: [],
        description="List of band names in this BandSetConfig. One of bands or num_bands must be set.",
    )
    num_bands: int | None = Field(
        default=None,
        description="The number of bands in this band set. The bands will be named B0, B1, B2, etc.",
    )
    format: dict[str, Any] = Field(
        default_factory=lambda: {
            "class_path": "rslearn.utils.raster_format.GeotiffRasterFormat"
        },
        description="jsonargparse configuration for the RasterFormat to store the tiles in.",
    )

    # Store images at a resolution higher or lower than the window resolution. This
    # enables keeping source data at its native resolution, either to save storage
    # space (for lower resolution data) or to retain details (for higher resolution
    # data). If positive, store data at the window resolution divided by
    # 2^(zoom_offset) (higher resolution). If negative, store data at the window
    # resolution multiplied by 2^(-zoom_offset) (lower resolution).
    zoom_offset: int = Field(
        default=0,
        description="Store data at the window resolution multiplied by 2^(-zoom_offset).",
    )

    remap: dict[str, Any] | None = Field(
        default=None,
        description="Optional jsonargparse configuration for a Remapper to remap pixel values.",
    )

    # Optional list of names for the different possible values of each band. The length
    # of this list must equal the number of bands. For example, [["forest", "desert"]]
    # means that it is a single-band raster where values can be 0 (forest) or 1
    # (desert).
    class_names: list[list[str]] | None = Field(
        default=None,
        description="Optional list of names for the different possible values of each band.",
    )

    # Optional list of nodata values for this band set. This is used during
    # materialization when creating mosaics, to determine which parts of the source
    # images should be copied.
    nodata_vals: list[float] | None = Field(
        default=None, description="Optional nodata value for each band."
    )

    @model_validator(mode="after")
    def after_validator(self) -> "BandSetConfig":
        """Ensure the BandSetConfig is valid, and handle the num_bands field."""
        if (len(self.bands) == 0 and self.num_bands is None) or (
            len(self.bands) != 0 and self.num_bands is not None
        ):
            raise ValueError("exactly one of bands and num_bands must be specified")

        if self.num_bands is not None:
            self.bands = [f"B{band_idx}" for band_idx in range(self.num_bands)]
            self.num_bands = None

        return self

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

    @field_validator("format", mode="before")
    @classmethod
    def convert_format_from_legacy(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Support legacy format of the RasterFormat.

        The legacy format sets 'name' instead of 'class_path', and uses custom parsing
        for the init_args.
        """
        if "name" not in v:
            # New version, it is all good.
            return v

        warnings.warn(
            "`format = {'name': ...}` is deprecated; "
            "use `{'class_path': '...', 'init_args': {...}}` instead.",
            DeprecationWarning,
        )
        logger.warning(
            "BandSet.format uses legacy format; support will be removed after 2026-03-01."
        )

        legacy_name_to_class_path = {
            "image_tile": "rslearn.utils.raster_format.ImageTileRasterFormat",
            "geotiff": "rslearn.utils.raster_format.GeotiffRasterFormat",
            "single_image": "rslearn.utils.raster_format.SingleImageRasterFormat",
        }
        if v["name"] not in legacy_name_to_class_path:
            raise ValueError(
                f"could not parse legacy format with unknown raster format {v['name']}"
            )
        init_args = dict(v)
        class_path = legacy_name_to_class_path[init_args.pop("name")]

        return dict(
            class_path=class_path,
            init_args=init_args,
        )

    def instantiate_raster_format(self) -> RasterFormat:
        """Instantiate the RasterFormat specified by this BandSetConfig."""
        from rslearn.utils.jsonargparse import init_jsonargparse

        init_jsonargparse()
        parser = jsonargparse.ArgumentParser()
        parser.add_argument("--raster_format", type=RasterFormat)
        cfg = parser.parse_object({"raster_format": self.format})
        raster_format = parser.instantiate_classes(cfg).raster_format
        return raster_format


class SpaceMode(StrEnum):
    """Spatial matching mode when looking up items corresponding to a window."""

    CONTAINS = "CONTAINS"
    """Items must contain the entire window."""

    INTERSECTS = "INTERSECTS"
    """Items must overlap any portion of the window."""

    MOSAIC = "MOSAIC"
    """Groups of items should be computed that cover the entire window.

    During materialization, items in each group are merged to form a mosaic in the
    dataset.
    """

    PER_PERIOD_MOSAIC = "PER_PERIOD_MOSAIC"
    """Create one mosaic per sub-period of the time range.

    The duration of the sub-periods is controlled by another option in QueryConfig.
    """

    COMPOSITE = "COMPOSITE"
    """Creates one composite covering the entire window.

    During querying all items intersecting the window are placed in one group.
    The compositing_method in the rasterlayer config specifies how these items are reduced
    to a single item (e.g MEAN/MEDIAN/FIRST_VALID) during materialization.
    """

    # TODO add PER_PERIOD_COMPOSITE


class TimeMode(StrEnum):
    """Temporal  matching mode when looking up items corresponding to a window."""

    WITHIN = "WITHIN"
    """Items must be within the window time range."""

    NEAREST = "NEAREST"
    """Select items closest to the window time range, up to max_matches."""

    BEFORE = "BEFORE"
    """Select items before the end of the window time range, up to max_matches."""

    AFTER = "AFTER"
    """Select items after the start of the window time range, up to max_matches."""


class QueryConfig(BaseModel):
    """A configuration for querying items in a data source."""

    model_config = ConfigDict(frozen=True)

    space_mode: SpaceMode = Field(
        default=SpaceMode.MOSAIC,
        description="Specifies how items should be matched with windows spatially.",
    )
    time_mode: TimeMode = Field(
        default=TimeMode.WITHIN,
        description="Specifies how items should be matched with windows temporally.",
    )

    # Minimum number of item groups. If there are fewer than this many matches, then no
    # matches will be returned. This can be used to prevent unnecessary data ingestion
    # if the user plans to discard windows that do not have a sufficient amount of data.
    min_matches: int = Field(
        default=0, description="The minimum number of item groups."
    )

    max_matches: int = Field(
        default=1, description="The maximum number of item groups."
    )
    period_duration: Annotated[
        timedelta,
        BeforeValidator(ensure_timedelta),
        PlainSerializer(serialize_optional_timedelta),
    ] = Field(
        default=timedelta(days=30),
        description="The duration of the periods, if the space mode is PER_PERIOD_MOSAIC.",
    )


class DataSourceConfig(BaseModel):
    """Configuration for a DataSource in a dataset layer."""

    model_config = ConfigDict(frozen=True)

    class_path: str = Field(description="Class path for the data source.")
    init_args: dict[str, Any] = Field(
        default_factory=lambda: {},
        description="jsonargparse init args for the data source.",
    )
    query_config: QueryConfig = Field(
        default_factory=lambda: QueryConfig(),
        description="QueryConfig specifying how to match items with windows.",
    )
    time_offset: Annotated[
        timedelta | None,
        BeforeValidator(ensure_optional_timedelta),
        PlainSerializer(serialize_optional_timedelta),
    ] = Field(
        default=None,
        description="Optional timedelta to add to the window's time range before matching.",
    )
    duration: Annotated[
        timedelta | None,
        BeforeValidator(ensure_optional_timedelta),
        PlainSerializer(serialize_optional_timedelta),
    ] = Field(
        default=None,
        description="Optional, if the window's time range is (t0, t1), then update to (t0, t0 + duration).",
    )
    ingest: bool = Field(
        default=True,
        description="Whether to ingest this layer (default True). If False, it will be directly materialized without ingestion.",
    )

    @model_validator(mode="before")
    @classmethod
    def convert_from_legacy(cls, d: dict[str, Any]) -> dict[str, Any]:
        """Support legacy format of the DataSourceConfig.

        The legacy format sets 'name' instead of 'class_path', and mixes the arguments
        for the DataSource in with the DataSourceConfig keys.
        """
        if "name" not in d:
            # New version, it is all good.
            return d

        warnings.warn(
            "`Data source configuration {'name': ...}` is deprecated; "
            "use `{'class_path': '...', 'init_args': {...}, ...}` instead.",
            DeprecationWarning,
        )
        logger.warning(
            "Data source configuration uses legacy format; support will be removed after 2026-03-01."
        )

        # Split the dict into the base config that is in the pydantic model, and the
        # source-specific options that should be moved to init_args dict.
        class_path = d["name"]
        base_config: dict[str, Any] = {}
        ds_init_args: dict[str, Any] = {}
        for k, v in d.items():
            if k == "name":
                continue
            if k in cls.model_fields:
                base_config[k] = v
            else:
                ds_init_args[k] = v

        # Some legacy configs erroneously specify these keys, which are now caught by
        # validation. But we still want those specific legacy configs to work.
        if (
            class_path == "rslearn.data_sources.planetary_computer.Sentinel2"
            and "max_cloud_cover" in ds_init_args
        ):
            warnings.warn(
                "Data source configuration specifies invalid 'max_cloud_cover' option.",
                DeprecationWarning,
            )
            del ds_init_args["max_cloud_cover"]

        base_config["class_path"] = class_path
        base_config["init_args"] = ds_init_args
        return base_config


class LayerType(StrEnum):
    """The layer type (raster or vector)."""

    RASTER = "raster"
    VECTOR = "vector"


class CompositingMethod(StrEnum):
    """Method how to select pixels for the composite from corresponding items of a window."""

    FIRST_VALID = "FIRST_VALID"
    """Select first valid pixel in order of corresponding items (might be sorted)"""

    MEAN = "MEAN"
    """Select per-pixel mean value of corresponding items of a window"""

    MEDIAN = "MEDIAN"
    """Select per-pixel median value of corresponding items of a window"""


class LayerConfig(BaseModel):
    """Configuration of a layer in a dataset."""

    model_config = ConfigDict(frozen=True)

    type: LayerType = Field(description="The LayerType (raster or vector).")
    data_source: DataSourceConfig | None = Field(
        default=None,
        description="Optional DataSourceConfig if this layer is retrievable.",
    )
    alias: str | None = Field(
        default=None, description="Alias for this layer to use in the tile store."
    )

    # Raster layer options.
    band_sets: list[BandSetConfig] = Field(
        default_factory=lambda: [],
        description="For raster layers, the bands to store in this layer.",
    )
    resampling_method: ResamplingMethod = Field(
        default=ResamplingMethod.BILINEAR,
        description="For raster layers, how to resample rasters (if neeed), default bilinear resampling.",
    )
    compositing_method: CompositingMethod = Field(
        default=CompositingMethod.FIRST_VALID,
        description="For raster layers, how to compute pixel values in the composite of each window's items.",
    )

    # Vector layer options.
    vector_format: dict[str, Any] = Field(
        default_factory=lambda: {
            "class_path": "rslearn.utils.vector_format.GeojsonVectorFormat"
        },
        description="For vector layers, the jsonargparse configuration for the VectorFormat.",
    )
    class_property_name: str | None = Field(
        default=None,
        description="Optional metadata field indicating that the GeoJSON features contain a property that corresponds to a class label, and this is the name of that property.",
    )
    class_names: list[str] | None = Field(
        default=None,
        description="The list of classes that the class_property_name property could be set to.",
    )

    @model_validator(mode="after")
    def after_validator(self) -> "LayerConfig":
        """Ensure the LayerConfig is valid."""
        if self.type == LayerType.RASTER and len(self.band_sets) == 0:
            raise ValueError(
                "band sets must be specified and non-empty for raster layers"
            )

        return self

    def __hash__(self) -> int:
        """Return a hash of this LayerConfig."""
        return hash(json.dumps(self.model_dump(mode="json"), sort_keys=True))

    def __eq__(self, other: Any) -> bool:
        """Returns whether other is the same as this LayerConfig.

        Args:
            other: the other object to compare.
        """
        if not isinstance(other, LayerConfig):
            return False
        return self.model_dump() == other.model_dump()

    @functools.cache
    def instantiate_data_source(self, ds_path: UPath | None = None) -> "DataSource":
        """Instantiate the data source specified by this config.

        Args:
            ds_path: optional dataset path to include in the DataSourceContext.

        Returns:
            the DataSource object.
        """
        from rslearn.data_sources.data_source import DataSource, DataSourceContext
        from rslearn.utils.jsonargparse import data_source_context_serializer

        logger.debug("getting a data source for dataset at %s", ds_path)
        if self.data_source is None:
            raise ValueError("This layer does not specify a data source")

        # Inject the DataSourceContext into the args.
        context = DataSourceContext(
            ds_path=ds_path,
            layer_config=self,
        )
        ds_config: dict[str, Any] = {
            "class_path": self.data_source.class_path,
            "init_args": copy.deepcopy(self.data_source.init_args),
        }
        ds_config["init_args"]["context"] = data_source_context_serializer(context)

        # Now we can parse with jsonargparse.
        from rslearn.utils.jsonargparse import (
            data_source_context_serializer,
            init_jsonargparse,
        )

        init_jsonargparse()
        parser = jsonargparse.ArgumentParser()
        parser.add_argument("--data_source", type=DataSource)
        cfg = parser.parse_object({"data_source": ds_config})
        data_source = parser.instantiate_classes(cfg).data_source
        return data_source

    def instantiate_vector_format(self) -> VectorFormat:
        """Instantiate the vector format specified by this config."""
        if self.type != LayerType.VECTOR:
            raise ValueError(
                f"cannot instantiate vector format for layer with type {self.type}"
            )

        from rslearn.utils.jsonargparse import init_jsonargparse

        init_jsonargparse()
        parser = jsonargparse.ArgumentParser()
        parser.add_argument("--vector_format", type=VectorFormat)
        cfg = parser.parse_object({"vector_format": self.vector_format})
        vector_format = parser.instantiate_classes(cfg).vector_format
        return vector_format


class DatasetConfig(BaseModel):
    """Overall dataset configuration."""

    layers: dict[str, LayerConfig] = Field(description="Layers in the dataset.")
    tile_store: dict[str, Any] = Field(
        default={"class_path": "rslearn.tile_stores.default.DefaultTileStore"},
        description="jsonargparse configuration for the TileStore.",
    )
