"""Abstract classes for materialized window data storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from rasterio.enums import Resampling

from rslearn.utils.feature import Feature
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import RasterFormat
from rslearn.utils.vector_format import VectorFormat

if TYPE_CHECKING:
    from rslearn.dataset.window import Window


class LayerWriter(ABC):
    """A layer-scoped writer used as a context manager.

    Returned by :meth:`WindowDataStorage.open_layer_writer`. Callers iterate
    over item groups, calling :meth:`write_raster` / :meth:`write_vector`,
    and the writer's ``__exit__`` flushes any buffered data.
    """

    @abstractmethod
    def write_raster(
        self,
        bands: list[str],
        raster_format: RasterFormat,
        projection: Projection,
        bounds: PixelBounds,
        raster: RasterArray,
        group_idx: int = 0,
    ) -> None:
        """Write a single item group's raster for one band set."""

    @abstractmethod
    def write_vector(
        self,
        vector_format: VectorFormat,
        features: list[Feature],
        group_idx: int = 0,
    ) -> None:
        """Write a single item group's vector features."""

    def __enter__(self) -> LayerWriter:
        """Return ``self`` so the writer can be used as a context manager."""
        return self

    @abstractmethod
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Flush any buffered data on success.

        Implementations should not flush if ``exc_type`` is not ``None``.
        """


class WindowDataStorage(ABC):
    """Storage backend for per-window materialized raster and vector data.

    A WindowDataStorage is bound to a specific window. It is created by a
    :class:`WindowDataStorageFactory` and holds a reference to its window.
    """

    def __init__(self, window: Window) -> None:
        """Initialize the storage bound to a specific window.

        Args:
            window: the window this storage is bound to.
        """
        self.window = window

    @abstractmethod
    def open_layer_writer(
        self,
        layer_name: str,
    ) -> LayerWriter:
        """Open a writer for one materialization pass over a layer.

        Args:
            layer_name: the layer name.
        """

    @abstractmethod
    def read_raster(
        self,
        layer_name: str,
        bands: list[str],
        raster_format: RasterFormat,
        projection: Projection | None = None,
        bounds: PixelBounds | None = None,
        group_idx: int = 0,
        resampling: Resampling = Resampling.bilinear,
    ) -> RasterArray:
        """Read a single item group's raster.

        Args:
            layer_name: the layer name.
            bands: the band set to read.
            raster_format: the raster format to decode with.
            projection: target projection (defaults to window projection).
            bounds: target bounds (defaults to window bounds).
            group_idx: the item group index (default 0).
            resampling: resampling method (defaults to bilinear).
        """

    def read_rasters(
        self,
        layer_name: str,
        bands: list[str],
        group_idxs: list[int],
        raster_format: RasterFormat,
        projection: Projection | None = None,
        bounds: PixelBounds | None = None,
        resampling: Resampling = Resampling.bilinear,
    ) -> list[RasterArray]:
        """Read rasters for the specified item groups.

        The default implementation loops over :meth:`read_raster`.
        """
        return [
            self.read_raster(
                layer_name,
                bands,
                raster_format,
                projection,
                bounds,
                group_idx=group_idx,
                resampling=resampling,
            )
            for group_idx in group_idxs
        ]

    @abstractmethod
    def read_vector(
        self,
        layer_name: str,
        vector_format: VectorFormat,
        projection: Projection | None = None,
        bounds: PixelBounds | None = None,
        group_idx: int = 0,
    ) -> list[Feature]:
        """Read a single item group's vector features.

        Args:
            layer_name: the layer name.
            vector_format: the vector format to decode with.
            projection: target projection (defaults to window projection).
            bounds: target bounds (defaults to window bounds).
            group_idx: the item group index (default 0).
        """


class WindowDataStorageFactory(ABC):
    """Factory that creates a :class:`WindowDataStorage` bound to a window.

    The dataset config selects which implementation to use via
    :class:`WindowDataStorageConfig`. The dataset holds a factory and uses it
    to bind data storage to each loaded window.
    """

    @abstractmethod
    def create(self, window: Window) -> WindowDataStorage:
        """Create a WindowDataStorage bound to the given window.

        Args:
            window: the window to bind the storage to.
        """
