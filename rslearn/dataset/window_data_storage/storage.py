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

    A WindowDataStorage is dataset-level: one instance per dataset, used for
    all windows and layers in that dataset. The dataset config selects which
    implementation to use via :class:`WindowDataStorageConfig`.
    """

    @abstractmethod
    def open_layer_writer(
        self,
        window: Window,
        layer_name: str,
    ) -> LayerWriter:
        """Open a writer for one materialization pass over a layer.

        Args:
            window: the window being written.
            layer_name: the layer name.
        """

    @abstractmethod
    def read_raster(
        self,
        window: Window,
        layer_name: str,
        bands: list[str],
        raster_format: RasterFormat,
        projection: Projection,
        bounds: PixelBounds,
        group_idx: int = 0,
        resampling: Resampling = Resampling.bilinear,
    ) -> RasterArray:
        """Read a single item group's raster.."""

    def read_rasters(
        self,
        window: Window,
        layer_name: str,
        bands: list[str],
        group_idxs: list[int],
        raster_format: RasterFormat,
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> list[RasterArray]:
        """Read rasters for the specified item groups.

        The default implementation loops over :meth:`read_raster`.
        """
        return [
            self.read_raster(
                window,
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
        window: Window,
        layer_name: str,
        vector_format: VectorFormat,
        projection: Projection,
        bounds: PixelBounds,
        group_idx: int = 0,
    ) -> list[Feature]:
        """Read a single item group's vector features."""
