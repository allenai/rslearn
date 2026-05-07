"""Default WindowDataStorage that writes one directory per item group.

- Raster: ``layers/{layer_name}.{group_idx}/{bandset_dir}/...``
- Vector: ``layers/{layer_name}.{group_idx}/...``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rasterio.enums import Resampling
from typing_extensions import override
from upath import UPath

from rslearn.utils.feature import Feature
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import RasterFormat, get_bandset_dirname
from rslearn.utils.vector_format import VectorFormat

from .storage import LayerWriter, WindowDataStorage

if TYPE_CHECKING:
    from rslearn.dataset.window import Window

LAYERS_SUBDIR = "layers"


def _per_item_group_layer_dir(
    window_root: UPath, layer_name: str, group_idx: int = 0
) -> UPath:
    """Per-item-group layer directory: ``layers/{layer_name}`` or ``layers/{layer_name}.{group_idx}``."""
    folder_name = layer_name if group_idx == 0 else f"{layer_name}.{group_idx}"
    return window_root / LAYERS_SUBDIR / folder_name


def _per_item_group_raster_dir(
    window_root: UPath, layer_name: str, bands: list[str], group_idx: int
) -> UPath:
    """Per-item-group raster directory: ``layers/{layer_name}.{group_idx}/{bandset}/``."""
    dirname = get_bandset_dirname(bands)
    return _per_item_group_layer_dir(window_root, layer_name, group_idx) / dirname


class _PerItemGroupLayerWriter(LayerWriter):
    """Writer for :class:`PerItemGroupStorage`. Writes each item group immediately."""

    def __init__(self, window: Window, layer_name: str) -> None:
        """Initialize a writer bound to ``window`` and ``layer_name``."""
        self._window = window
        self._layer_name = layer_name

    @override
    def write_raster(
        self,
        bands: list[str],
        raster_format: RasterFormat,
        projection: Projection,
        bounds: PixelBounds,
        raster: RasterArray,
        group_idx: int = 0,
    ) -> None:
        """Encode ``raster`` to the per-group on-disk path immediately."""
        raster_dir = _per_item_group_raster_dir(
            self._window.window_root, self._layer_name, bands, group_idx
        )
        raster_format.encode_raster(raster_dir, projection, bounds, raster)

    @override
    def write_vector(
        self,
        vector_format: VectorFormat,
        features: list[Feature],
        group_idx: int = 0,
    ) -> None:
        """Encode ``features`` to the per-group on-disk path immediately."""
        layer_dir = _per_item_group_layer_dir(
            self._window.window_root, self._layer_name, group_idx
        )
        vector_format.encode_vector(layer_dir, features)

    @override
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """No-op: writes are flushed inside each ``write_*`` call."""
        return None


class PerItemGroupStorage(WindowDataStorage):
    """Default storage that writes one directory per item group.

    On-disk layout:

    - Raster: ``layers/{layer_name}.{group_idx}/{bandset_dir}/...``
    - Vector: ``layers/{layer_name}.{group_idx}/...``
    """

    @override
    def open_layer_writer(
        self,
        window: Window,
        layer_name: str,
    ) -> LayerWriter:
        """Return a writer that writes each raster individually upon ``write_raster`` call."""
        return _PerItemGroupLayerWriter(window, layer_name)

    @override
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
        """Decode the raster from the per-group directory."""
        raster_dir = _per_item_group_raster_dir(
            window.window_root, layer_name, bands, group_idx
        )
        return raster_format.decode_raster(raster_dir, projection, bounds, resampling)

    @override
    def read_vector(
        self,
        window: Window,
        layer_name: str,
        vector_format: VectorFormat,
        projection: Projection,
        bounds: PixelBounds,
        group_idx: int = 0,
    ) -> list[Feature]:
        """Decode the vector features from the per-group directory."""
        layer_dir = _per_item_group_layer_dir(window.window_root, layer_name, group_idx)
        return vector_format.decode_vector(layer_dir, projection, bounds)
