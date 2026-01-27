"""Raster data storage abstraction for materialized window data."""

import json
from abc import ABC, abstractmethod
from typing import Any, override

import numpy as np
import numpy.typing as npt
from rasterio.enums import Resampling
from upath import UPath

from rslearn.utils import Projection
from rslearn.utils.geometry import PixelBounds
from rslearn.utils.raster_format import RasterFormat, get_bandset_dirname

from .window import LAYERS_DIRECTORY_NAME, get_window_layer_dir


class RasterDataStorage(ABC):
    """Abstract base class for storing materialized raster data.

    This abstraction handles reading/writing raster data for layers, separating
    the storage strategy from RasterFormat (which handles encoding/decoding CHW
    arrays) and WindowStorage (which handles metadata).

    Implementations:
        - PerItemGroupStorage: One file per item group (default behavior)
        - PerLayerStorage: All item groups in a single file per layer
    """

    @abstractmethod
    def write_raster(
        self,
        window_root: UPath,
        layer_name: str,
        bands: list[str],
        group_idx: int,
        raster_format: RasterFormat,
        projection: Projection,
        bounds: PixelBounds,
        array: npt.NDArray[Any],
    ) -> None:
        """Write raster for a single item group.

        Args:
            window_root: Root directory of the window.
            layer_name: Name of the layer.
            bands: List of band names for this raster.
            group_idx: Index of the item group.
            raster_format: RasterFormat to use for encoding.
            projection: Projection of the raster data.
            bounds: Bounds in the projection.
            array: CHW numpy array to write.
        """

    def write_all_rasters(
        self,
        window_root: UPath,
        layer_name: str,
        bands: list[str],
        raster_format: RasterFormat,
        projection: Projection,
        bounds: PixelBounds,
        rasters: npt.NDArray[Any],
    ) -> None:
        """Write rasters for all item groups.

        Default implementation loops over write_raster. Subclasses may override
        for more efficient batch writing.

        Args:
            window_root: Root directory of the window.
            layer_name: Name of the layer.
            bands: List of band names for this raster.
            raster_format: RasterFormat to use for encoding.
            projection: Projection of the raster data.
            bounds: Bounds in the projection.
            rasters: TCHW numpy array where T is the number of item groups.
        """
        for group_idx in range(rasters.shape[0]):
            self.write_raster(
                window_root,
                layer_name,
                bands,
                group_idx,
                raster_format,
                projection,
                bounds,
                rasters[group_idx],
            )

    @abstractmethod
    def read_raster(
        self,
        window_root: UPath,
        layer_name: str,
        bands: list[str],
        group_idx: int,
        raster_format: RasterFormat,
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> npt.NDArray[Any]:
        """Read raster for a single item group.

        Args:
            window_root: Root directory of the window.
            layer_name: Name of the layer.
            bands: List of band names for this raster.
            group_idx: Index of the item group.
            raster_format: RasterFormat to use for decoding.
            projection: Projection to read in.
            bounds: Bounds to read in the projection.
            resampling: Resampling method if needed.

        Returns:
            CHW numpy array.
        """

    def read_all_rasters(
        self,
        window_root: UPath,
        layer_name: str,
        bands: list[str],
        num_groups: int,
        raster_format: RasterFormat,
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> npt.NDArray[Any]:
        """Read rasters for all item groups.

        Default implementation loops over read_raster and stacks. Subclasses may
        override for more efficient batch reading.

        Args:
            window_root: Root directory of the window.
            layer_name: Name of the layer.
            bands: List of band names for this raster.
            num_groups: Number of item groups to read.
            raster_format: RasterFormat to use for decoding.
            projection: Projection to read in.
            bounds: Bounds to read in the projection.
            resampling: Resampling method if needed.

        Returns:
            TCHW numpy array where T is num_groups.
        """
        arrays = []
        for group_idx in range(num_groups):
            arrays.append(
                self.read_raster(
                    window_root,
                    layer_name,
                    bands,
                    group_idx,
                    raster_format,
                    projection,
                    bounds,
                    resampling,
                )
            )
        return np.stack(arrays, axis=0)


def _get_raster_dir(
    window_root: UPath, layer_name: str, bands: list[str], group_idx: int
) -> UPath:
    """Get the directory where a raster is stored for PerItemGroupStorage.

    This is an internal helper that mirrors the old get_window_raster_dir behavior.

    Args:
        window_root: Root directory of the window.
        layer_name: Name of the layer.
        bands: List of band names.
        group_idx: Index of the item group.

    Returns:
        Directory path for the raster.
    """
    dirname = get_bandset_dirname(bands)
    return get_window_layer_dir(window_root, layer_name, group_idx) / dirname


class PerItemGroupStorage(RasterDataStorage):
    """Storage that writes one file per item group (current/default behavior).

    File structure: layers/{layer_name}.{group_idx}/{bandset_dir}/

    This implementation maintains full backward compatibility with existing
    datasets created before the RasterDataStorage abstraction was introduced.
    """

    @override
    def write_raster(
        self,
        window_root: UPath,
        layer_name: str,
        bands: list[str],
        group_idx: int,
        raster_format: RasterFormat,
        projection: Projection,
        bounds: PixelBounds,
        array: npt.NDArray[Any],
    ) -> None:
        raster_dir = _get_raster_dir(window_root, layer_name, bands, group_idx)
        raster_format.encode_raster(raster_dir, projection, bounds, array)

    @override
    def read_raster(
        self,
        window_root: UPath,
        layer_name: str,
        bands: list[str],
        group_idx: int,
        raster_format: RasterFormat,
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> npt.NDArray[Any]:
        raster_dir = _get_raster_dir(window_root, layer_name, bands, group_idx)
        return raster_format.decode_raster(raster_dir, projection, bounds, resampling)


class PerLayerStorage(RasterDataStorage):
    """Storage that writes all item groups to a single file per layer.

    File structure: layers/{layer_name}/{bandset_dir}/

    The T (time/item group) dimension is flattened into the channel dimension,
    resulting in (T*C, H, W) shape. A metadata sidecar file stores num_groups
    and num_channels so the original shape can be reconstructed on read.

    This storage type is more efficient for reading all item groups at once
    but does not support incremental/concurrent writes. Use PerItemGroupStorage
    (the default) for prediction outputs or any case requiring write_raster.
    """

    METADATA_FILENAME = "raster_storage_meta.json"

    @override
    def write_raster(
        self,
        window_root: UPath,
        layer_name: str,
        bands: list[str],
        group_idx: int,
        raster_format: RasterFormat,
        projection: Projection,
        bounds: PixelBounds,
        array: npt.NDArray[Any],
    ) -> None:
        """Not supported - raises NotImplementedError.

        PerLayerStorage cannot support concurrent writes from multiple processes.
        Use write_all_rasters instead, or use PerItemGroupStorage for cases
        requiring incremental writes.
        """
        raise NotImplementedError(
            "PerLayerStorage does not support write_raster. Use write_all_rasters "
            "for batch writing, or use PerItemGroupStorage for incremental writes."
        )

    @override
    def write_all_rasters(
        self,
        window_root: UPath,
        layer_name: str,
        bands: list[str],
        raster_format: RasterFormat,
        projection: Projection,
        bounds: PixelBounds,
        rasters: npt.NDArray[Any],
    ) -> None:
        """Write all item groups to a single file.

        Flattens TCHW to (T*C, H, W) for storage. Writes metadata to reconstruct
        the original shape.
        """
        if rasters.size == 0:
            return

        num_groups = rasters.shape[0]
        num_channels = rasters.shape[1]

        # Reshape from (T, C, H, W) to (T*C, H, W)
        flattened = rasters.reshape(-1, rasters.shape[2], rasters.shape[3])

        # Get directory for this layer (no group_idx suffix)
        dirname = get_bandset_dirname(bands)
        raster_dir = window_root / LAYERS_DIRECTORY_NAME / layer_name / dirname

        # Write the flattened raster
        raster_format.encode_raster(raster_dir, projection, bounds, flattened)

        # Write metadata for reconstruction
        metadata = {
            "num_groups": num_groups,
            "num_channels": num_channels,
        }
        metadata_path = raster_dir / self.METADATA_FILENAME
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w") as f:
            json.dump(metadata, f)

    def _read_metadata(self, raster_dir: UPath) -> dict[str, Any]:
        """Read the metadata sidecar file."""
        metadata_path = raster_dir / self.METADATA_FILENAME
        with metadata_path.open("r") as f:
            return json.load(f)

    @override
    def read_raster(
        self,
        window_root: UPath,
        layer_name: str,
        bands: list[str],
        group_idx: int,
        raster_format: RasterFormat,
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> npt.NDArray[Any]:
        """Read raster for a single item group.

        This reads the entire file and extracts the requested group, which is
        slow. Use read_all_rasters for efficient batch reading.
        """
        dirname = get_bandset_dirname(bands)
        raster_dir = window_root / LAYERS_DIRECTORY_NAME / layer_name / dirname

        # Read metadata
        metadata = self._read_metadata(raster_dir)
        num_groups = metadata["num_groups"]
        num_channels = metadata["num_channels"]

        if group_idx < 0 or group_idx >= num_groups:
            raise ValueError(f"group_idx {group_idx} out of range [0, {num_groups})")

        # Read the full flattened raster
        flattened = raster_format.decode_raster(
            raster_dir, projection, bounds, resampling
        )

        # Reshape to (T, C, H, W) and extract the requested group
        reshaped = flattened.reshape(
            num_groups, num_channels, flattened.shape[1], flattened.shape[2]
        )
        return reshaped[group_idx]

    @override
    def read_all_rasters(
        self,
        window_root: UPath,
        layer_name: str,
        bands: list[str],
        num_groups: int,
        raster_format: RasterFormat,
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> npt.NDArray[Any]:
        """Read all item groups efficiently from the single file."""
        dirname = get_bandset_dirname(bands)
        raster_dir = window_root / LAYERS_DIRECTORY_NAME / layer_name / dirname

        # Read metadata
        metadata = self._read_metadata(raster_dir)
        stored_num_groups = metadata["num_groups"]
        num_channels = metadata["num_channels"]

        if num_groups != stored_num_groups:
            raise ValueError(
                f"Expected {num_groups} groups but file contains {stored_num_groups}"
            )

        # Read the full flattened raster
        flattened = raster_format.decode_raster(
            raster_dir, projection, bounds, resampling
        )

        # Reshape to (T, C, H, W)
        return flattened.reshape(
            stored_num_groups, num_channels, flattened.shape[1], flattened.shape[2]
        )
