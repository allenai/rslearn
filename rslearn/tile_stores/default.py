"""Default TileStore implementation."""

import math
import shutil
from typing import Any

import affine
import numpy.typing as npt
import rasterio.vrt
import shapely
from rasterio.enums import Resampling
from upath import UPath

from rslearn.utils import Feature, PixelBounds, Projection, STGeometry
from rslearn.utils.fsspec import (
    join_upath,
    open_rasterio_upath_reader,
    open_rasterio_upath_writer,
)
from rslearn.utils.raster_format import (
    GeotiffRasterFormat,
)
from rslearn.utils.vector_format import (
    GeojsonVectorFormat,
)

from .tile_store import TileStore

# Special filename to indicate writing is done.
COMPLETED_FNAME = "completed"

# Filename to use for vector data.
VECTOR_FNAME = "data.geojson"


class DefaultTileStore(TileStore):
    """Default TileStore implementation.

    It stores raster and vector data under the provided UPath.
    """

    def __init__(
        self,
        path_suffix: str = "tiles",
        convert_rasters_to_cogs: bool = True,
        tile_size: int = 256,
        geotiff_options: dict[str, Any] = {},
    ):
        """Create a new DefaultTileStore.

        Args:
            path_suffix: the path suffix to store files under, which is joined with
                the dataset path if it does not contain a protocol string. See
                rslearn.utils.fsspec.join_upath.
            convert_rasters_to_cogs: whether to re-encode all raster files to tiled
                GeoTIFFs.
            tile_size: if converting to COGs, the tile size to use.
            geotiff_options: other options to pass to rasterio.open (for writes).
        """
        self.path_suffix = path_suffix
        self.convert_rasters_to_cogs = convert_rasters_to_cogs
        self.tile_size = tile_size
        self.geotiff_options = geotiff_options

        self.path: UPath | None = None

    def set_dataset_path(self, ds_path: UPath) -> None:
        """Set the dataset path.

        Args:
            ds_path: the dataset path.
        """
        self.path = join_upath(ds_path, self.path_suffix)

    def _get_raster_dir(
        self, layer_name: str, item_name: str, bands: list[str]
    ) -> UPath:
        assert self.path is not None
        if any(["_" in band for band in bands]):
            raise ValueError("band names must not contain '_'")
        return self.path / layer_name / item_name / "_".join(bands)

    def is_raster_ready(
        self, layer_name: str, item_name: str, bands: list[str]
    ) -> bool:
        """Checks if this raster has been written to the store.

        Args:
            layer_name: the layer name or alias.
            item_name: the item.
            bands: the list of bands identifying which specific raster to read.

        Returns:
            whether there is a raster in the store matching the source, item, and
                bands.
        """
        raster_dir = self._get_raster_dir(layer_name, item_name, bands)
        return (raster_dir / COMPLETED_FNAME).exists()

    def get_raster_bands(self, layer_name: str, item_name: str) -> list[list[str]]:
        """Get the sets of bands that have been stored for the specified item.

        Args:
            layer_name: the layer name or alias.
            item_name: the item.

        Returns:
            a list of lists of bands that are in the tile store (with one raster
                stored corresponding to each inner list).
        """
        assert isinstance(self.path, UPath)
        item_dir = self.path / layer_name / item_name
        if not item_dir.exists():
            return []

        bands: list[list[str]] = []
        for raster_dir in item_dir.iterdir():
            parts = raster_dir.name.split("_")
            bands.append(parts)
        return bands

    def get_raster_bounds(
        self, layer_name: str, item_name: str, bands: list[str], projection: Projection
    ) -> PixelBounds:
        """Get the bounds of the raster in the specified projection.

        Args:
            layer_name: the layer name or alias.
            item_name: the item to check.
            bands: the list of bands identifying which specific raster to read. These
                bands must match the bands of a stored raster.
            projection: the projection to get the raster's bounds in.

        Returns:
            the bounds of the raster in the projection.
        """
        raster_dir = self._get_raster_dir(layer_name, item_name, bands)
        fnames = [
            fname for fname in raster_dir.iterdir() if fname.name != COMPLETED_FNAME
        ]
        assert len(fnames) == 1
        raster_fname = fnames[0]

        with open_rasterio_upath_reader(raster_fname) as src:
            with rasterio.vrt.WarpedVRT(src, crs=projection.crs) as vrt:
                bounds = (
                    vrt.bounds[0] / projection.x_resolution,
                    vrt.bounds[1] / projection.y_resolution,
                    vrt.bounds[2] / projection.x_resolution,
                    vrt.bounds[3] / projection.y_resolution,
                )
                return (
                    math.floor(min(bounds[0], bounds[2])),
                    math.floor(min(bounds[1], bounds[3])),
                    math.ceil(max(bounds[0], bounds[2])),
                    math.ceil(max(bounds[1], bounds[3])),
                )

    def read_raster(
        self,
        layer_name: str,
        item_name: str,
        bands: list[str],
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> npt.NDArray[Any]:
        """Read raster data from the store.

        Args:
            layer_name: the layer name or alias.
            item_name: the item to read.
            bands: the list of bands identifying which specific raster to read. These
                bands must match the bands of a stored raster.
            projection: the projection to read in.
            bounds: the bounds to read.
            resampling: resampling method to use in case resampling is needed.

        Returns:
            the raster data
        """
        raster_dir = self._get_raster_dir(layer_name, item_name, bands)
        fnames = [
            fname for fname in raster_dir.iterdir() if fname.name != COMPLETED_FNAME
        ]
        assert len(fnames) == 1
        raster_fname = fnames[0]

        # Construct the transform to use for the warped dataset.
        wanted_transform = affine.Affine(
            projection.x_resolution,
            0,
            bounds[0] * projection.x_resolution,
            0,
            projection.y_resolution,
            bounds[1] * projection.y_resolution,
        )
        with open_rasterio_upath_reader(raster_fname) as src:
            with rasterio.vrt.WarpedVRT(
                src,
                crs=projection.crs,
                transform=wanted_transform,
                width=bounds[2] - bounds[0],
                height=bounds[3] - bounds[1],
                resampling=resampling,
            ) as vrt:
                return vrt.read()

    def write_raster(
        self,
        layer_name: str,
        item_name: str,
        bands: list[str],
        projection: Projection,
        bounds: PixelBounds,
        array: npt.NDArray[Any],
    ) -> None:
        """Write raster data to the store.

        Args:
            layer_name: the layer name or alias.
            item_name: the item to write.
            bands: the list of bands in the array.
            projection: the projection of the array.
            bounds: the bounds of the array.
            array: the raster data.
        """
        raster_dir = self._get_raster_dir(layer_name, item_name, bands)
        raster_format = GeotiffRasterFormat(geotiff_options=self.geotiff_options)
        raster_format.encode_raster(raster_dir, projection, bounds, array)
        (raster_dir / COMPLETED_FNAME).touch()

    def write_raster_file(
        self, layer_name: str, item_name: str, bands: list[str], fname: UPath
    ) -> None:
        """Write raster data to the store.

        Args:
            layer_name: the layer name or alias.
            item_name: the item to write.
            bands: the list of bands in the array.
            fname: the raster file, which must be readable by rasterio.
        """
        raster_dir = self._get_raster_dir(layer_name, item_name, bands)
        raster_dir.mkdir(parents=True, exist_ok=True)

        if self.convert_rasters_to_cogs:
            with open_rasterio_upath_reader(fname) as src:
                profile = src.profile
                array = src.read()

            output_profile = {
                "driver": "GTiff",
                "compress": "lzw",
                "width": array.shape[2],
                "height": array.shape[1],
                "count": array.shape[0],
                "dtype": array.dtype.name,
                "crs": profile["crs"],
                "transform": profile["transform"],
                "BIGTIFF": "IF_SAFER",
                "tiled": True,
                "blockxsize": self.tile_size,
                "blockysize": self.tile_size,
            }

            output_profile.update(self.geotiff_options)

            with open_rasterio_upath_writer(
                raster_dir / "geotiff.tif", **output_profile
            ) as dst:
                dst.write(array)

        else:
            # Just copy the file directly.
            dst_fname = raster_dir / fname.name
            with fname.open("rb") as src:
                with dst_fname.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

        (raster_dir / COMPLETED_FNAME).touch()

    def _get_vector_dir(self, layer_name: str, item_name: str) -> UPath:
        assert self.path is not None
        return self.path / layer_name / item_name

    def is_vector_ready(self, layer_name: str, item_name: str) -> bool:
        """Checks if this vector item has been written to the store.

        Args:
            layer_name: the layer name or alias.
            item_name: the item.

        Returns:
            whether the vector data from the item has been stored.
        """
        vector_dir = self._get_vector_dir(layer_name, item_name)
        return (vector_dir / COMPLETED_FNAME).exists()

    def read_vector(
        self,
        layer_name: str,
        item_name: str,
        projection: Projection,
        bounds: PixelBounds,
    ) -> list[Feature]:
        """Read vector data from the store.

        Args:
            layer_name: the layer name or alias.
            item_name: the item to read.
            projection: the projection to read in.
            bounds: the bounds within which to read.

        Returns:
            the vector data
        """
        vector_dir = self._get_vector_dir(layer_name, item_name)
        features = GeojsonVectorFormat().decode_vector(
            vector_dir / VECTOR_FNAME, bounds
        )

        # Filter for vector data that intersects the requested projection and bounds.
        if len(features) == 0:
            return features
        feat_projection = features[0].geometry.projection
        requested_geom = STGeometry(projection, shapely.box(*bounds), None)
        # We could re-project the features to the requested projection and then match
        # against requested_geom, but we instead project the requested geometry to the
        # projection of the features. This helps to:
        # (a) Avoid unnecessary re-projection of features that don't match the
        #     requested bounds, which is compute-intensive.
        # (b) Avoid re-projection errors when there is a large GeoJSON and some
        #     features are outside the projection bounds.
        requested_geom = requested_geom.to_projection(feat_projection)
        reprojected_features = []
        for feat in features:
            if not requested_geom.intersects(feat.geometry):
                continue
            reprojected_features.append(feat.to_projection(projection))

        return reprojected_features

    def write_vector(
        self, layer_name: str, item_name: str, features: list[Feature]
    ) -> None:
        """Write vector data to the store.

        Args:
            layer_name: the layer name or alias.
            item_name: the item to write.
            features: the vector data.
        """
        vector_dir = self._get_vector_dir(layer_name, item_name)
        vector_dir.mkdir(parents=True, exist_ok=True)
        GeojsonVectorFormat().encode_vector(
            vector_dir / VECTOR_FNAME, features[0].geometry.projection, features
        )
        (vector_dir / COMPLETED_FNAME).touch()
