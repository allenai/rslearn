"""Default TileStore implementation."""

import json
import math
import shutil
from datetime import datetime
from typing import Any

import rasterio.transform
import rasterio.vrt
import shapely
from rasterio.enums import Resampling
from typing_extensions import override
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.utils.feature import Feature
from rslearn.utils.fsspec import (
    iter_nonhidden_files,
    iter_nonhidden_subdirs,
    join_upath,
    open_atomic,
    open_rasterio_upath_reader,
    open_rasterio_upath_writer,
)
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_crs
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import (
    METADATA_FNAME,
    GeotiffRasterFormat,
    GeotiffRasterMetadata,
    get_bandset_dirname,
)
from rslearn.utils.vector_format import (
    GeojsonVectorFormat,
    VectorFormat,
)

from .tile_store import TileStore

# Special filename to indicate writing is done.
COMPLETED_FNAME = "completed"

# Special filename to store the bands that are present in a raster.
BANDS_FNAME = "bands.json"


class DefaultTileStore(TileStore):
    """Default TileStore implementation.

    It stores raster and vector data under the provided UPath.

    Raster data is always stored as a geo-referenced image, while vector data can use
    any provided VectorFormat. This is because for raster data we support reading in an
    arbitrary projection, but this is not supported in GeotiffRasterFormat, so we
    directly use rasterio to access the file.
    """

    def __init__(
        self,
        path_suffix: str = "tiles",
        convert_rasters_to_cogs: bool = True,
        tile_size: int = 256,
        geotiff_options: dict[str, Any] = {},
        vector_format: VectorFormat = GeojsonVectorFormat(),
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
            vector_format: format to use for storing vector data.
        """
        self.path_suffix = path_suffix
        self.convert_rasters_to_cogs = convert_rasters_to_cogs
        self.tile_size = tile_size
        self.geotiff_options = geotiff_options
        self.vector_format = vector_format

        self.path: UPath | None = None

    @override
    def set_dataset_path(self, ds_path: UPath) -> None:
        self.path = join_upath(ds_path, self.path_suffix)

    def _get_raster_dir(
        self, layer_name: str, item_name: str, bands: list[str], write: bool = False
    ) -> UPath:
        """Get the directory where the specified raster is stored.

        Args:
            layer_name: the name of the dataset layer.
            item_name: the name of the item from the data source.
            bands: list of band names that are expected to be stored together.
            write: whether to create the directory and write the bands to a file inside
                the directory.

        Returns:
            the UPath directory where the raster should be stored.
        """
        assert self.path is not None
        dir_name = self.path / layer_name / item_name / get_bandset_dirname(bands)

        if write:
            dir_name.mkdir(parents=True, exist_ok=True)
            with (dir_name / BANDS_FNAME).open("w") as f:
                json.dump(bands, f)

        return dir_name

    def _get_raster_fname(
        self, layer_name: str, item_name: str, bands: list[str]
    ) -> UPath:
        """Get the filename of the specified raster.

        Args:
            layer_name: the name of the dataset layer.
            item_name: the name of the item from the data source.
            bands: list of band names that are expected to be stored together.

        Returns:
            the UPath filename of the raster, which should be readable by rasterio.

        Raises:
            ValueError: if no file is found.
        """
        raster_dir = self._get_raster_dir(layer_name, item_name, bands)
        for fname in iter_nonhidden_files(raster_dir):
            # Ignore completed sentinel files, bands files, as well as temporary files created by
            # open_atomic (in case this tile store is on local filesystem).
            if fname.name == COMPLETED_FNAME:
                continue
            if fname.name == BANDS_FNAME:
                continue
            if fname.name == METADATA_FNAME:
                continue
            if ".tmp." in fname.name:
                continue
            return fname
        raise ValueError(f"no raster found in {raster_dir}")

    @override
    def is_raster_ready(
        self, layer_name: str, item_name: str, bands: list[str]
    ) -> bool:
        raster_dir = self._get_raster_dir(layer_name, item_name, bands)
        return (raster_dir / COMPLETED_FNAME).exists()

    @override
    def get_raster_bands(self, layer_name: str, item_name: str) -> list[list[str]]:
        assert isinstance(self.path, UPath)
        item_dir = self.path / layer_name / item_name
        if not item_dir.exists():
            return []

        bands: list[list[str]] = []
        for raster_dir in iter_nonhidden_subdirs(item_dir):
            if not (raster_dir / BANDS_FNAME).exists():
                # This is likely a legacy directory where the bands are only encoded in
                # the directory name, so we have to rely on that.
                parts = raster_dir.name.split("_")
                bands.append(parts)
                continue

            # We use the BANDS_FNAME here -- although it is slower to read the file, it
            # is more reliable since sometimes the directory name is a hash of the
            # bands in case there are too many bands (filename too long) or some bands
            # contain the underscore character.
            with (raster_dir / BANDS_FNAME).open() as f:
                bands.append(json.load(f))

        return bands

    @override
    def get_raster_bounds(
        self, layer_name: str, item_name: str, bands: list[str], projection: Projection
    ) -> PixelBounds:
        raster_fname = self._get_raster_fname(layer_name, item_name, bands)

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

    @override
    def read_raster(
        self,
        layer_name: str,
        item_name: str,
        bands: list[str],
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> RasterArray:
        raster_fname = self._get_raster_fname(layer_name, item_name, bands)
        return GeotiffRasterFormat().decode_raster(
            path=raster_fname.parent,
            fname=raster_fname.name,
            projection=projection,
            bounds=bounds,
            resampling=resampling,
        )

    @override
    def write_raster(
        self,
        layer_name: str,
        item_name: str,
        bands: list[str],
        projection: Projection,
        bounds: PixelBounds,
        raster: RasterArray,
    ) -> None:
        raster_dir = self._get_raster_dir(layer_name, item_name, bands, write=True)
        raster_format = GeotiffRasterFormat(geotiff_options=self.geotiff_options)
        raster_format.encode_raster(raster_dir, projection, bounds, raster)
        (raster_dir / COMPLETED_FNAME).touch()

    @override
    def write_raster_file(
        self,
        layer_name: str,
        item_name: str,
        bands: list[str],
        fname: UPath,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> None:
        raster_dir = self._get_raster_dir(layer_name, item_name, bands, write=True)
        raster_dir.mkdir(parents=True, exist_ok=True)

        if self.convert_rasters_to_cogs:
            with open_rasterio_upath_reader(fname) as src:
                profile = src.profile
                array = src.read()

                # If raster specifies ground control points, use WarpedVRT to get it in
                # an appropriate projection.
                # Previously we used rasterio.transform.from_gcps(gcps) but I think the
                # problem is that it computes one transform for the entire raster but
                # the raster might actually need warping.
                if profile["crs"] is None and src.gcps:
                    gcps, gcp_crs = src.gcps
                    # Use the first ground control point to pick a UTM/UPS projection.
                    first_gcp_orig = STGeometry(
                        Projection(gcp_crs, 1, 1),
                        shapely.Point(gcps[0].x, gcps[0].y),
                        None,
                    )
                    first_gcp_wgs84 = first_gcp_orig.to_projection(WGS84_PROJECTION)
                    crs = get_utm_ups_crs(first_gcp_wgs84.shp.x, first_gcp_wgs84.shp.y)
                    with rasterio.vrt.WarpedVRT(
                        src, crs=crs, resampling=Resampling.cubic
                    ) as vrt:
                        array = vrt.read()
                        transform = vrt.transform

                else:
                    crs = profile["crs"]
                    transform = profile["transform"]

            output_profile = {
                "driver": "GTiff",
                "compress": "lzw",
                "width": array.shape[2],
                "height": array.shape[1],
                "count": array.shape[0],
                "dtype": array.dtype.name,
                "crs": crs,
                "transform": transform,
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
                with open_atomic(dst_fname, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        if time_range is not None:
            # Add GeotiffRasterFormat metadata file so that GeotiffRasterFormat will
            # see it upon read and use it to obtain the time range information for the
            # raster.
            GeotiffRasterFormat.encode_metadata(
                raster_dir,
                GeotiffRasterMetadata(
                    num_channels=len(bands),
                    num_timesteps=1,
                    timestamps=[(time_range[0], time_range[1])],
                ),
            )

        (raster_dir / COMPLETED_FNAME).touch()

    def _get_vector_dir(self, layer_name: str, item_name: str) -> UPath:
        assert self.path is not None
        return self.path / layer_name / item_name

    @override
    def is_vector_ready(self, layer_name: str, item_name: str) -> bool:
        vector_dir = self._get_vector_dir(layer_name, item_name)
        return (vector_dir / COMPLETED_FNAME).exists()

    @override
    def read_vector(
        self,
        layer_name: str,
        item_name: str,
        projection: Projection,
        bounds: PixelBounds,
    ) -> list[Feature]:
        vector_dir = self._get_vector_dir(layer_name, item_name)
        return self.vector_format.decode_vector(vector_dir, projection, bounds)

    @override
    def write_vector(
        self, layer_name: str, item_name: str, features: list[Feature]
    ) -> None:
        vector_dir = self._get_vector_dir(layer_name, item_name)
        vector_dir.mkdir(parents=True, exist_ok=True)
        self.vector_format.encode_vector(vector_dir, features)
        (vector_dir / COMPLETED_FNAME).touch()
