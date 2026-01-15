"""Data source for Sentinel-2 from public AWS bucket maintained by Element 84."""

import os
import tempfile
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import affine
import numpy as np
import numpy.typing as npt
import rasterio
import requests
from rasterio.enums import Resampling
from upath import UPath

from rslearn.config import LayerConfig
from rslearn.data_sources.stac import SourceItem, StacDataSource
from rslearn.dataset import Window
from rslearn.dataset.manage import RasterMaterializer
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStore, TileStoreWithLayer
from rslearn.utils import Projection, STGeometry
from rslearn.utils.fsspec import join_upath
from rslearn.utils.geometry import PixelBounds
from rslearn.utils.raster_format import get_raster_projection_and_bounds

from .data_source import (
    DataSourceContext,
)

logger = get_logger(__name__)


class Sentinel2(StacDataSource, TileStore):
    """A data source for Sentinel-2 L2A imagery on AWS from s3://sentinel-cogs.

    The S3 bucket has COGs so this data source supports direct materialization. It also
    allows anonymous free access, so no credentials are needed.

    See https://aws.amazon.com/marketplace/pp/prodview-ykj5gyumkzlme for details.
    """

    STAC_ENDPOINT = "https://earth-search.aws.element84.com/v1"
    COLLECTION_NAME = "sentinel-2-l2a"
    ASSET_BANDS = {
        "coastal": ["B01"],
        "blue": ["B02"],
        "green": ["B03"],
        "red": ["B04"],
        "rededge1": ["B05"],
        "rededge2": ["B06"],
        "rededge3": ["B07"],
        "nir": ["B08"],
        "nir09": ["B09"],
        "swir16": ["B11"],
        "swir22": ["B12"],
        "nir08": ["B8A"],
        "visual": ["R", "G", "B"],
    }
    HARMONIZE_OFFSET = -1000
    HARMONIZE_PROPERTY_NAME = "earthsearch:boa_offset_applied"

    def __init__(
        self,
        assets: list[str] | None = None,
        query: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_ascending: bool = True,
        cache_dir: str | None = None,
        harmonize: bool = False,
        timeout: timedelta = timedelta(seconds=10),
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Initialize a new Sentinel2 instance.

        Args:
            assets: only ingest these asset names. This is only used if context.layer_config is not set.
                If neither assets nor context.layer_config is set, then all assets are ingested.
            query: optional STAC query filter to use.
            sort_by: STAC item property to sort by. For example, use "eo:cloud_cover" to sort by cloud cover.
            sort_ascending: whether to sort ascending or descending.
            cache_dir: directory to cache discovered items.
            harmonize: harmonize pixel values across different processing baselines,
                see https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
            timeout: timeout to use for requests.
            context: the data source context.
        """  # noqa: E501
        # Determine the cache_upath to use.
        cache_upath: UPath | None = None
        if cache_dir is not None:
            if context.ds_path is not None:
                cache_upath = join_upath(context.ds_path, cache_dir)
            else:
                cache_upath = UPath(cache_dir)

            cache_upath.mkdir(parents=True, exist_ok=True)

        # Determine which assets we need based on the bands in the layer config.
        self.asset_bands: dict[str, list[str]]
        if context.layer_config is not None:
            self.asset_bands = {}
            for asset_key, band_names in self.ASSET_BANDS.items():
                # See if the bands provided by this asset intersect with the bands in
                # at least one configured band set.
                for band_set in context.layer_config.band_sets:
                    if not set(band_set.bands).intersection(set(band_names)):
                        continue
                    self.asset_bands[asset_key] = band_names
                    break
        elif assets is not None:
            self.asset_bands = {
                asset_key: self.ASSET_BANDS[asset_key] for asset_key in assets
            }
        else:
            self.asset_bands = self.ASSET_BANDS

        super().__init__(
            endpoint=self.STAC_ENDPOINT,
            collection_name=self.COLLECTION_NAME,
            query=query,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
            required_assets=list(self.asset_bands.keys()),
            cache_dir=cache_upath,
            properties_to_record=[self.HARMONIZE_PROPERTY_NAME],
        )

        self.harmonize = harmonize
        self.timeout = timeout

    def _get_harmonize_callback(
        self, item: SourceItem
    ) -> Callable[[npt.NDArray], npt.NDArray] | None:
        """Get the harmonization callback to remove offset for newly processed scenes.

        We do not use copernicus.get_harmonize_callback here because the S3 bucket does
        not seem to provide the product metadata XML file. So instead we check the
        earthsearch:boa_offset_applied property on the item.
        """
        if not item.properties[self.HARMONIZE_PROPERTY_NAME]:
            # This means no offset was applied so we don't need to subtract it.
            return None

        def harmonize_callback(array: npt.NDArray) -> npt.NDArray:
            # We assume the offset is -1000 since that is the standard.
            # To work with uint16 array, we clip to 1000+ and then subtract 1000.
            assert array.shape[0] == 1 and array.dtype == np.uint16
            return np.clip(array, -self.HARMONIZE_OFFSET, None) - (
                -self.HARMONIZE_OFFSET
            )

        return harmonize_callback

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[SourceItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        for item in items:
            for asset_key, band_names in self.asset_bands.items():
                if asset_key not in item.asset_urls:
                    continue
                if tile_store.is_raster_ready(item.name, band_names):
                    continue

                asset_url = item.asset_urls[asset_key]

                with tempfile.TemporaryDirectory() as tmp_dir:
                    local_fname = os.path.join(tmp_dir, f"{asset_key}.tif")
                    logger.debug(
                        "Download item %s asset %s to %s",
                        item.name,
                        asset_key,
                        local_fname,
                    )
                    with requests.get(
                        asset_url, stream=True, timeout=self.timeout.total_seconds()
                    ) as r:
                        r.raise_for_status()
                        with open(local_fname, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)

                    logger.debug(
                        "Ingest item %s asset %s",
                        item.name,
                        asset_key,
                    )

                    # Harmonize values if needed.
                    # TCI does not need harmonization.
                    harmonize_callback = None
                    if self.harmonize and asset_key != "visual":
                        harmonize_callback = self._get_harmonize_callback(item)

                    if harmonize_callback is not None:
                        # In this case we need to read the array, convert the pixel
                        # values, and pass modified array directly to the TileStore.
                        with rasterio.open(local_fname) as src:
                            array = src.read()
                            projection, bounds = get_raster_projection_and_bounds(src)
                        array = harmonize_callback(array)
                        tile_store.write_raster(
                            item.name, band_names, projection, bounds, array
                        )

                    else:
                        tile_store.write_raster_file(
                            item.name, band_names, UPath(local_fname)
                        )

                logger.debug(
                    "Done ingesting item %s asset %s",
                    item.name,
                    asset_key,
                )

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
        # Always ready since we wrap accesses to underlying API.
        return True

    def get_raster_bands(self, layer_name: str, item_name: str) -> list[list[str]]:
        """Get the sets of bands that have been stored for the specified item.

        Args:
            layer_name: the layer name or alias.
            item_name: the item.

        Returns:
            a list of lists of bands that are in the tile store (with one raster
                stored corresponding to each inner list). If no rasters are ready for
                this item, returns empty list.
        """
        return list(self.asset_bands.values())

    def _get_asset_by_band(self, bands: list[str]) -> str:
        """Get the name of the asset based on the band names."""
        for asset_key, asset_bands in self.asset_bands.items():
            if bands == asset_bands:
                return asset_key

        raise ValueError(f"no known asset with bands {bands}")

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
        item = self.get_item_by_name(item_name)
        geom = item.geometry.to_projection(projection)
        return (
            int(geom.shp.bounds[0]),
            int(geom.shp.bounds[1]),
            int(geom.shp.bounds[2]),
            int(geom.shp.bounds[3]),
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
            resampling: the resampling method to use in case reprojection is needed.

        Returns:
            the raster data
        """
        asset_key = self._get_asset_by_band(bands)
        item = self.get_item_by_name(item_name)
        asset_url = item.asset_urls[asset_key]

        # Construct the transform to use for the warped dataset.
        wanted_transform = affine.Affine(
            projection.x_resolution,
            0,
            bounds[0] * projection.x_resolution,
            0,
            projection.y_resolution,
            bounds[1] * projection.y_resolution,
        )

        # Read from the raster under the specified projection/bounds.
        with rasterio.open(asset_url) as src:
            with rasterio.vrt.WarpedVRT(
                src,
                crs=projection.crs,
                transform=wanted_transform,
                width=bounds[2] - bounds[0],
                height=bounds[3] - bounds[1],
                resampling=resampling,
            ) as vrt:
                raw_data = vrt.read()

        # We can return the data now if harmonization is not needed.
        if not self.harmonize or bands == self.ASSET_BANDS["visual"]:
            return raw_data

        # Otherwise we apply the harmonize_callback.
        item = self.get_item_by_name(item_name)
        harmonize_callback = self._get_harmonize_callback(item)

        if harmonize_callback is None:
            return raw_data

        array = harmonize_callback(raw_data)
        return array

    def materialize(
        self,
        window: Window,
        item_groups: list[list[SourceItem]],
        layer_name: str,
        layer_cfg: LayerConfig,
    ) -> None:
        """Materialize data for the window.

        Args:
            window: the window to materialize
            item_groups: the items from get_items
            layer_name: the name of this layer
            layer_cfg: the config of this layer
        """
        RasterMaterializer().materialize(
            TileStoreWithLayer(self, layer_name),
            window,
            layer_name,
            layer_cfg,
            item_groups,
        )
