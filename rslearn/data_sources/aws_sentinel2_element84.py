"""Data source for Sentinel-2 from public AWS bucket maintained by Element 84."""

import os
import tempfile
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import numpy as np
import numpy.typing as npt
import rasterio
import requests
from upath import UPath

from rslearn.data_sources.stac import SourceItem, StacDataSource
from rslearn.data_sources.tile_store_data_source import TileStoreDataSource
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils import STGeometry
from rslearn.utils.fsspec import join_upath
from rslearn.utils.raster_format import get_raster_projection_and_bounds

from .data_source import (
    DataSourceContext,
)

logger = get_logger(__name__)


class Sentinel2(TileStoreDataSource[SourceItem], StacDataSource):
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
        asset_bands: dict[str, list[str]]
        if context.layer_config is not None:
            asset_bands = {}
            for asset_key, band_names in self.ASSET_BANDS.items():
                # See if the bands provided by this asset intersect with the bands in
                # at least one configured band set.
                for band_set in context.layer_config.band_sets:
                    if not set(band_set.bands).intersection(set(band_names)):
                        continue
                    asset_bands[asset_key] = band_names
                    break
        elif assets is not None:
            asset_bands = {
                asset_key: self.ASSET_BANDS[asset_key] for asset_key in assets
            }
        else:
            asset_bands = dict(self.ASSET_BANDS)

        # Initialize TileStoreDataSource with asset_bands
        TileStoreDataSource.__init__(self, asset_bands=asset_bands)

        # Initialize StacDataSource
        StacDataSource.__init__(
            self,
            endpoint=self.STAC_ENDPOINT,
            collection_name=self.COLLECTION_NAME,
            query=query,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
            required_assets=list(asset_bands.keys()),
            cache_dir=cache_upath,
            properties_to_record=[self.HARMONIZE_PROPERTY_NAME],
        )

        self.harmonize = harmonize
        self.timeout = timeout

    # --- TileStoreDataSource implementation ---

    def get_asset_url(self, item_name: str, bands: list[str]) -> str:
        """Get the URL to read the asset for the given item and bands.

        Args:
            item_name: the name of the item.
            bands: the list of bands identifying which asset to get.

        Returns:
            the URL to read the asset from.
        """
        asset_key = self._get_asset_by_band(bands)
        item = self.get_item_by_name(item_name)
        return item.asset_urls[asset_key]

    def get_read_callback(
        self, item_name: str, bands: list[str]
    ) -> Callable[[npt.NDArray[Any]], npt.NDArray[Any]] | None:
        """Return a callback to harmonize Sentinel-2 data if needed.

        Args:
            item_name: the name of the item being read.
            bands: the bands being read.

        Returns:
            A callback function for harmonization, or None if not needed.
        """
        # Visual bands do not need harmonization.
        if not self.harmonize or bands == self.ASSET_BANDS["visual"]:
            return None

        item = self.get_item_by_name(item_name)
        return self._get_harmonize_callback(item)

    # --- Harmonization helpers ---

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
