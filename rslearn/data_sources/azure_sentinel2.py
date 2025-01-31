"""Sentinel-2 on Planetary Computer."""

import os
import tempfile
import xml.etree.ElementTree as ET
from typing import Any

import affine
import numpy.typing as npt
import planetary_computer
import pystac
import pystac_client
import rasterio
import requests
import shapely
from rasterio.enums import Resampling
from upath import UPath

from rslearn.config import LayerConfig, QueryConfig, RasterLayerConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource, Item
from rslearn.data_sources.raster_source import is_raster_needed
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.dataset import Window
from rslearn.dataset.materialize import RasterMaterializer
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStore, TileStoreWithLayer
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.raster_format import get_raster_projection_and_bounds

from .copernicus import get_harmonize_callback

logger = get_logger(__name__)


class Sentinel2(DataSource, TileStore):
    """A data source for Sentinel-2 L2A data on Microsoft Planetary Computer.

    See https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a.

    The PC_SDK_SUBSCRIPTION_KEY environment variable can be set but is not needed.
    """

    STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"

    COLLECTION_NAME = "sentinel-2-l2a"

    BANDS = {
        "B01": ["B01"],
        "B02": ["B02"],
        "B03": ["B03"],
        "B04": ["B04"],
        "B05": ["B05"],
        "B06": ["B06"],
        "B07": ["B07"],
        "B08": ["B08"],
        "B09": ["B09"],
        "B11": ["B11"],
        "B12": ["B12"],
        "B8A": ["B8A"],
        "visual": ["R", "G", "B"],
    }

    def __init__(
        self,
        config: RasterLayerConfig,
        harmonize: bool = False,
        query: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_ascending: bool = True,
        timeout: int = 10,
    ):
        """Initialize a new Sentinel2 instance.

        Args:
            config: the LayerConfig of the layer containing this data source.
            harmonize: harmonize pixel values across different processing baselines,
                see https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
            query: optional query argument to STAC searches.
            sort_by: sort by this property in the STAC items.
            sort_ascending: whether to sort ascending (or descending).
            timeout: timeout for API requests in seconds.
        """
        self.config = config
        self.harmonize = harmonize
        self.query = query
        self.sort_by = sort_by
        self.sort_ascending = sort_ascending
        self.timeout = timeout

        self.client = pystac_client.Client.open(
            self.STAC_ENDPOINT, modifier=planetary_computer.sign_inplace
        )
        self.collection = self.client.get_collection(self.COLLECTION_NAME)

    @staticmethod
    def from_config(config: RasterLayerConfig, ds_path: UPath) -> "Sentinel2":
        """Creates a new Sentinel2 instance from a configuration dictionary."""
        if config.data_source is None:
            raise ValueError("config.data_source is required")
        d = config.data_source.config_dict
        kwargs: dict[str, Any] = dict(
            config=config,
        )

        simple_optionals = [
            "harmonize",
            "query",
            "sort_by",
            "sort_ascending",
            "timeout",
        ]
        for k in simple_optionals:
            if k in d:
                kwargs[k] = d[k]

        return Sentinel2(**kwargs)

    def _stac_item_to_item(self, stac_item: pystac.Item) -> Item:
        shp = shapely.geometry.shape(stac_item.geometry)

        # Get time range.
        metadata = stac_item.common_metadata
        if metadata.start_datetime is not None and metadata.end_datetime is not None:
            time_range = (
                metadata.start_datetime,
                metadata.end_datetime,
            )
        elif stac_item.datetime is not None:
            time_range = (stac_item.datetime, stac_item.datetime)
        else:
            raise ValueError(
                f"item {stac_item.id} unexpectedly missing start_datetime, end_datetime, and datetime"
            )

        geom = STGeometry(WGS84_PROJECTION, shp, time_range)
        return Item(stac_item.id, geom)

    def get_item_by_name(self, name: str) -> Item:
        """Gets an item by name.

        Args:
            name: the name of the item to get

        Returns:
            the item object
        """
        stac_item = self.collection.get_item(name)
        return self._stac_item_to_item(stac_item)

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[Item]]]:
        """Get a list of items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        groups = []
        for geometry in geometries:
            # Get potentially relevant items from the collection by performing one search
            # for each requested geometry.
            wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
            logger.debug("performing STAC search for geometry %s", wgs84_geometry)
            result = self.client.search(
                collections=[self.COLLECTION_NAME],
                intersects=shapely.to_geojson(wgs84_geometry.shp),
                datetime=wgs84_geometry.time_range,
                query=self.query,
            )
            stac_items = [item for item in result.item_collection()]
            logger.debug("STAC search yielded %d items", len(stac_items))

            if self.sort_by is not None:
                stac_items.sort(
                    key=lambda stac_item: stac_item.properties[self.sort_by],
                    reverse=not self.sort_ascending,
                )

            candidate_items = [
                self._stac_item_to_item(stac_item) for stac_item in stac_items
            ]
            cur_groups = match_candidate_items_to_window(
                geometry, candidate_items, query_config
            )
            groups.append(cur_groups)

        return groups

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return Item.deserialize(serialized_item)

    def _get_product_xml(self, stac_item: pystac.Item) -> ET.Element:
        asset_url = stac_item.assets["product-metadata"].href
        response = requests.get(asset_url, timeout=self.timeout)
        response.raise_for_status()
        return ET.fromstring(response.content)

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        for item in items:
            stac_item = self.collection.get_item(item.name)

            for asset_key, band_names in self.BANDS.items():
                if not is_raster_needed(band_names, self.config.band_sets):
                    continue
                if tile_store.is_raster_ready(item.name, band_names):
                    continue

                asset_url = stac_item.assets[asset_key].href

                with tempfile.TemporaryDirectory() as tmp_dir:
                    fname = os.path.join(tmp_dir, f"{asset_key}.tif")
                    logger.debug(
                        "azure_sentinel2 start downloading item %s asset %s",
                        item.name,
                        asset_key,
                    )
                    with requests.get(
                        asset_url, stream=True, timeout=self.timeout
                    ) as r:
                        r.raise_for_status()
                        with open(fname, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)

                    logger.debug(
                        "azure_sentinel2 ingest item %s asset %s", item.name, asset_key
                    )

                    # Harmonize values if needed.
                    # TCI does not need harmonization.
                    harmonize_callback = None
                    if self.harmonize and asset_key != "visual":
                        harmonize_callback = get_harmonize_callback(
                            self._get_product_xml(stac_item)
                        )

                    if harmonize_callback is not None:
                        # In this case we need to read the array, convert the pixel
                        # values, and pass modified array directly to the TileStore.
                        with rasterio.open(fname) as src:
                            array = src.read()
                            projection, bounds = get_raster_projection_and_bounds(src)
                        array = harmonize_callback(array)
                        tile_store.write_raster(
                            item.name, band_names, projection, bounds, array
                        )

                    else:
                        tile_store.write_raster_file(
                            item.name, band_names, UPath(fname)
                        )

                logger.debug(
                    "azure_sentinel2 done ingesting item %s asset %s",
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
        return list(self.BANDS.values())

    def _get_asset_by_band(self, bands: list[str]) -> str:
        # Get the name of the asset based on the bands.
        for asset_key, asset_bands in self.BANDS.items():
            if bands == asset_bands:
                return asset_key

        raise ValueError(f"no raster with bands {bands}")

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
        stac_item = self.collection.get_item(item_name)
        asset_url = stac_item.assets[asset_key].href

        # Construct the transform to use for the warped dataset.
        wanted_transform = affine.Affine(
            projection.x_resolution,
            0,
            bounds[0] * projection.x_resolution,
            0,
            projection.y_resolution,
            bounds[1] * projection.y_resolution,
        )

        with rasterio.open(asset_url) as src:
            with rasterio.vrt.WarpedVRT(
                src,
                crs=projection.crs,
                transform=wanted_transform,
                width=bounds[2] - bounds[0],
                height=bounds[3] - bounds[1],
                resampling=resampling,
            ) as vrt:
                return vrt.read()

    def materialize(
        self,
        window: Window,
        item_groups: list[list[Item]],
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
        assert isinstance(layer_cfg, RasterLayerConfig)
        RasterMaterializer().materialize(
            TileStoreWithLayer(self, layer_name),
            window,
            layer_name,
            layer_cfg,
            item_groups,
        )
