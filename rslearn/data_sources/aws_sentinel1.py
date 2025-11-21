"""Data source for Sentinel-1 on AWS."""

import os
import tempfile
from typing import Any

import boto3
from upath import UPath

from rslearn.data_sources.copernicus import (
    CopernicusItem,
    Sentinel1OrbitDirection,
    Sentinel1Polarisation,
    Sentinel1ProductType,
)
from rslearn.data_sources.copernicus import Sentinel1 as CopernicusSentinel1
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStore, TileStoreWithLayer
from rslearn.utils.geometry import STGeometry

from .data_source import DataSource, DataSourceContext, QueryConfig

WRS2_GRID_SIZE = 1.0

logger = get_logger(__name__)


class Sentinel1(DataSource, TileStore):
    """A data source for Sentinel-1 GRD imagery on AWS.

    Specifically, uses the sentinel-s1-l1c S3 bucket maintained by Sinergise. See
    https://aws.amazon.com/marketplace/pp/prodview-uxrsbvhd35ifw for details about the
    bucket.

    We use the Copernicus API for metadata search. So the bucket is only used for
    downloading the images.

    Currently, it only supports GRD IW DV scenes.
    """

    bucket_name = "sentinel-s1-l1c"
    bands = ["vv", "vh"]

    def __init__(
        self,
        orbit_direction: Sentinel1OrbitDirection | None = None,
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Initialize a new Sentinel1 instance.

        Args:
            orbit_direction: optional orbit direction to filter by.
            context: the data source context.
        """
        self.client = boto3.client("s3")
        self.bucket = boto3.resource("s3").Bucket(self.bucket_name)
        self.sentinel1 = CopernicusSentinel1(
            product_type=Sentinel1ProductType.IW_GRDH,
            polarisation=Sentinel1Polarisation.VV_VH,
            orbit_direction=orbit_direction,
        )

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[CopernicusItem]]]:
        """Get a list of items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        return self.sentinel1.get_items(geometries, query_config)

    def get_item_by_name(self, name: str) -> CopernicusItem:
        """Gets an item by name."""
        return self.sentinel1.get_item_by_name(name)

    def deserialize_item(self, serialized_item: Any) -> CopernicusItem:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return CopernicusItem.deserialize(serialized_item)

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[CopernicusItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        for item in items:
            for band in self.bands:
                band_names = [band]
                if tile_store.is_raster_ready(item.name, band_names):
                    continue

                # Item name is like "S1C_IW_GRDH_1SDV_20250528T172106_20250528T172131_002534_00545C_B433.SAFE".
                item_name_prefix = item.name.split(".")[0]
                time_str = item_name_prefix.split("_")[4]
                if len(time_str) != 15:
                    raise ValueError(
                        f"expected 15-character time string but got {time_str}"
                    )
                # We convert to int here since path in bucket isn't padded with leading 0s.
                year = int(time_str[0:4])
                month = int(time_str[4:6])
                day = int(time_str[6:8])
                blob_path = f"GRD/{year}/{month}/{day}/IW/DV/{item_name_prefix}/measurement/iw-{band}.tiff"

                with tempfile.TemporaryDirectory() as tmp_dir:
                    fname = os.path.join(tmp_dir, f"{band}.tif")
                    try:
                        self.bucket.download_file(
                            blob_path,
                            fname,
                            ExtraArgs={"RequestPayer": "requester"},
                        )
                    except:
                        logger.error(
                            f"encountered error while downloading s3://{self.bucket_name}/{blob_path}"
                        )
                        raise
                    tile_store.write_raster_file(item.name, band_names, UPath(fname))
