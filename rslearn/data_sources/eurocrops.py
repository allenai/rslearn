"""Data source for vector EuroCrops crop type data."""

import glob
import os
import tempfile
import zipfile
from datetime import UTC, datetime, timedelta
from typing import Any

import fiona
import requests
from rasterio.crs import CRS
from upath import UPath

from rslearn.config import QueryConfig, VectorLayerConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource, Item
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import Projection, STGeometry, get_global_geometry

logger = get_logger(__name__)


class EuroCropsItem(Item):
    """An item in the EuroCrops data source.

    For simplicity, we have just one item per year, so each item combines all of the
    country-level files for that year.
    """

    def __init__(self, name: str, geometry: STGeometry, zip_fnames: list[str]):
        """Creates a new EuroCropsItem.

        Args:
            name: unique name of the item. It is just the year that this item
                corresponds to.
            geometry: the spatial and temporal extent of the item
            zip_fnames: the filenames of the zip files that contain country-level crop
                type data for this year.
        """
        super().__init__(name, geometry)
        self.zip_fnames = zip_fnames

    def serialize(self) -> dict:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["zip_fnames"] = self.zip_fnames
        return d

    @staticmethod
    def deserialize(d: dict) -> "EuroCropsItem":
        """Deserializes an item from a JSON-decoded dictionary."""
        item = super(EuroCropsItem, EuroCropsItem).deserialize(d)
        return EuroCropsItem(
            name=item.name, geometry=item.geometry, zip_fnames=d["zip_fnames"]
        )


class EuroCrops(DataSource[EuroCropsItem]):
    """A data source for EuroCrops vector data (v11).

    See https://zenodo.org/records/14094196 for details.

    While the source data is split into country-level files, this data source uses one
    item per year for simplicity. So each item corresponds to all of the country-level
    files for that year.

    Note that the RO_ny.zip file is not used.
    """

    BASE_URL = "https://zenodo.org/records/14094196/files/"
    FILENAMES_BY_YEAR = {
        2018: [
            "FR_2018.zip",
        ],
        2019: [
            "DK_2019.zip",
        ],
        2020: [
            "ES_NA_2020.zip",
            "FI_2020.zip",
            "HR_2020.zip",
            "NL_2020.zip",
        ],
        2021: [
            "AT_2021.zip",
            "BE_VLG_2021.zip",
            "BE_WAL_2021.zip",
            "EE_2021.zip",
            "LT_2021.zip",
            "LV_2021.zip",
            "PT_2021.zip",
            "SE_2021.zip",
            "SI_2021.zip",
            "SK_2021.zip",
        ],
        2023: [
            "CZ_2023.zip",
            "DE_BB_2023.zip",
            "DE_LS_2021.zip",
            "DE_NRW_2021.zip",
            "ES_2023.zip",
            "IE_2023.zip",
        ],
    }
    TIMEOUT = timedelta(seconds=10)

    @staticmethod
    def from_config(config: VectorLayerConfig, ds_path: UPath) -> "EuroCrops":
        """Creates a new EuroCrops instance from a configuration dictionary."""
        if config.data_source is None:
            raise ValueError("data_source is required")
        return EuroCrops()

    def _get_all_items(self) -> list[EuroCropsItem]:
        """Get a list of all available items in the data source."""
        items: list[EuroCropsItem] = []
        for year, fnames in self.FILENAMES_BY_YEAR.items():
            items.append(
                EuroCropsItem(
                    str(year),
                    get_global_geometry(
                        time_range=(
                            datetime(year, 1, 1, tzinfo=UTC),
                            datetime(year + 1, 1, 1, tzinfo=UTC),
                        ),
                    ),
                    fnames,
                )
            )
        return items

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[EuroCropsItem]]]:
        """Get a list of items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        wgs84_geometries = [
            geometry.to_projection(WGS84_PROJECTION) for geometry in geometries
        ]
        all_items = self._get_all_items()
        groups = []
        for geometry in wgs84_geometries:
            cur_groups = match_candidate_items_to_window(
                geometry, all_items, query_config
            )
            groups.append(cur_groups)
        return groups

    def deserialize_item(self, serialized_item: Any) -> EuroCropsItem:
        """Deserializes an item from JSON-decoded data."""
        return EuroCropsItem.deserialize(serialized_item)

    def _extract_features(self, fname: str) -> list[Feature]:
        """Download the given zip file, extract shapefile, and return list of features."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download the zip file.
            url = self.BASE_URL + fname
            logger.debug(f"Downloading zip file from {url}")
            response = requests.get(
                url,
                stream=True,
                timeout=self.TIMEOUT.total_seconds(),
                allow_redirects=False,
            )
            response.raise_for_status()
            zip_fname = os.path.join(tmp_dir, "data.zip")
            with open(zip_fname, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Extract all of the files and look for shapefile filename.
            logger.debug(f"Extracting zip file {fname}")
            with zipfile.ZipFile(zip_fname) as zip_f:
                zip_f.extractall(path=tmp_dir)

            # The shapefiles can appear at any of three levels in the file hierarchy.
            shp_fnames = (
                glob.glob("*.shp", root_dir=tmp_dir)
                + glob.glob("*/*.shp", root_dir=tmp_dir)
                + glob.glob("*/*/*.shp", root_dir=tmp_dir)
            )
            if len(shp_fnames) == 0:
                tmp_dir_fnames = os.listdir(tmp_dir)
                raise ValueError(
                    f"expected {fname} to contain .shp file but none found (matches={shp_fnames}, ls={tmp_dir_fnames})"
                )

            # Load the features from the shapefile(s).
            features = []
            for shp_fname in shp_fnames:
                logger.debug(f"Loading feature list from {shp_fname}")
                with fiona.open(os.path.join(tmp_dir, shp_fname)) as src:
                    crs = CRS.from_wkt(src.crs.to_wkt())
                    # Normal GeoJSON should have coordinates in CRS coordinates, i.e. it
                    # should be 1 projection unit/pixel.
                    projection = Projection(crs, 1, 1)

                    for feat in src:
                        features.append(
                            Feature.from_geojson(
                                projection,
                                {
                                    "type": "Feature",
                                    "geometry": dict(feat.geometry),
                                    "properties": dict(feat.properties),
                                },
                            )
                        )

            return features

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[EuroCropsItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        for item in items:
            if tile_store.is_vector_ready(item.name):
                continue

            # Get features across all shapefiles.
            features: list[Feature] = []
            for fname in item.zip_fnames:
                logger.debug(f"Getting features from {fname} for item {item.name}")
                features.extend(self._extract_features(fname))

            logger.debug(f"Writing features for {item.name} to the tile store")
            tile_store.write_vector(item.name, features)
