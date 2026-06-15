"""Data source for the ESA CCI above-ground biomass (AGB) map.

The AGB maps are generated primarily from SAR -- ESA Sentinel-1 (C-band) and JAXA
ALOS PALSAR / PALSAR-2 (L-band) -- so this is a strong supervision/decode target for
encouraging a model to use SAR rather than relying on optical bands alone.

The data is the official ESA CCI product, distributed by the NERC EDS Centre for
Environmental Data Analysis (CEDA), publicly accessible over HTTPS without
authentication. It is provided as 10x10 degree GeoTIFF tiles at 100 m resolution, with
one map per epoch year. This source models the data like ``usda_cdl.CDL``: there is one
item per (year, tile), each carrying the tile's spatial bounds and a one-year time
range, so a window's time range selects the matching biomass year via the layer's
``QueryConfig``.

See https://climate.esa.int/en/projects/biomass/ and
https://catalogue.ceda.ac.uk/uuid/95913ffb6467447ca72c4e9d8cf30501/ (v6.0).
"""

import os
import tempfile
from datetime import UTC, datetime, timedelta

import requests
import shapely
from upath import UPath

from rslearn.config import QueryConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource, DataSourceContext, Item
from rslearn.data_sources.utils import MatchedItemGroup, match_candidate_items_to_window
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.geometry import STGeometry

logger = get_logger(__name__)

# Tiles span 10 degrees and are named by their NW (top-left) corner, e.g. "N40E000"
# covers latitude [30, 40] and longitude [0, 10] (verified against the published tiles).
TILE_DEGREES = 10


class CCIBiomass(DataSource):
    """Data source for the ESA CCI above-ground biomass (AGB) map.

    AGB (Mg/ha) is a SAR-derived, continuous global product at 100 m, with annual maps.
    There is one item per (year, 10x10 degree tile); the per-pixel uncertainty
    (band ``agb_sd``) can optionally be ingested as a second band. Ocean/missing tiles
    (which return HTTP 404) are skipped at ingest time.

    See https://climate.esa.int/en/projects/biomass/.
    """

    # Official ESA CCI distribution on CEDA (public, no auth).
    BASE_URL = "https://dap.ceda.ac.uk/neodc/esacci/biomass/data/agb/maps"
    VERSION = "v6.0"
    FILE_VERSION = "fv6.0"
    # Epoch years available in v6.0.
    AVAILABLE_YEARS = (2007, 2010, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022)

    AGB_BAND = "agb"
    AGB_SD_BAND = "agb_sd"

    def __init__(
        self,
        years: list[int] | None = None,
        include_uncertainty: bool = False,
        timeout: timedelta = timedelta(seconds=60),
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Create a new CCIBiomass.

        Args:
            years: the epoch years to expose for matching (subset of AVAILABLE_YEARS).
                Defaults to all available years, so a window's time range selects the
                matching map year.
            include_uncertainty: if True, also ingest the per-pixel AGB standard
                deviation as a second band ("agb_sd").
            timeout: timeout for HTTP requests.
            context: the data source context.
        """
        if years is None:
            years = list(self.AVAILABLE_YEARS)
        unknown = [y for y in years if y not in self.AVAILABLE_YEARS]
        if unknown:
            raise ValueError(
                f"years {unknown} not in available CCI biomass {self.VERSION} years "
                f"{self.AVAILABLE_YEARS}"
            )
        self.years = sorted(years)
        self.include_uncertainty = include_uncertainty
        self.timeout = timeout

    # --- tile <-> name/bounds helpers ---

    @staticmethod
    def _tile_name(top_lat: int, left_lon: int) -> str:
        """Tile name from its NW corner, e.g. (40, 0) -> 'N40E000'."""
        lat_str = f"N{top_lat:02d}" if top_lat >= 0 else f"S{abs(top_lat):02d}"
        lon_str = f"E{left_lon:03d}" if left_lon >= 0 else f"W{abs(left_lon):03d}"
        return f"{lat_str}{lon_str}"

    @staticmethod
    def _tile_bounds(top_lat: int, left_lon: int) -> shapely.Geometry:
        """WGS84 box for a tile given its NW corner: [lat-10, lat] x [lon, lon+10]."""
        return shapely.box(
            left_lon, top_lat - TILE_DEGREES, left_lon + TILE_DEGREES, top_lat
        )

    @staticmethod
    def _parse_tile(tile: str) -> tuple[int, int]:
        """Parse a tile name (e.g. 'S10W050') back to its (top_lat, left_lon) corner."""
        top_lat = int(tile[1:3]) * (1 if tile[0] == "N" else -1)
        left_lon = int(tile[4:7]) * (1 if tile[3] == "E" else -1)
        return top_lat, left_lon

    def _year_time_range(self, year: int) -> tuple[datetime, datetime]:
        """Validity interval for a map year, tiling the timeline with no gaps.

        rslearn only supports ``TimeMode.WITHIN`` (a window matches an item only if
        their time ranges overlap), and the available years are sparse (2007, 2010,
        then annual 2015-2022). To let every window pull the *nearest* available map,
        each year owns the interval bounded by the midpoints to its neighbouring years
        (open-ended at the two extremes). Biomass changes slowly, so snapping a window
        to the nearest epoch is reasonable.
        """
        idx = self.years.index(year)
        if idx == 0:
            start = datetime(1970, 1, 1, tzinfo=UTC)
        else:
            prev_year = self.years[idx - 1]
            start = datetime((prev_year + year) // 2 + 1, 1, 1, tzinfo=UTC)
        if idx == len(self.years) - 1:
            end = datetime(2100, 1, 1, tzinfo=UTC)
        else:
            next_year = self.years[idx + 1]
            end = datetime((year + next_year) // 2 + 1, 1, 1, tzinfo=UTC)
        return start, end

    def _item_for(self, year: int, top_lat: int, left_lon: int) -> Item:
        """Build the Item for a (year, tile): tile bounds + the year's validity range."""
        tile = self._tile_name(top_lat, left_lon)
        geometry = STGeometry(
            WGS84_PROJECTION,
            self._tile_bounds(top_lat, left_lon),
            self._year_time_range(year),
        )
        # Name encodes year and tile so ingest can reconstruct the download URL.
        return Item(f"{year}_{tile}", geometry)

    def _enumerate_items(self) -> list[Item]:
        """All (year, tile) candidate items over the global 10-degree grid."""
        items: list[Item] = []
        for year in self.years:
            # top_lat is the tile's north edge: 90 (covers 80-90) down to -80 (covers
            # -90..-80). left_lon is the west edge. Ocean tiles are skipped at ingest.
            for top_lat in range(90, -TILE_DEGREES * 8 - 1, -TILE_DEGREES):
                for left_lon in range(-180, 180, TILE_DEGREES):
                    items.append(self._item_for(year, top_lat, left_lon))
        return items

    def _download_filename(self, year: int, tile: str, product: str) -> str:
        """GeoTIFF filename for a (year, tile, product in {AGB, AGB_SD})."""
        return (
            f"{tile}_ESACCI-BIOMASS-L4-{product}-MERGED-100m-"
            f"{year}-{self.FILE_VERSION}.tif"
        )

    # --- DataSource implementation ---

    def get_item_by_name(self, name: str) -> Item:
        """Gets an item by name (format '<year>_<tile>', e.g. '2020_N40E000')."""
        year = int(name.split("_")[0])
        tile = name.split("_")[1]
        top_lat, left_lon = self._parse_tile(tile)
        return self._item_for(year, top_lat, left_lon)

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[MatchedItemGroup[Item]]]:
        """Get items intersecting each geometry (spatially by tile, temporally by year).

        Args:
            geometries: the spatiotemporal geometries.
            query_config: the query configuration (its time mode selects the year).

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        items = self._enumerate_items()
        groups = []
        for geometry in geometries:
            cur_groups = match_candidate_items_to_window(geometry, items, query_config)
            groups.append(cur_groups)
        return groups

    def deserialize_item(self, serialized_item: dict) -> Item:
        """Deserializes an item from JSON-decoded data."""
        return Item.deserialize(serialized_item)

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into.
            items: the items to ingest.
            geometries: a list of geometries needed for each item.
        """
        products = [(self.AGB_BAND, "AGB")]
        if self.include_uncertainty:
            products.append((self.AGB_SD_BAND, "AGB_SD"))

        for item in items:
            year = int(item.name.split("_")[0])
            tile = item.name.split("_")[1]

            for band, product in products:
                if tile_store.is_raster_ready(item, [band]):
                    continue

                fname = self._download_filename(year, tile, product)
                url = f"{self.BASE_URL}/{self.VERSION}/geotiff/{year}/{fname}"
                with requests.get(
                    url, stream=True, timeout=self.timeout.total_seconds()
                ) as response:
                    # Ocean / non-land tiles simply don't exist: skip them.
                    if response.status_code == 404:
                        logger.debug(
                            "CCI biomass tile %s missing (404), skipping", fname
                        )
                        continue
                    response.raise_for_status()

                    with tempfile.TemporaryDirectory() as tmp_dir:
                        local_fname = os.path.join(tmp_dir, fname)
                        with open(local_fname, "wb") as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)

                        logger.debug("ingesting CCI biomass %s band %s", fname, band)
                        tile_store.write_raster_file(
                            item,
                            [band],
                            UPath(local_fname),
                            time_range=item.geometry.time_range,
                        )
