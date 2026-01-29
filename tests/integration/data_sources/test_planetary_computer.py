import pathlib
from datetime import UTC, datetime

import pytest
import shapely
from rasterio import CRS
from upath import UPath

from rslearn.config import (
    QueryConfig,
    SpaceMode,
)
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.planetary_computer import CopDemGlo30, Sentinel1, Sentinel2
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils import Projection, STGeometry


def test_sentinel1(tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
    """Test ingesting an item corresponding to seattle2020 to local filesystem."""
    band_name = "vv"
    # The asset band is vv but in the STAC metadata it is capitalized.
    # We search for a VV+VH image since that is the standard one for GRD/IW.
    s1_query_dict = {"sar:polarizations": {"eq": ["VV", "VH"]}}
    data_source = Sentinel1(
        band_names=[band_name],
        query=s1_query_dict,
    )

    print("get items")
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    item = item_groups[0][0]

    tile_store_dir = UPath(tmp_path)
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)

    print("ingest")
    layer_name = "layer"
    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name), item_groups[0], [[seattle2020]]
    )
    assert tile_store.is_raster_ready(layer_name, item.name, [band_name])


def test_sentinel2(tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
    """Test ingesting an item corresponding to seattle2020 to local filesystem."""
    band_name = "B04"
    data_source = Sentinel2(assets=[band_name])

    print("get items")
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    item = item_groups[0][0]

    tile_store_dir = UPath(tmp_path)
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)

    print("ingest")
    layer_name = "layer"
    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name), item_groups[0], [[seattle2020]]
    )
    assert tile_store.is_raster_ready(layer_name, item.name, [band_name])


def test_cache_dir(tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
    """Make sure cache directory is populated when set."""
    # Use a subdirectory so we also ensure the directory is automatically created.
    cache_dir = UPath(tmp_path / "cache_dir")
    band_name = "B04"
    data_source = Sentinel2(assets=[band_name], cache_dir=cache_dir)
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    data_source.get_items([seattle2020], query_config)[0]
    assert len(list(cache_dir.iterdir())) > 0


class TestSentinel2Pagination:
    """Tests for PlanetaryComputerStacClient ID pagination fallback."""

    @pytest.fixture
    def seattle_long_time_range(self) -> STGeometry:
        """Seattle 64x64 box (10m/pixel UTM) with a 7-year time range.

        This time range (2017-01-01 to 2024-01-01) returns more than 1000 Sentinel-2
        scenes for Seattle, which triggers the ID pagination fallback.
        """
        start_time = datetime(2017, 1, 1, tzinfo=UTC)
        end_time = datetime(2024, 1, 1, tzinfo=UTC)
        time_range = (start_time, end_time)

        # Seattle point in WGS84, then convert to UTM.
        wgs84_shp = shapely.Point(-122.33, 47.61)
        wgs84_geom = STGeometry(WGS84_PROJECTION, wgs84_shp, time_range)

        # Convert to UTM zone 10N (EPSG:32610) at 10m resolution.
        dst_projection = Projection(CRS.from_epsg(32610), 10, -10)
        dst_geom = wgs84_geom.to_projection(dst_projection)
        point = dst_geom.shp

        # Create a 64x64 pixel box (640m x 640m at 10m/pixel).
        size = 64
        box = shapely.box(
            point.x - size // 2,
            point.y - size // 2,
            point.x + size // 2,
            point.y + size // 2,
        )
        return STGeometry(dst_projection, box, time_range)

    def test_pagination(self, seattle_long_time_range: STGeometry) -> None:
        """Test that ID pagination works for queries with more than 1000 items.

        This test uses a small geometry (64x64 box around Seattle) with a 7-year
        time range that returns more than 1000 Sentinel-2 scenes. This triggers
        the ID pagination fallback in PlanetaryComputerStacClient.
        """
        # Use the Sentinel2 data source with a high max_matches to get all items.
        data_source = Sentinel2(assets=["B04"])
        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS, max_matches=2000)

        item_groups = data_source.get_items([seattle_long_time_range], query_config)[0]

        # Flatten all items from all groups (each group has one item with INTERSECTS).
        all_items = [item for group in item_groups for item in group]

        # Verify we got more than 1000 items (proving pagination worked).
        assert len(all_items) > 1000, (
            f"Expected more than 1000 items, got {len(all_items)}"
        )


def test_cop_dem_glo_30(tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
    """Test retrieving Copernicus DEM items without time filtering."""
    band_name = "DEM"
    data_source = CopDemGlo30(band_name=band_name)

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    item = item_groups[0][0]

    tile_store_dir = UPath(tmp_path)
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)

    layer_name = "layer"
    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name), item_groups[0], [[seattle2020]]
    )
    assert tile_store.is_raster_ready(layer_name, item.name, [band_name])
