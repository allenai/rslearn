from datetime import datetime, timedelta, timezone

import pytest
import shapely
from rasterio.crs import CRS

from rslearn.config import QueryConfig, SpaceMode, TimeMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import Item
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.utils.geometry import STGeometry, get_global_geometry


def test_global_geometry() -> None:
    """Verify that a global geometry matches with everything."""
    global_geometry = get_global_geometry(None)
    window_geom = STGeometry(
        CRS.from_epsg(32610), shapely.box(500000, 500000, 500001, 500001), None
    )
    item_groups = match_candidate_items_to_window(
        window_geom, [Item("item", global_geometry)], QueryConfig()
    )
    assert len(item_groups) == 1
    assert len(item_groups[0]) == 1


class TestTimeMode:
    START_TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)
    END_TIME = datetime(2024, 1, 2, tzinfo=timezone.utc)
    BBOX = shapely.box(0, 0, 1, 1)

    @pytest.fixture
    def item_list(self) -> list[Item]:
        def make_item(name: str, ts: datetime) -> Item:
            return Item(name, STGeometry(WGS84_PROJECTION, self.BBOX, (ts, ts)))

        item0 = make_item("item0", self.START_TIME - timedelta(hours=1))
        item1 = make_item("item1", self.START_TIME + timedelta(hours=18))
        item2 = make_item("item2", self.START_TIME + timedelta(hours=6))
        item3 = make_item("item3", self.START_TIME + timedelta(hours=12))
        item4 = make_item("item4", self.START_TIME + timedelta(days=2))
        return [item0, item1, item2, item3, item4]

    def test_within_mode(self, item_list: list[Item]) -> None:
        """Verify that WITHIN time mode preserves the item order."""
        window_geom = STGeometry(
            WGS84_PROJECTION, self.BBOX, (self.START_TIME, self.END_TIME)
        )
        query_config = QueryConfig(
            space_mode=SpaceMode.INTERSECTS, time_mode=TimeMode.WITHIN, max_matches=10
        )
        item_groups = match_candidate_items_to_window(
            window_geom, item_list, query_config
        )
        assert item_groups == [[item_list[1]], [item_list[2]], [item_list[3]]]

    def test_before_mode(self, item_list: list[Item]) -> None:
        """Verify that BEFORE time mode yields items in reverse temporal order."""
        window_geom = STGeometry(
            WGS84_PROJECTION, self.BBOX, (self.START_TIME, self.END_TIME)
        )
        query_config = QueryConfig(
            space_mode=SpaceMode.INTERSECTS, time_mode=TimeMode.BEFORE, max_matches=10
        )
        item_groups = match_candidate_items_to_window(
            window_geom, item_list, query_config
        )
        assert item_groups == [[item_list[1]], [item_list[3]], [item_list[2]]]

    def test_after_mode(self, item_list: list[Item]) -> None:
        """Verify that AFTER time mode yields items in temporal order."""
        window_geom = STGeometry(
            WGS84_PROJECTION, self.BBOX, (self.START_TIME, self.END_TIME)
        )
        query_config = QueryConfig(
            space_mode=SpaceMode.INTERSECTS, time_mode=TimeMode.AFTER, max_matches=10
        )
        item_groups = match_candidate_items_to_window(
            window_geom, item_list, query_config
        )
        assert item_groups == [[item_list[2]], [item_list[3]], [item_list[1]]]
