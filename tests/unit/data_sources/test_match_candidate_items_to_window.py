from datetime import UTC, datetime, timedelta

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
    START_TIME = datetime(2024, 1, 1, tzinfo=UTC)
    END_TIME = datetime(2024, 1, 2, tzinfo=UTC)
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


class TestMosaic:
    @pytest.fixture
    def six_items(self) -> list[Item]:
        part1 = STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 0.5, 1), None)
        part2 = STGeometry(WGS84_PROJECTION, shapely.box(0.5, 0, 1, 1), None)

        return [
            Item("part1_item1", part1),
            Item("part1_item2", part1),
            Item("part1_item3", part1),
            Item("part2_item1", part2),
            Item("part2_item2", part2),
            Item("part2_item3", part2),
        ]

    def test_two_mosaics(self, six_items: list[Item]) -> None:
        """Test mosaic creation.

        We split up overall geometry into two parts, and pass three items for each
        part. We make sure that the mosaic is created with the first two items for each
        box (in the same order we pass them.)
        """
        window_geom = STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 1, 1), None)
        query_config = QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=2)
        item_groups = match_candidate_items_to_window(
            window_geom, six_items, query_config
        )
        assert item_groups == [
            [six_items[0], six_items[3]],
            [six_items[1], six_items[4]],
        ]

    def test_three_mosaics(self, six_items: list[Item]) -> None:
        """Test mosaic creation.

        Like above but three groups should be returned (we have exactly enough items
        for those mosaics).
        """
        window_geom = STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 1, 1), None)
        query_config = QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=3)
        item_groups = match_candidate_items_to_window(
            window_geom, six_items, query_config
        )
        assert item_groups == [
            [six_items[0], six_items[3]],
            [six_items[1], six_items[4]],
            [six_items[2], six_items[5]],
        ]

    def test_ten_mosaics(self, six_items: list[Item]) -> None:
        """Test mosaic creation.

        Like above but ensure that if we ask for ten mosaics, only three groups are
        returned.
        """
        window_geom = STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 1, 1), None)
        query_config = QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=10)
        item_groups = match_candidate_items_to_window(
            window_geom, six_items, query_config
        )
        assert item_groups == [
            [six_items[0], six_items[3]],
            [six_items[1], six_items[4]],
            [six_items[2], six_items[5]],
        ]

    def test_zero_mosaics(self, six_items: list[Item]) -> None:
        """Ensure zero mosaics are created when items do not intersect geometry."""
        window_geom = STGeometry(WGS84_PROJECTION, shapely.box(1, 0, 2, 1), None)
        query_config = QueryConfig(space_mode=SpaceMode.MOSAIC)
        item_groups = match_candidate_items_to_window(
            window_geom, six_items, query_config
        )
        assert len(item_groups) == 0

    def test_partial_mosaics(self, six_items: list[Item]) -> None:
        """Ensure partial mosaics are produced.

        Here we will pass three items on the left and one item on the right, requesting
        three mosaics. We should get three mosaics where the second two only have
        partial coverage of the window geometry.
        """
        items_to_use = [
            # Left.
            six_items[0],
            six_items[1],
            six_items[2],
            # Right.
            six_items[3],
        ]
        window_geom = STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 1, 1), None)
        query_config = QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=3)
        item_groups = match_candidate_items_to_window(
            window_geom, items_to_use, query_config
        )
        assert item_groups == [
            [six_items[0], six_items[3]],
            [six_items[1]],
            [six_items[2]],
        ]
