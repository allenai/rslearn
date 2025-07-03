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


class TestSpaceMode:
    """Test the contains and intersects space modes."""

    START_TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)
    END_TIME = datetime(2024, 1, 2, tzinfo=timezone.utc)

    @pytest.fixture
    def window_geometry(self) -> STGeometry:
        return STGeometry(
            WGS84_PROJECTION,
            shapely.box(0, 0, 0.7, 0.7),
            (self.START_TIME, self.END_TIME),
        )

    @pytest.fixture
    def item_list(self) -> list[Item]:
        def make_item(name: str, geom: shapely.Geometry) -> Item:
            return Item(
                name,
                STGeometry(WGS84_PROJECTION, geom, (self.START_TIME, self.END_TIME)),
            )

        item0 = make_item("item0", shapely.box(-0.1, -0.1, 0.5, 0.5))
        item1 = make_item("item1", shapely.box(-0.1, -0.1, 0.75, 0.75))
        item2 = make_item("item2", shapely.box(0.65, 0.65, 0.75, 0.75))
        item3 = make_item("item3", shapely.box(0.1, 0.1, 0.2, 0.2))
        item4 = make_item("item4", shapely.box(1, 1, 2, 2))
        return [item0, item1, item2, item3, item4]

    def test_contains_mode(
        self, window_geometry: STGeometry, item_list: list[Item]
    ) -> None:
        """Verify that CONTAINS selects only items that fully contain the window."""
        query_config = QueryConfig(space_mode=SpaceMode.CONTAINS, max_matches=10)
        item_groups = match_candidate_items_to_window(
            window_geometry, item_list, query_config
        )
        print([group[0].name for group in item_groups])
        assert item_groups == [[item_list[1]]]

    def test_intersects_mode(
        self, window_geometry: STGeometry, item_list: list[Item]
    ) -> None:
        """Verify that INTERSECTS selects all items that intersect."""
        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS, max_matches=10)
        item_groups = match_candidate_items_to_window(
            window_geometry, item_list, query_config
        )
        assert item_groups == [
            [item_list[0]],
            [item_list[1]],
            [item_list[2]],
            [item_list[3]],
        ]


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


class TestPerPeriodMosaic:
    """Tests for the PER_PERIOD_MOSAIC SpaceMode."""

    def test_three_mosaics(self) -> None:
        """Test creating two full mosaics and one partial mosaic.

        We provide time range with four time periods, but the full mosaic for first
        (oldest) time period should not be used due to the max_matches=3.
        """
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        period_duration = timedelta(days=30)
        periods = [
            (base_ts, base_ts + period_duration),
            (base_ts + period_duration, base_ts + period_duration * 2),
            (base_ts + period_duration * 2, base_ts + period_duration * 3),
            (base_ts + period_duration * 3, base_ts + period_duration * 4),
        ]
        bbox = shapely.box(0, 0, 1, 1)
        window_geometry = STGeometry(
            WGS84_PROJECTION, bbox, (base_ts, base_ts + period_duration * 4)
        )
        item_list = [
            # Full first time period.
            Item("item0", STGeometry(WGS84_PROJECTION, bbox, periods[0])),
            # Full second time period with two items.
            Item(
                "item1",
                STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 1, 0.5), periods[1]),
            ),
            Item(
                "item2",
                STGeometry(WGS84_PROJECTION, shapely.box(0, 0.5, 1, 1), periods[1]),
            ),
            # Partial third time period.
            Item(
                "item3",
                STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 0.5, 0.5), periods[2]),
            ),
            # Full fourth time period.
            Item("item4", STGeometry(WGS84_PROJECTION, bbox, periods[3])),
        ]
        query_config = QueryConfig(
            space_mode=SpaceMode.PER_PERIOD_MOSAIC, max_matches=3
        )
        item_groups = match_candidate_items_to_window(
            window_geometry, item_list, query_config
        )
        assert item_groups == [
            [item_list[4]],
            [item_list[3]],
            [item_list[1], item_list[2]],
        ]

    def test_skip_empty_period(self) -> None:
        """Ensure that empty periods are skipped so it falls back to earlier period."""
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        period_duration = timedelta(days=30)
        periods = [
            (base_ts, base_ts + period_duration),
            (base_ts + period_duration, base_ts + period_duration * 2),
            (base_ts + period_duration * 2, base_ts + period_duration * 3),
            (base_ts + period_duration * 3, base_ts + period_duration * 4),
        ]
        bbox = shapely.box(0, 0, 1, 1)
        window_geometry = STGeometry(
            WGS84_PROJECTION, bbox, (base_ts, base_ts + period_duration * 4)
        )
        item_list = [
            # Full first time period.
            Item("item0", STGeometry(WGS84_PROJECTION, bbox, periods[0])),
            # Full second time period.
            Item("item1", STGeometry(WGS84_PROJECTION, bbox, periods[1])),
            # Full third time period.
            Item("item2", STGeometry(WGS84_PROJECTION, bbox, periods[2])),
            # Fourth time period has no items within the window geometry so it should be skipped.
            Item(
                "item3",
                STGeometry(WGS84_PROJECTION, shapely.box(2, 2, 3, 3), periods[3]),
            ),
        ]
        query_config = QueryConfig(
            space_mode=SpaceMode.PER_PERIOD_MOSAIC, max_matches=2
        )
        item_groups = match_candidate_items_to_window(
            window_geometry, item_list, query_config
        )
        assert item_groups == [
            [item_list[2]],
            [item_list[1]],
        ]


def test_min_matches() -> None:
    """Ensure that no groups are returned if minimum matches is set."""
    bbox = shapely.box(0, 0, 1, 1)
    time_range = (
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 2, 1, tzinfo=timezone.utc),
    )
    geom = STGeometry(WGS84_PROJECTION, bbox, time_range)
    item_list = [
        Item("item0", geom),
        Item("item1", geom),
    ]
    query_config = QueryConfig(
        space_mode=SpaceMode.CONTAINS, max_matches=3, min_matches=3
    )
    item_groups = match_candidate_items_to_window(geom, item_list, query_config)
    assert item_groups == []
