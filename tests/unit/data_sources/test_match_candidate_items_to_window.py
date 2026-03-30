from datetime import UTC, datetime, timedelta

import pytest
import shapely
from rasterio.crs import CRS

from rslearn.config import QueryConfig, SpaceMode, TimeMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import Item
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.utils.geometry import Projection, STGeometry, get_global_geometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection


def test_global_geometry() -> None:
    """Verify that a global geometry matches with everything."""
    global_geometry = get_global_geometry(None)
    window_geom = STGeometry(
        Projection(CRS.from_epsg(32610), 1, 1),
        shapely.box(500000, 500000, 500001, 500001),
        None,
    )
    item_groups = match_candidate_items_to_window(
        window_geom, [Item("item", global_geometry)], QueryConfig()
    )
    assert len(item_groups) == 1
    assert len(item_groups[0]) == 1


class TestReprojectionHandling:
    """Tests for reprojection edge cases: antimeridian, close vertices, large geometry."""

    def test_window_geometry_crossing_antimeridian(self) -> None:
        """Verify that a window geometry crossing the antimeridian is handled correctly."""
        item_geom = STGeometry(
            WGS84_PROJECTION,
            shapely.Polygon(
                [
                    (-179.997854, -16.170659),
                    (-179.969444, -16.170659),
                    (-179.969444, -16.143371),
                    (-179.997854, -16.143371),
                    (-179.997854, -16.170659),
                ]
            ),
            (
                datetime(2025, 1, 27, 9, 5, 59, 24000, tzinfo=UTC),
                datetime(2025, 1, 27, 9, 5, 59, 24000, tzinfo=UTC),
            ),
        )
        window_geom = STGeometry(
            Projection(CRS.from_epsg(32701), 1, -1),
            shapely.box(179162, -8211693, 180177, -8210678),
            (
                datetime(2024, 12, 31, 14, 0, tzinfo=UTC),
                datetime(2025, 8, 27, 14, 0, tzinfo=UTC),
            ),
        )
        item_groups = match_candidate_items_to_window(
            window_geom, [Item("item", item_geom)], QueryConfig()
        )
        assert len(item_groups) == 1
        assert len(item_groups[0]) == 1

    @pytest.fixture
    def cdse_item_with_near_duplicate_vertices(self) -> Item:
        """CDSE STAC geometry for S2B_MSIL2A_20251115T224759_N0511_R058_T01RBM.

        This Sentinel-2 tile straddles the antimeridian in UTM zone 1N. The
        STAC API returns a 3-part MultiPolygon that is valid in WGS84 but
        contains near-duplicate vertices and coordinates at
        179.99999999999994 (very close to but not exactly 180). When
        reprojected to UTM zone 1N, these cause a degenerate triangle
        (area=0) and a TopologyException.

        Queried from https://stac.dataspace.copernicus.eu/v1/search?collections=sentinel-2-l2a&ids=S2B_MSIL2A_20251115T224759_N0511_R058_T01RBM_20251116T000614
        """
        shp = shapely.geometry.shape(
            {
                "type": "MultiPolygon",
                "coordinates": [
                    [
                        [
                            [-178.93747545748153, 28.16415854963138],
                            [-178.93747545748153, 28.164158549631395],
                            [-178.93747545748147, 28.16415854963138],
                            [-178.93747545748153, 28.16415854963138],
                        ]
                    ],
                    [
                        [
                            [-180, 28.894559786649314],
                            [-178.94913222743162, 28.802724820493484],
                            [-178.93747545748153, 28.16415854963139],
                            [-180, 28.405489029508054],
                            [-180, 28.894559786649314],
                        ]
                    ],
                    [
                        [
                            [179.93733286444046, 28.41807604484601],
                            [179.9236638370703, 28.893134383068205],
                            [179.99999999999994, 28.894559786649314],
                            [180, 28.894559786649314],
                            [180, 28.405489029508054],
                            [179.99999999999994, 28.405489029508065],
                            [179.93733286444046, 28.41807604484601],
                        ]
                    ],
                ],
            }
        )
        return Item(
            "S2B_MSIL2A_20251115T224759_N0511_R058_T01RBM",
            STGeometry(WGS84_PROJECTION, shp, None),
        )

    def test_close_vertices_geometry_contains(
        self, cdse_item_with_near_duplicate_vertices: Item
    ) -> None:
        """Item with degenerate triangle should not crash or give wrong results.

        The CDSE STAC geometry for T01RBM has a degenerate zero-area triangle from
        near-duplicate vertices. If it is naively reprojected to UTM zone 1N, it may
        trigger a TopologyException. We should be robust to that issue.
        """
        window_proj = Projection(CRS.from_epsg(32601), 10, -10)
        window_shp = shapely.box(25000, -315000, 25500, -314500)
        window_geom = STGeometry(window_proj, window_shp, None)

        item_groups = match_candidate_items_to_window(
            window_geom,
            [cdse_item_with_near_duplicate_vertices],
            QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1),
        )
        assert len(item_groups) == 1
        assert len(item_groups[0]) == 1

    def test_large_geometry_contains(self) -> None:
        """Large WGS84 item covering a UTM window should CONTAINS-match.

        Item WGS84: (-47, 30, 53, 50)
        Window WGS84: (2.5, 39.5, 3.5, 40.5)

        With naive re-projection to UTM, the large geometry no longer contains the
        window:

        Item EPSG:32631: (-4599K, 4650K, 5599K, 6839K)
        Window EPSG:32631: (457K, 4372K, 543K, 4483K)
        """
        item_shp = shapely.box(-47, 30, 53, 50)
        item_geom = STGeometry(WGS84_PROJECTION, item_shp, None)

        window_proj = Projection(CRS.from_epsg(32631), 10, -10)
        window_geom = STGeometry(
            WGS84_PROJECTION, shapely.box(2.5, 39.5, 3.5, 40.5), None
        ).to_projection(window_proj)

        item_groups = match_candidate_items_to_window(
            window_geom,
            [Item("large", item_geom)],
            QueryConfig(space_mode=SpaceMode.CONTAINS, max_matches=1),
        )
        assert len(item_groups) == 1
        assert len(item_groups[0]) == 1

    def test_large_geometry_non_intersecting(self) -> None:
        """Large WGS84 item that doesn't intersect a small UTM window should return no matches."""
        item_shp = shapely.box(100, 30, 110, 40)
        item_geom = STGeometry(WGS84_PROJECTION, item_shp, None)

        # Create UTM window that is nearby but not intersecting.
        window_geom_wgs84 = STGeometry(
            WGS84_PROJECTION, shapely.box(98, 35, 99, 36), None
        )
        window_proj = get_utm_ups_projection(
            window_geom_wgs84.shp.bounds[0], window_geom_wgs84.shp.bounds[1], 10, -10
        )
        window_geom = window_geom_wgs84.to_projection(window_proj)

        item_groups = match_candidate_items_to_window(
            window_geom,
            [Item("item", item_geom)],
            QueryConfig(space_mode=SpaceMode.CONTAINS, max_matches=1),
        )
        assert len(item_groups) == 0


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


class TestSpaceMode:
    """Test the contains and intersects space modes."""

    START_TIME = datetime(2024, 1, 1, tzinfo=UTC)
    END_TIME = datetime(2024, 1, 2, tzinfo=UTC)

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


class TestMosaicCompositingOverlaps:
    """Tests for the mosaic_compositing_overlaps option in QueryConfig."""

    @pytest.fixture
    def six_items(self) -> list[Item]:
        """Create six items that need to be mosaicked.

        Items are arranged in two parts (left and right halves of the window).
        Three items cover each half.
        """
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

    def test_overlaps_two(self, six_items: list[Item]) -> None:
        """Test overlaps=2.

        With overlaps=2, pairs of single-coverage mosaics are consolidated. We have 6
        items that form 3 complete mosaics, so after consolidation we should have one
        item group from the first 2 mosaics and another with the 3rd mosaic.
        """
        window_geom = STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 1, 1), None)
        query_config = QueryConfig(
            space_mode=SpaceMode.MOSAIC, max_matches=2, mosaic_compositing_overlaps=2
        )
        item_groups = match_candidate_items_to_window(
            window_geom, six_items, query_config
        )
        # We get 2 groups: first consolidates 2 mosaics, second has partial (1 mosaic)
        assert len(item_groups) == 2
        # First group should have items from first two mosaics (4 items)
        assert set(item.name for item in item_groups[0]) == {
            "part1_item1",
            "part2_item1",
            "part1_item2",
            "part2_item2",
        }
        # Second group has the third mosaic (2 items)
        assert set(item.name for item in item_groups[1]) == {
            "part1_item3",
            "part2_item3",
        }

    def test_overlaps_high_value(self, six_items: list[Item]) -> None:
        """Test very high overlaps value.

        With overlaps=100, all mosaics should be consolidated into one group.
        """
        window_geom = STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 1, 1), None)
        query_config = QueryConfig(
            space_mode=SpaceMode.MOSAIC, max_matches=1, mosaic_compositing_overlaps=100
        )
        item_groups = match_candidate_items_to_window(
            window_geom, six_items, query_config
        )
        # All items should be in one group
        assert len(item_groups) == 1
        assert len(item_groups[0]) == 6


class TestMosaicWithPeriodDuration:
    """Tests for MOSAIC SpaceMode with period_duration set."""

    def test_three_periods(self) -> None:
        """Test creating one mosaic per period, capped at max_matches.

        Four time periods exist but only three are kept due to max_matches=3.
        Most recent periods are prioritized, so the oldest period is dropped.
        """
        base_ts = datetime(2024, 1, 1, tzinfo=UTC)
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
            space_mode=SpaceMode.MOSAIC,
            max_matches=3,
            period_duration=timedelta(days=30),
            per_period_mosaic_reverse_time_order=False,
        )
        item_groups = match_candidate_items_to_window(
            window_geometry, item_list, query_config
        )
        # Most recent 3 periods kept, returned in chronological order:
        # period 2, period 3, period 4 (period 1 dropped by max_matches)
        assert item_groups == [
            [item_list[1], item_list[2]],
            [item_list[3]],
            [item_list[4]],
        ]

    def test_reverse_time_order(self) -> None:
        """Test that per_period_mosaic_reverse_time_order keeps reverse chronological order.

        With per_period_mosaic_reverse_time_order=True (current default), groups
        are returned in reverse chronological order (most recent first).
        """
        base_ts = datetime(2024, 1, 1, tzinfo=UTC)
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
            space_mode=SpaceMode.MOSAIC,
            max_matches=3,
            period_duration=timedelta(days=30),
            per_period_mosaic_reverse_time_order=True,
        )
        with pytest.warns(FutureWarning, match="per_period_mosaic_reverse_time_order"):
            item_groups = match_candidate_items_to_window(
                window_geometry, item_list, query_config
            )
        # Most recent 3 periods in reverse chronological order: period 4, 3, 2
        assert item_groups == [
            [item_list[4]],
            [item_list[3]],
            [item_list[1], item_list[2]],
        ]

    def test_skip_empty_period(self) -> None:
        """Ensure that empty periods are skipped and don't count toward max_matches."""
        base_ts = datetime(2024, 1, 1, tzinfo=UTC)
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
            space_mode=SpaceMode.MOSAIC,
            max_matches=2,
            period_duration=timedelta(days=30),
            per_period_mosaic_reverse_time_order=False,
        )
        item_groups = match_candidate_items_to_window(
            window_geometry, item_list, query_config
        )
        # Most recent first: period 4 skipped (no spatial match), period 3, 2
        # kept. Reversed to chronological: period 2, period 3.
        assert item_groups == [
            [item_list[1]],
            [item_list[2]],
        ]


class TestSingleComposite:
    """Tests for SINGLE_COMPOSITE space mode."""

    def test_spatiotemporal_filtering(self) -> None:
        """All spatially and temporally intersecting items go into one group.

        Four items:
        - item_a: intersects both spatially and temporally (full window, timestep 1)
        - item_b: intersects both spatially and temporally (full window, timestep 2)
        - item_c: intersects temporally but NOT spatially
        - item_d: intersects spatially but NOT temporally

        Only item_a and item_b should be in a single item group.
        """
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        t1 = datetime(2024, 2, 1, tzinfo=UTC)
        t2 = datetime(2024, 3, 1, tzinfo=UTC)
        t3 = datetime(2024, 6, 1, tzinfo=UTC)

        window_bbox = shapely.box(0, 0, 1, 1)
        window_geom = STGeometry(WGS84_PROJECTION, window_bbox, (t0, t2))

        item_a = Item(
            "item_a",
            STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 1, 1), (t0, t1)),
        )
        item_b = Item(
            "item_b",
            STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 1, 1), (t1, t2)),
        )
        # Temporally overlapping but spatially disjoint.
        item_c = Item(
            "item_c",
            STGeometry(WGS84_PROJECTION, shapely.box(5, 5, 6, 6), (t0, t1)),
        )
        # Spatially overlapping but temporally disjoint.
        item_d = Item(
            "item_d",
            STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 1, 1), (t2, t3)),
        )

        query_config = QueryConfig(
            space_mode=SpaceMode.SINGLE_COMPOSITE,
            max_matches=10,
        )
        item_groups = match_candidate_items_to_window(
            window_geom, [item_a, item_b, item_c, item_d], query_config
        )
        assert len(item_groups) == 1
        assert item_groups[0] == [item_a, item_b]


class TestMinMatches:
    """Test that min_matches is respected for all space modes."""

    def test_min_matches_contains(self) -> None:
        """Test min_matches with CONTAINS mode."""
        bbox = shapely.box(0, 0, 1, 1)
        time_range = (
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 2, 1, tzinfo=UTC),
        )
        geom = STGeometry(WGS84_PROJECTION, bbox, time_range)
        item_list = [
            Item("item0", STGeometry(WGS84_PROJECTION, bbox, time_range)),
            Item("item1", STGeometry(WGS84_PROJECTION, bbox, time_range)),
        ]
        # Only 2 items, but min_matches=3, so should return empty
        query_config = QueryConfig(
            space_mode=SpaceMode.CONTAINS, max_matches=10, min_matches=3
        )
        item_groups = match_candidate_items_to_window(geom, item_list, query_config)
        assert item_groups == []

    def test_min_matches_intersects(self) -> None:
        """Test min_matches with INTERSECTS mode."""
        bbox = shapely.box(0, 0, 1, 1)
        time_range = (
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 2, 1, tzinfo=UTC),
        )
        geom = STGeometry(WGS84_PROJECTION, bbox, time_range)
        item_list = [
            Item("item0", STGeometry(WGS84_PROJECTION, bbox, time_range)),
            Item("item1", STGeometry(WGS84_PROJECTION, bbox, time_range)),
        ]
        # Only 2 items, but min_matches=3, so should return empty
        query_config = QueryConfig(
            space_mode=SpaceMode.INTERSECTS, max_matches=10, min_matches=3
        )
        item_groups = match_candidate_items_to_window(geom, item_list, query_config)
        assert item_groups == []

    def test_min_matches_mosaic(self) -> None:
        """Test min_matches with MOSAIC mode."""
        part1 = STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 0.5, 1), None)
        part2 = STGeometry(WGS84_PROJECTION, shapely.box(0.5, 0, 1, 1), None)
        items = [
            Item("part1_item1", part1),
            Item("part2_item1", part2),
        ]
        window_geom = STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 1, 1), None)
        # Only 1 mosaic can be created, but min_matches=2, so should return empty
        query_config = QueryConfig(
            space_mode=SpaceMode.MOSAIC, max_matches=10, min_matches=2
        )
        item_groups = match_candidate_items_to_window(window_geom, items, query_config)
        assert item_groups == []

    def test_min_matches_mosaic_with_overlaps(self) -> None:
        """Test min_matches with MOSAIC mode and high overlaps (compositing behavior)."""
        bbox = shapely.box(0, 0, 1, 1)
        time_range = (
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 2, 1, tzinfo=UTC),
        )
        geom = STGeometry(WGS84_PROJECTION, bbox, time_range)
        item_list = [
            Item("item0", STGeometry(WGS84_PROJECTION, bbox, time_range)),
            Item("item1", STGeometry(WGS84_PROJECTION, bbox, time_range)),
        ]
        # MOSAIC with high overlaps consolidates items, returning 1 group
        # But min_matches=2, so should return empty
        query_config = QueryConfig(
            space_mode=SpaceMode.MOSAIC,
            max_matches=10,
            min_matches=2,
            mosaic_compositing_overlaps=100,
        )
        item_groups = match_candidate_items_to_window(geom, item_list, query_config)
        assert item_groups == []

    def test_min_matches_mosaic_with_period_duration(self) -> None:
        """Test min_matches with MOSAIC mode and period_duration."""
        base_ts = datetime(2024, 1, 1, tzinfo=UTC)
        period_duration = timedelta(days=30)
        bbox = shapely.box(0, 0, 1, 1)
        window_geometry = STGeometry(
            WGS84_PROJECTION, bbox, (base_ts, base_ts + period_duration * 4)
        )
        # Only 1 period has items, but min_matches=2, so should return empty
        item_list = [
            Item(
                "item0",
                STGeometry(
                    WGS84_PROJECTION,
                    bbox,
                    (base_ts, base_ts + period_duration),
                ),
            ),
        ]
        query_config = QueryConfig(
            space_mode=SpaceMode.MOSAIC,
            max_matches=10,
            min_matches=2,
            period_duration=period_duration,
            per_period_mosaic_reverse_time_order=False,
        )
        item_groups = match_candidate_items_to_window(
            window_geometry, item_list, query_config
        )
        assert item_groups == []
