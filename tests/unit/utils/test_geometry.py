import json
from datetime import datetime

import pytest
import shapely
from rasterio import CRS

from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry, split_shape_at_prime_meridian

WEB_MERCATOR_EPSG = 3857


class TestProjection:
    def test_equals(self) -> None:
        crs = CRS.from_epsg(WEB_MERCATOR_EPSG)
        proj1 = Projection(crs, 10, -10)
        proj2 = Projection(crs, 10, -10)
        assert proj1 == proj2
        proj2 = Projection(crs, 11, -10)
        assert proj1 != proj2
        proj2 = Projection(WGS84_PROJECTION.crs, 10, -10)
        assert proj1 != proj2

    def test_serialize(self) -> None:
        proj = Projection(CRS.from_epsg(WEB_MERCATOR_EPSG), 10, -10)
        encoded = json.dumps(proj.serialize())
        new_proj = Projection.deserialize(json.loads(encoded))
        assert proj == new_proj


class TestSTGeometry:
    @pytest.fixture(scope="class")
    def shp(self) -> shapely.Geometry:
        return shapely.box(10, 10, 11, 11)

    @pytest.fixture(scope="class")
    def time_range(self) -> tuple[datetime, datetime]:
        return (datetime(2022, 1, 15), datetime(2022, 1, 16))

    @pytest.fixture(scope="class")
    def geom(
        self, shp: shapely.Geometry, time_range: tuple[datetime, datetime]
    ) -> STGeometry:
        return STGeometry(WGS84_PROJECTION, shp, time_range)

    def test_contains_time(self, geom: STGeometry) -> None:
        assert geom.contains_time(datetime(2022, 1, 15, 12))
        assert not geom.contains_time(datetime(2022, 1, 16, 12))
        assert not geom.contains_time(datetime(2022, 1, 14, 12))

    def test_distance_to_time(self, geom: STGeometry) -> None:
        # In range, should be zero.
        delta1 = geom.distance_to_time(datetime(2022, 1, 15, 12))
        assert delta1.seconds == 0

        # Out of range, should be 12 hours.
        delta2 = geom.distance_to_time(datetime(2022, 1, 14, 12))
        assert delta2.seconds == 12 * 3600

    def test_distance_to_time_range(self, geom: STGeometry) -> None:
        # Intersecting, should be zero.
        rng1 = (datetime(2022, 1, 15, 12), datetime(2022, 1, 16, 12))
        delta1 = geom.distance_to_time_range(rng1)
        assert delta1.seconds == 0

        # Non-intersecting, should be 12 hours.
        rng2 = (datetime(2022, 1, 16, 12), datetime(2022, 1, 16, 13))
        delta2 = geom.distance_to_time_range(rng2)
        assert delta2.seconds == 12 * 3600

    def test_intersects_time_range(self, geom: STGeometry) -> None:
        rng1 = (datetime(2022, 1, 15, 12), datetime(2022, 1, 16, 12))
        assert geom.intersects_time_range(rng1)
        rng2 = (datetime(2022, 1, 16, 12), datetime(2022, 1, 16, 13))
        assert not geom.intersects_time_range(rng2)

    def test_intersects(self, geom: STGeometry) -> None:
        other_proj = Projection(CRS.from_epsg(WEB_MERCATOR_EPSG), 10, -10)

        # Intersecting.
        shp2 = shapely.Point(10.5, 10.5)
        rng_good = (datetime(2022, 1, 15, 12), datetime(2022, 1, 16, 12))
        geom2a = STGeometry(WGS84_PROJECTION, shp2, rng_good)
        assert geom.intersects(geom2a)
        geom2a_reproj = geom2a.to_projection(other_proj)
        assert geom.intersects(geom2a_reproj)

        # Time mismatch.
        rng_bad = (datetime(2022, 1, 16, 12), datetime(2022, 1, 16, 13))
        geom2b = STGeometry(WGS84_PROJECTION, shp2, rng_bad)
        assert not geom.intersects(geom2b)

        shp3 = shapely.Point(12, 12)
        geom3 = STGeometry(WGS84_PROJECTION, shp3, rng_good)
        assert not geom.intersects(geom3)

    def test_to_projection_double(self, geom: STGeometry) -> None:
        # Halve the unit/pixel -> double the coordinates.
        # New box should be 20, 20 to 22, 22.
        dst_proj = Projection(WGS84_PROJECTION.crs, 0.5, 0.5)
        dst_geom = geom.to_projection(dst_proj)
        assert dst_geom.time_range == geom.time_range
        assert dst_geom.shp.equals(shapely.box(20, 20, 22, 22))

    def test_to_projection_webmercator(self, geom: STGeometry) -> None:
        # Just check that it goes back to the same geometry.
        dst_proj = Projection(CRS.from_epsg(WEB_MERCATOR_EPSG), 10, -10)
        dst_geom = geom.to_projection(dst_proj)
        final_geom = dst_geom.to_projection(WGS84_PROJECTION)

        def is_same_shp(shp1: shapely.Geometry, shp2: shapely.Geometry) -> bool:
            intersection = shp1.intersection(shp2).area
            union = shp1.union(shp2).area
            return abs(intersection - union) < 1e-3

        assert not is_same_shp(geom.shp, dst_geom.shp)
        assert is_same_shp(geom.shp, final_geom.shp)

    def test_serialize(self, geom: STGeometry) -> None:
        encoded = json.dumps(geom.serialize())
        new_geom = STGeometry.deserialize(json.loads(encoded))
        assert new_geom.shp == geom.shp
        assert new_geom.time_range == geom.time_range


class TestSplitPrimeMeridian:
    epsilon = 1e-3

    def test_point_unaffected(self) -> None:
        # This point shouldn't be modified.
        p = split_shape_at_prime_meridian(shapely.Point(0, 0))
        assert p.x == 0 and p.y == 0

    def test_point_negative_antimeridian(self) -> None:
        p = split_shape_at_prime_meridian(shapely.Point(-180, 45), epsilon=self.epsilon)
        assert abs(p.x - (-180 + self.epsilon)) < self.epsilon / 2 and p.y == 45

    def test_point_positive_antimeridian(self) -> None:
        p = split_shape_at_prime_meridian(shapely.Point(180, 45), epsilon=self.epsilon)
        assert abs(p.x - (180 - self.epsilon)) < self.epsilon / 2 and p.y == 45

    def test_line(self) -> None:
        line = shapely.LineString(
            [
                [175, 1],
                [-175, 2],
            ]
        )
        output = split_shape_at_prime_meridian(line)
        # Should consist of two lines, one from (175, 1) to (180-ish, 1.5).
        # And another from (-180-ish, 1.5) to (-175, 2).
        assert isinstance(output, shapely.MultiLineString)
        assert len(output.geoms) == 2
        assert (output.geoms[0].coords[0][0] - 175) < self.epsilon
        assert (output.geoms[0].coords[0][1] - 1) < self.epsilon
        assert (output.geoms[0].coords[1][0] - 180) < self.epsilon
        assert (output.geoms[0].coords[1][1] - 1.5) < self.epsilon
        assert (output.geoms[1].coords[0][0] + 180) < self.epsilon
        assert (output.geoms[1].coords[0][1] - 1.5) < self.epsilon
        assert (output.geoms[1].coords[1][0] + 175) < self.epsilon
        assert (output.geoms[1].coords[1][1] - 2) < self.epsilon

    def test_polygon_crossing_antimeridian(self) -> None:
        polygon = shapely.Polygon(
            [
                [-179, 45],
                [179, 45],
                [179, 44],
                [-179, 44],
            ]
        )
        expected_area = 2
        output = split_shape_at_prime_meridian(polygon)
        assert abs(output.area - expected_area) < self.epsilon
        assert (output.bounds[0] + 180) < self.epsilon and (
            output.bounds[2] - 180
        ) < self.epsilon

    def test_polygon_crossing_zero_longitude(self) -> None:
        # Splitting shouldn't affect shapes that don't need to be split.
        polygon = shapely.box(-1, -1, 1, 1)
        output = split_shape_at_prime_meridian(polygon)
        assert output == polygon
