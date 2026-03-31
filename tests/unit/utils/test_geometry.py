import json
from datetime import datetime

import pytest
import rasterio
import shapely
from rasterio import CRS

from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import (
    Projection,
    ResolutionFactor,
    STGeometry,
    safely_reproject_within_valid_area,
    split_shape_at_antimeridian,
)

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

    @pytest.fixture
    def utm_geometry_crossing_antimeridian(self) -> STGeometry:
        """A small UTM geometry that straddles the antimeridian."""
        utm_proj = Projection(CRS.from_epsg(32701), 1, -1)
        # Reproject two WGS84 points on opposite sides of the antimeridian,
        # then build a box from the resulting UTM pixel coordinates. The result should
        # be a small box crossing the antimeridian, not the big box spanning the whole
        # world.
        p1 = STGeometry(
            WGS84_PROJECTION, shapely.Point(-179.99, -16.17), None
        ).to_projection(utm_proj)
        p2 = STGeometry(
            WGS84_PROJECTION, shapely.Point(179.99, -16.14), None
        ).to_projection(utm_proj)
        shp = shapely.box(p1.shp.x, p1.shp.y, p2.shp.x, p2.shp.y)
        return STGeometry(utm_proj, shp, None)

    def test_intersects_antimeridian_true(
        self, utm_geometry_crossing_antimeridian: STGeometry
    ) -> None:
        """WGS84 geometry near antimeridian should intersect a UTM geometry crossing it."""
        nearby_item = STGeometry(
            WGS84_PROJECTION, shapely.box(-180.0, -16.17, -179.995, -16.14), None
        )
        assert nearby_item.intersects(utm_geometry_crossing_antimeridian)

    def test_intersects_antimeridian_false(
        self, utm_geometry_crossing_antimeridian: STGeometry
    ) -> None:
        """WGS84 item far from antimeridian should NOT intersect a UTM geometry crossing it."""
        distant_item = STGeometry(
            WGS84_PROJECTION, shapely.box(10, -16.17, 11, -16.14), None
        )
        assert not distant_item.intersects(utm_geometry_crossing_antimeridian)

    def test_to_wgs84_noop_for_wgs84(self) -> None:
        """to_wgs84() on a WGS84 geometry should return self."""
        geom = STGeometry(WGS84_PROJECTION, shapely.box(10, 20, 11, 21), None)
        assert geom.to_wgs84() is geom

    def test_to_wgs84_antimeridian_split(
        self, utm_geometry_crossing_antimeridian: STGeometry
    ) -> None:
        """to_wgs84() on an antimeridian-crossing UTM geometry should produce a compact MultiPolygon."""
        wgs84 = utm_geometry_crossing_antimeridian.to_wgs84()
        assert wgs84.projection == WGS84_PROJECTION
        # The result should be a MultiPolygon with two small components near +/-180,
        # not a single polygon spanning -180 to +180.
        assert wgs84.shp.geom_type == "MultiPolygon"
        for part in wgs84.shp.geoms:
            lon_extent = part.bounds[2] - part.bounds[0]
            assert lon_extent < 1.0  # each component < 1 degree wide

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


class TestSplitAntiMeridian:
    epsilon = 1e-3

    def test_point_unaffected(self) -> None:
        # This point shouldn't be modified.
        p = split_shape_at_antimeridian(shapely.Point(0, 0))
        assert p.x == 0 and p.y == 0

    def test_point_negative_antimeridian(self) -> None:
        p = split_shape_at_antimeridian(shapely.Point(-180, 45), epsilon=self.epsilon)
        assert abs(p.x - (-180 + self.epsilon)) < self.epsilon / 2 and p.y == 45

    def test_point_positive_antimeridian(self) -> None:
        p = split_shape_at_antimeridian(shapely.Point(180, 45), epsilon=self.epsilon)
        assert abs(p.x - (180 - self.epsilon)) < self.epsilon / 2 and p.y == 45

    def test_line(self) -> None:
        line = shapely.LineString(
            [
                [175, 1],
                [-175, 2],
            ]
        )
        output = split_shape_at_antimeridian(line)
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
        output = split_shape_at_antimeridian(polygon)
        assert abs(output.area - expected_area) < self.epsilon
        assert (output.bounds[0] + 180) < self.epsilon and (
            output.bounds[2] - 180
        ) < self.epsilon

    def test_polygon_crossing_zero_longitude(self) -> None:
        # Splitting shouldn't affect shapes that don't need to be split.
        polygon = shapely.box(-1, -1, 1, 1)
        output = split_shape_at_antimeridian(polygon)
        assert output == polygon


class TestSafelyReprojectWithinValidArea:
    """Unit tests for safely_reproject_within_valid_area."""

    def test_same_projection_passthrough(self) -> None:
        """When src and valid_geom share a projection, return src unchanged."""
        proj = Projection(CRS.from_epsg(32631), 10, -10)
        src = STGeometry(proj, shapely.box(0, 0, 100, 100), None)
        valid = STGeometry(proj, shapely.box(50, 50, 150, 150), None)
        result = safely_reproject_within_valid_area([src], valid)[0]
        assert result is src

    def test_invalid_projection(self) -> None:
        """Test on geometries that would have error with direct re-projection."""
        src_geom = STGeometry(WGS84_PROJECTION, shapely.Point(104.672, 0.099), None)
        dst_geom = STGeometry(
            Projection(CRS.from_epsg(32632), 10, -10),
            shapely.box(63232, -628224, 63488, -627968),
            None,
        )

        # First verify that direct re-projection has error.
        with pytest.raises(rasterio._err.CPLE_AppDefinedError):
            src_geom.to_projection(dst_geom.projection)

        # Now verify that it works with safely_reproject_within_valid_area.
        # It should return None since these geometries don't actually intersect.
        result = safely_reproject_within_valid_area([src_geom], dst_geom)[0]
        assert result is None

    def test_antimeridian_handling_disjoint(self) -> None:
        """Verify that antimeridian handling works for non-intersecting geometries."""
        # Source geometry intersects antimeridian at +10 degrees latitude.
        # We also include the point from test_invalid_projection that should fail
        # direct re-projection.
        # The point is outside the area of use for the CRS but it seems to be okay.
        box = shapely.box(-180.1, 10, -179.9, 10.1)
        point = shapely.Point(104.672, 0.099)
        wgs84_geom = STGeometry(WGS84_PROJECTION, box.union(point), None)
        src_geom = wgs84_geom.to_projection(Projection(CRS.from_epsg(32601), 1, 1))

        # Destination geometry is at +10 degrees latitude but not on the antimeridian.
        wgs84_geom = STGeometry(
            WGS84_PROJECTION, shapely.box(-179.1, 10, -178.9, 10.1), None
        )
        dst_geom = wgs84_geom.to_projection(Projection(CRS.from_epsg(32632), 10, -10))

        with pytest.raises(rasterio._err.CPLE_AppDefinedError):
            src_geom.to_projection(dst_geom.projection)

        result = safely_reproject_within_valid_area([src_geom], dst_geom)[0]
        assert result is None

    def test_antimeridian_large_source_contains_window(self) -> None:
        """Re-projection should work when the window crosses the antimeridian.

        In this case, the window should be split into two parts and each should be
        buffered separately to handle the clipping.
        """
        utm_proj = Projection(CRS.from_epsg(32701), 10, -10)
        p1 = STGeometry(
            WGS84_PROJECTION, shapely.Point(-179.99, -16.17), None
        ).to_projection(utm_proj)
        p2 = STGeometry(
            WGS84_PROJECTION, shapely.Point(179.99, -16.14), None
        ).to_projection(utm_proj)
        valid_geom = STGeometry(
            utm_proj, shapely.box(p1.shp.x, p1.shp.y, p2.shp.x, p2.shp.y), None
        )

        src_geom = STGeometry(
            WGS84_PROJECTION,
            shapely.box(-180, -20, -90, 0).union(shapely.box(-179, -20, 180, 0)),
            None,
        )
        result = safely_reproject_within_valid_area([src_geom], valid_geom)[0]
        assert result is not None
        # contains check may fail since it is on the border, but the intersection area
        # should be almost the same as valid_geom's area.
        assert result.shp.intersection(valid_geom.shp).area == pytest.approx(
            valid_geom.shp.area
        )

    def test_antimeridian_handling_intersecting(self) -> None:
        """Verify that antimeridian handling works for intersecting geometries."""
        wgs84_geom = STGeometry(
            WGS84_PROJECTION, shapely.box(-180.1, 10, -179.9, 10.1), None
        )
        src_geom = wgs84_geom.to_projection(Projection(CRS.from_epsg(32601), 1, 1))

        wgs84_geom = STGeometry(
            WGS84_PROJECTION, shapely.box(-179.95, 10, -179.85, 10.1), None
        )
        dst_geom = wgs84_geom.to_projection(Projection(CRS.from_epsg(32660), 1, 1))

        result = safely_reproject_within_valid_area([src_geom], dst_geom)[0]
        assert result is not None
        assert result.shp.area > 0
        # The reprojected source should overlap the destination area (source only
        # partially covers destination since it spans -180.1 to -179.9 while destination
        # spans -179.95 to -179.85).
        assert result.shp.intersects(dst_geom.shp)


class TestResolutionFactor:
    """Tests for ResolutionFactor."""

    def test_floating_point_resolution(self) -> None:
        """Verify that ResolutionFactor works correctly with non-integer resolution."""
        # We test this because previously we had bug where it would round non-integer
        # resolution to integer.
        proj = Projection(CRS.from_epsg(4326), 0.001, -0.001)
        factor = ResolutionFactor()
        result = factor.multiply_projection(proj)
        assert result.x_resolution == pytest.approx(0.001)
        assert result.y_resolution == pytest.approx(-0.001)

    def test_divide_resolution(self) -> None:
        """Verify that ResolutionFactor can make resolution smaller (more fine-grained)."""
        proj = Projection(CRS.from_epsg(32610), 10, -10)
        factor = ResolutionFactor(numerator=3)
        result = factor.multiply_projection(proj)
        assert result.x_resolution == pytest.approx(10 / 3)
        assert result.y_resolution == pytest.approx(-10 / 3)
