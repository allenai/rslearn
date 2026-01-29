"""Unit tests for XyzTiles data source."""

from datetime import UTC, datetime, timedelta

from rasterio.crs import CRS

from rslearn.data_sources.xyz_tiles import XyzTiles
from rslearn.utils import Projection


class TestXyzTilesGetRasterBounds:
    """Tests for XyzTiles.get_raster_bounds method."""

    def test_get_raster_bounds_utm(self) -> None:
        """Test that bounds properly cover the valid UTM zone extent.

        UTM zone 10N (EPSG:32610) covers roughly -126 to -120 degrees longitude.
        The get_raster_bounds should return bounds that at least cover this zone's
        valid extent, not collapse to a degenerate geometry.

        Previously there was a bug because the bounds would be computed by
        re-projecting from WebMercator, and after re-projection all the coordinates
        were the same because they were all out of bounds and so somehow got resolved
        to the same coordinate.
        """
        ts = datetime(2020, 7, 15, tzinfo=UTC)
        time_range = (ts, ts + timedelta(days=30))

        data_source = XyzTiles(
            url_templates=["https://example.com/{z}/{x}/{y}.png"],
            time_ranges=[time_range],
            zoom=13,
        )

        # Test various UTM zones (both Northern and Southern hemispheres)
        utm_epsg_codes = [
            32601,  # UTM zone 1N
            32610,  # UTM zone 10N
            32618,  # UTM zone 18N
            32632,  # UTM zone 32N
            32701,  # UTM zone 1S
            32718,  # UTM zone 18S
            32756,  # UTM zone 56S
        ]

        for utm_epsg_code in utm_epsg_codes:
            utm_projection = Projection(CRS.from_epsg(utm_epsg_code), 10, -10)

            bounds = data_source.get_raster_bounds(
                layer_name="test",
                item_name="https://example.com/{z}/{x}/{y}.png",
                bands=["R", "G", "B"],
                projection=utm_projection,
            )

            # UTM zone is about 6 degrees wide at equator, which is ~667km.
            # At 10m resolution, that's about 66,700 pixels minimum.
            # The XyzTiles source should claim to cover at least this extent.
            min_expected_width_and_height = 50000  # Conservative estimate

            assert bounds[2] - bounds[0] >= min_expected_width_and_height, (
                f"Bounds {bounds} does not cover the UTM zone."
            )
            assert bounds[3] - bounds[1] >= min_expected_width_and_height, (
                f"Bounds {bounds} does not cover the UTM zone."
            )
