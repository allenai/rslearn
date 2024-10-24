from rasterio.crs import CRS

from rslearn.utils.get_utm_ups_crs import (
    get_proj_bounds,
    get_utm_ups_crs,
    get_wgs84_bounds,
)


def test_seattle() -> None:
    # Seattle is in UTM 10N which is EPSG:32610
    lon = -122.34
    lat = 47.62
    crs = get_utm_ups_crs(lon, lat)
    assert crs == CRS.from_epsg(32610)

    epsilon = 1e-6
    wgs84_bounds = get_wgs84_bounds(crs)
    expected = [-126, 0, -120, 84]
    assert all([abs(a - b) < epsilon for a, b in zip(wgs84_bounds, expected)])

    epsilon = 1e-2
    proj_bounds = get_proj_bounds(crs)
    # from https://epsg.io/32610
    expected_2 = [166021.44, 0, 833978.56, 9329005.18]
    assert all([abs(a - b) < epsilon for a, b in zip(proj_bounds, expected_2)])


def test_antarctica() -> None:
    # South pole should use UPS South which is EPSG:5042 for (E, N) format.
    lon = -122.34
    lat = -88
    crs = get_utm_ups_crs(lon, lat)
    assert crs == CRS.from_epsg(5042)
