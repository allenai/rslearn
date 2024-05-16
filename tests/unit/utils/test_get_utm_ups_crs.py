from rasterio.crs import CRS

from rslearn.utils import get_utm_ups_crs

def test_seattle():
    # Seattle is in UTM 10N which is EPSG:32610
    lon = -122.34
    lat = 47.62
    crs = get_utm_ups_crs(lon, lat)
    assert crs == CRS.from_epsg(32610)

def test_antarctica():
    # South pole should use UPS South which is EPSG:5042 for (E, N) format.
    lon = -122.34
    lat = -88
    crs = get_utm_ups_crs(lon, lat)
    assert crs == CRS.from_epsg(5042)
