import pytest
from rasterio.crs import CRS

from rslearn.data_sources.soilgrids import _crs_to_rasterio, _crs_to_soilgrids_urn


def test_crs_to_soilgrids_urn_passthrough() -> None:
    assert (
        _crs_to_soilgrids_urn("urn:ogc:def:crs:EPSG::3857")
        == "urn:ogc:def:crs:EPSG::3857"
    )


def test_crs_to_soilgrids_urn_from_epsg_colon() -> None:
    assert _crs_to_soilgrids_urn("EPSG:3857") == "urn:ogc:def:crs:EPSG::3857"
    assert _crs_to_soilgrids_urn("epsg:4326") == "urn:ogc:def:crs:EPSG::4326"


def test_crs_to_soilgrids_urn_from_embedded_code() -> None:
    assert _crs_to_soilgrids_urn("urn:ogc:def:crs:EPSG:6.6:4326") == (
        "urn:ogc:def:crs:EPSG::4326"
    )


def test_crs_to_soilgrids_urn_unknown() -> None:
    assert _crs_to_soilgrids_urn("my-custom-crs") == "my-custom-crs"


def test_crs_to_rasterio_parses_epsg() -> None:
    assert _crs_to_rasterio("EPSG:3857") == CRS.from_epsg(3857)


def test_crs_to_rasterio_parses_urn() -> None:
    assert _crs_to_rasterio("urn:ogc:def:crs:EPSG::4326") == CRS.from_epsg(4326)


def test_crs_to_rasterio_raises_on_unknown() -> None:
    with pytest.raises(Exception):
        _crs_to_rasterio("definitely-not-a-crs")

