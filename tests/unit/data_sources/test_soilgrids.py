import pytest
from rasterio.crs import CRS

from rslearn.data_sources.soilgrids import (
    VALID_COVERAGE_IDS,
    VALID_SERVICE_IDS,
    SoilGrids,
    _crs_to_rasterio,
    _crs_to_soilgrids_urn,
)


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


def test_crs_to_rasterio_fallback_extracts_epsg_code() -> None:
    # rasterio can't parse this string, but it contains an EPSG code that we can
    # extract in the fallback path.
    assert _crs_to_rasterio("invalid EPSG:4326") == CRS.from_epsg(4326)


def test_crs_to_rasterio_raises_on_unknown() -> None:
    with pytest.raises(Exception):
        _crs_to_rasterio("definitely-not-a-crs")


def test_valid_service_ids_nonempty() -> None:
    assert len(VALID_SERVICE_IDS) > 0


def test_valid_coverage_ids_keyed_by_service_ids() -> None:
    assert set(VALID_COVERAGE_IDS.keys()) == set(VALID_SERVICE_IDS)


def test_valid_coverage_ids_spot_check() -> None:
    assert "clay_0-5cm_mean" in VALID_COVERAGE_IDS["clay"]
    assert "ocs_0-30cm_mean" in VALID_COVERAGE_IDS["ocs"]


def test_invalid_service_id_raises() -> None:
    with pytest.raises(ValueError, match="service_id"):
        SoilGrids(service_id="notaservice", coverage_id="x")


def test_invalid_coverage_id_raises() -> None:
    with pytest.raises(ValueError, match="coverage_id"):
        SoilGrids(service_id="clay", coverage_id="clay_bad")


def test_valid_service_and_coverage_id_accepted() -> None:
    SoilGrids(service_id="clay", coverage_id="clay_0-5cm_mean")
