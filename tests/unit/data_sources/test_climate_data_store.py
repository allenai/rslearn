from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
import shapely

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.climate_data_store import ERA5Land, ERA5LandHourlyTimeseries
from rslearn.data_sources.utils import MatchedItemGroup
from rslearn.utils.geometry import STGeometry

TEST_BANDS = ["2m-temperature", "total-precipitation"]


@patch("rslearn.data_sources.climate_data_store.cdsapi.Client")
def test_grid_snapping(mock_client: MagicMock) -> None:
    """Test that coordinates are correctly snapped to 0.1 degree grid."""
    data_source = ERA5LandHourlyTimeseries(band_names=TEST_BANDS)

    snapped_lon, snapped_lat = data_source._snap_to_grid(-122.38, 47.62)
    assert snapped_lon == -122.4
    assert snapped_lat == 47.6

    snapped_lon, snapped_lat = data_source._snap_to_grid(-122.32, 47.67)
    assert snapped_lon == -122.3
    assert snapped_lat == 47.7


@patch("rslearn.data_sources.climate_data_store.cdsapi.Client")
def test_era5_land_rejects_min_matches(mock_client: MagicMock) -> None:
    """ERA5Land custom matching should reject min_matches>0 explicitly."""
    data_source = ERA5Land(
        dataset="reanalysis-era5-land",
        product_type="reanalysis",
        band_names=TEST_BANDS,
    )
    geometry = STGeometry(
        WGS84_PROJECTION,
        shapely.box(-1, -1, 1, 1),
        (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC)),
    )

    with pytest.raises(ValueError, match="min_matches"):
        data_source.get_items(
            [geometry],
            query_config=QueryConfig(space_mode=SpaceMode.MOSAIC, min_matches=1),
        )


@patch("rslearn.data_sources.climate_data_store.cdsapi.Client")
def test_era5_land_hourly_timeseries_rejects_min_matches(
    mock_client: MagicMock,
) -> None:
    """ERA5LandHourlyTimeseries custom matching should reject min_matches>0."""
    data_source = ERA5LandHourlyTimeseries(band_names=TEST_BANDS)
    geometry = STGeometry(
        WGS84_PROJECTION,
        shapely.box(-1, -1, 1, 1),
        (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC)),
    )

    with pytest.raises(ValueError, match="min_matches"):
        data_source.get_items(
            [geometry],
            query_config=QueryConfig(space_mode=SpaceMode.MOSAIC, min_matches=1),
        )


@patch("rslearn.data_sources.climate_data_store.cdsapi.Client")
def test_era5_land_returns_matched_item_groups(mock_client: MagicMock) -> None:
    """ERA5Land should return MatchedItemGroup instances."""
    data_source = ERA5Land(
        dataset="reanalysis-era5-land",
        product_type="reanalysis",
        band_names=TEST_BANDS,
    )
    geometry = STGeometry(
        WGS84_PROJECTION,
        shapely.box(-1, -1, 1, 1),
        (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC)),
    )

    groups = data_source.get_items(
        [geometry],
        query_config=QueryConfig(space_mode=SpaceMode.SINGLE_COMPOSITE),
    )
    assert len(groups) == 1
    assert len(groups[0]) == 1
    assert isinstance(groups[0][0], MatchedItemGroup)
    assert groups[0][0].request_time_range == geometry.time_range
