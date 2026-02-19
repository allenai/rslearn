from rslearn.data_sources.climate_data_store import ERA5LandHourlyTimeseries

TEST_BANDS = ["2m-temperature", "total-precipitation"]


def test_grid_snapping() -> None:
    """Test that coordinates are correctly snapped to 0.1 degree grid."""
    data_source = ERA5LandHourlyTimeseries(band_names=TEST_BANDS)

    snapped_lon, snapped_lat = data_source._snap_to_grid(-122.38, 47.62)
    assert snapped_lon == -122.4
    assert snapped_lat == 47.6

    snapped_lon, snapped_lat = data_source._snap_to_grid(-122.32, 47.67)
    assert snapped_lon == -122.3
    assert snapped_lat == 47.7
