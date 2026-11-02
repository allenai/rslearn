"""Unit tests for rslearn.change_alerts.layers."""

from datetime import UTC, datetime, timedelta

import pytest

from rslearn.change_alerts.layers import ChangeAlertTiming, make_layer_configs
from rslearn.config.dataset import LayerConfig

DATA_SOURCE = {
    "class_path": "rslearn.data_sources.planetary_computer.Sentinel2",
    "query_config": {"max_matches": 1},
}
BAND_SETS = [{"bands": ["B02", "B03", "B04"], "dtype": "uint16"}]
TIMING = ChangeAlertTiming(
    frequent_period=timedelta(days=7),
    num_frequent=4,
    detection_range=timedelta(days=91),
)
CHANGE = datetime(2024, 5, 10, tzinfo=UTC)


def test_train_time_range() -> None:
    """The training window starts N-1 periods before the change date."""
    assert TIMING.train_time_range(CHANGE) == (
        CHANGE - timedelta(days=21),
        CHANGE + timedelta(days=91),
    )
    assert TIMING.frequent_max_matches == 16


def test_predict_time_range() -> None:
    """The prediction window covers the latest N periods."""
    assert TIMING.predict_time_range(CHANGE) == (CHANGE - timedelta(days=28), CHANGE)


def test_timing_invalid() -> None:
    """The detection range must be a positive multiple of the frequent period."""
    with pytest.raises(ValueError):
        ChangeAlertTiming(timedelta(days=7), 4, timedelta(days=90))
    with pytest.raises(ValueError):
        ChangeAlertTiming(timedelta(days=7), 4, timedelta(days=0))
    with pytest.raises(ValueError):
        ChangeAlertTiming(timedelta(days=7), 0, timedelta(days=91))


def test_make_layer_configs() -> None:
    """Frequent and infrequent layers cover the expected request time ranges."""
    layers = make_layer_configs(
        TIMING,
        data_source=DATA_SOURCE,
        band_sets=BAND_SETS,
        infrequent_period=timedelta(days=90),
        infrequent_duration=timedelta(days=720),
    )
    assert set(layers.keys()) == {"frequent", "infrequent"}
    frequent = LayerConfig.model_validate(layers["frequent"])
    infrequent = LayerConfig.model_validate(layers["infrequent"])
    assert frequent.data_source is not None
    assert infrequent.data_source is not None

    query_config = frequent.data_source.query_config
    assert query_config.period_duration == timedelta(days=7)
    assert query_config.max_matches == 16
    assert query_config.min_matches == 1
    assert not query_config.per_period_mosaic_reverse_time_order
    assert infrequent.data_source.query_config.period_duration == timedelta(days=90)
    assert infrequent.data_source.query_config.max_matches == 8
    assert infrequent.data_source.query_config.min_matches == 0

    # For both training and prediction windows, the frequent layer covers the window
    # and the infrequent layer ends at the window start.
    for window_time_range in [
        TIMING.train_time_range(CHANGE),
        TIMING.predict_time_range(CHANGE),
    ]:
        start = window_time_range[0]
        assert (
            frequent.data_source.get_request_time_range(window_time_range)
            == window_time_range
        )
        assert infrequent.data_source.get_request_time_range(window_time_range) == (
            start - timedelta(days=720),
            start,
        )

    # The input data source config should not be modified.
    assert "time_offset" not in DATA_SOURCE
    assert DATA_SOURCE["query_config"] == {"max_matches": 1}


def test_make_layer_configs_frequent_only() -> None:
    """No infrequent layer is created when infrequent_period is None."""
    layers = make_layer_configs(
        TIMING,
        data_source={**DATA_SOURCE, "time_offset": "1d"},
        band_sets=BAND_SETS,
        frequent_layer_name="s2_freq",
    )
    assert list(layers.keys()) == ["s2_freq"]
    assert "time_offset" not in layers["s2_freq"]["data_source"]


def test_make_layer_configs_invalid_infrequent() -> None:
    """The infrequent duration must be a multiple of the infrequent period."""
    with pytest.raises(ValueError):
        make_layer_configs(
            TIMING,
            data_source=DATA_SOURCE,
            band_sets=BAND_SETS,
            infrequent_period=timedelta(days=90),
            infrequent_duration=timedelta(days=700),
        )
    with pytest.raises(ValueError):
        make_layer_configs(
            TIMING,
            data_source=DATA_SOURCE,
            band_sets=BAND_SETS,
            infrequent_period=timedelta(days=90),
        )
