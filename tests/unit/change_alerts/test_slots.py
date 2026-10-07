"""Unit tests for rslearn.change_alerts.slots."""

from datetime import UTC, datetime, timedelta

import pytest

from rslearn.change_alerts.slots import make_slot_layer_configs, slot_end_offsets
from rslearn.config.dataset import LayerConfig

DATA_SOURCE = {
    "class_path": "rslearn.data_sources.planetary_computer.Sentinel2",
    "query_config": {"min_matches": 1},
}
BAND_SETS = [{"bands": ["B02", "B03", "B04"], "dtype": "uint16"}]


def test_slot_end_offsets() -> None:
    """Slots span from one period to the detection range, rounded to days."""
    offsets = slot_end_offsets(4, timedelta(days=90), timedelta(days=7))
    assert offsets == [timedelta(days=d) for d in (7, 35, 62, 90)]


def test_slot_end_offsets_single() -> None:
    """With one slot, the change appears in the latest image."""
    assert slot_end_offsets(1, timedelta(days=90), timedelta(days=7)) == [
        timedelta(days=7)
    ]


def test_slot_end_offsets_invalid() -> None:
    """The detection range must cover at least one period."""
    with pytest.raises(ValueError):
        slot_end_offsets(4, timedelta(days=3), timedelta(days=7))


def test_make_slot_layer_configs() -> None:
    """Frequent and infrequent layers cover the expected request time ranges."""
    layers = make_slot_layer_configs(
        end_offsets=[timedelta(days=7), timedelta(days=45)],
        data_source=DATA_SOURCE,
        band_sets=BAND_SETS,
        frequent_period=timedelta(days=7),
        frequent_duration=timedelta(days=63),
        infrequent_period=timedelta(days=90),
        infrequent_duration=timedelta(days=900),
        infrequent_end_before=timedelta(days=60),
        slot_names=["7", "45"],
    )
    assert set(layers.keys()) == {
        "frequent_7",
        "infrequent_7",
        "frequent_45",
        "infrequent_45",
    }

    change = datetime(2024, 5, 10, tzinfo=UTC)
    window_time_range = (change, change)

    frequent = LayerConfig.model_validate(layers["frequent_45"])
    assert frequent.data_source is not None
    assert frequent.data_source.get_request_time_range(window_time_range) == (
        change + timedelta(days=45 - 63),
        change + timedelta(days=45),
    )
    query_config = frequent.data_source.query_config
    assert query_config.period_duration == timedelta(days=7)
    assert query_config.max_matches == 9
    assert query_config.min_matches == 1
    assert not query_config.per_period_mosaic_reverse_time_order

    infrequent = LayerConfig.model_validate(layers["infrequent_45"])
    assert infrequent.data_source is not None
    assert infrequent.data_source.get_request_time_range(window_time_range) == (
        change + timedelta(days=45 - 60 - 900),
        change + timedelta(days=45 - 60),
    )
    assert infrequent.data_source.query_config.max_matches == 10

    # The input data source config should not be modified.
    assert "time_offset" not in DATA_SOURCE


def test_make_slot_layer_configs_frequent_only() -> None:
    """No infrequent layers are created when infrequent_period is None."""
    layers = make_slot_layer_configs(
        end_offsets=[timedelta(days=7)],
        data_source=DATA_SOURCE,
        band_sets=BAND_SETS,
        frequent_period=timedelta(days=7),
        frequent_duration=timedelta(days=180),
        frequent_layer_name="s2_freq_{slot}",
    )
    assert list(layers.keys()) == ["s2_freq_0"]
