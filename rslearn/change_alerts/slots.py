"""Dataset layer configurations for change alert time series.

Each training window is a point in time at the date when a change first becomes
observable. A "slot" is a time series that ends at a fixed offset after that date. With
several slots ending at different offsets, the change appears at different positions
within the time series, which teaches the model to detect both very recent changes and
changes that are a few weeks or months old.

Each slot consists of a frequent layer (e.g. 7-day mosaics) that ends at the slot end,
and optionally an infrequent layer (e.g. 90-day mosaics) that ends at or before the
start of the frequent images that the model will use. The layers are deliberately longer
than the model input so that ChangeTimeSeriesSampler can fall back to older periods when
the most recent ones have no images.
"""

import copy
from datetime import timedelta
from typing import Any

DAY = timedelta(days=1)


def slot_end_offsets(
    num_slots: int, detection_range: timedelta, period: timedelta
) -> list[timedelta]:
    """Compute evenly spaced slot end offsets relative to the change date.

    The first slot ends one period after the change, so the change appears in the
    latest image, and the last slot ends detection_range after the change. The
    offsets are rounded to whole days.

    Args:
        num_slots: the number of slots.
        detection_range: the time range before the end of the time series in which
            the model should detect changes.
        period: the duration of each frequent image.

    Returns:
        the offset from the change date to the end of each slot's time series.
    """
    if num_slots < 1:
        raise ValueError(f"num_slots must be positive, got {num_slots}")
    if detection_range < period:
        raise ValueError("detection_range must be at least one period")
    if num_slots == 1:
        return [period]
    span_days = (detection_range - period) / DAY
    return [
        period + timedelta(days=round(k * span_days / (num_slots - 1)))
        for k in range(num_slots)
    ]


def _format_timedelta(td: timedelta) -> str:
    """Format a timedelta for the dataset config (parsed with pytimeparse)."""
    if td % DAY == timedelta(0):
        return f"{td // DAY}d"
    return f"{int(td.total_seconds())}s"


def _make_layer(
    data_source: dict[str, Any],
    band_sets: list[dict[str, Any]],
    end_offset: timedelta,
    duration: timedelta,
    period: timedelta,
) -> dict[str, Any]:
    """Make a raster layer with one mosaic per period, ending at end_offset."""
    if duration < period:
        raise ValueError("layer duration must be at least one period")
    data_source = copy.deepcopy(data_source)
    query_config = data_source.get("query_config", {})
    query_config.update(
        {
            "space_mode": "MOSAIC",
            "period_duration": _format_timedelta(period),
            "max_matches": duration // period,
            "per_period_mosaic_reverse_time_order": False,
        }
    )
    data_source["query_config"] = query_config
    data_source["time_offset"] = _format_timedelta(end_offset - duration)
    data_source["duration"] = _format_timedelta(duration)
    return {
        "type": "raster",
        "band_sets": copy.deepcopy(band_sets),
        "data_source": data_source,
    }


def make_slot_layer_configs(
    end_offsets: list[timedelta],
    data_source: dict[str, Any],
    band_sets: list[dict[str, Any]],
    frequent_period: timedelta,
    frequent_duration: timedelta,
    infrequent_period: timedelta | None = None,
    infrequent_duration: timedelta | None = None,
    infrequent_end_before: timedelta = timedelta(0),
    frequent_layer_name: str = "frequent_{slot}",
    infrequent_layer_name: str = "infrequent_{slot}",
    slot_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Make the dataset layer configs for a set of slots.

    The window time range should be (change date, change date), since the layer time
    offsets are relative to the start of the window time range.

    Args:
        end_offsets: the offset from the change date to the end of each slot's time
            series, e.g. from slot_end_offsets.
        data_source: the data source config (as in the dataset config.json) to use for
            each layer. time_offset, duration, and some query_config options are
            overwritten.
        band_sets: the band sets for each layer.
        frequent_period: the duration of each frequent mosaic.
        frequent_duration: the total duration of the frequent layer, ending at the
            slot end.
        infrequent_period: the duration of each infrequent mosaic. If None, no
            infrequent layers are created.
        infrequent_duration: the total duration of the infrequent layer.
        infrequent_end_before: the infrequent layer ends this long before the slot
            end. This should match the lookback that the sampler uses to select
            frequent images so that the infrequent images end where the frequent
            images begin.
        frequent_layer_name: format string for frequent layer names, with a {slot}
            placeholder.
        infrequent_layer_name: format string for infrequent layer names.
        slot_names: the names to substitute for {slot}, defaults to the slot index.

    Returns:
        map from layer name to layer config dict.
    """
    if slot_names is None:
        slot_names = [str(idx) for idx in range(len(end_offsets))]
    if len(slot_names) != len(end_offsets):
        raise ValueError("slot_names must have the same length as end_offsets")
    if (infrequent_period is None) != (infrequent_duration is None):
        raise ValueError(
            "infrequent_period and infrequent_duration must be set together"
        )

    layers: dict[str, dict[str, Any]] = {}
    for slot_name, end_offset in zip(slot_names, end_offsets):
        layers[frequent_layer_name.format(slot=slot_name)] = _make_layer(
            data_source, band_sets, end_offset, frequent_duration, frequent_period
        )
        if infrequent_period is not None and infrequent_duration is not None:
            layers[infrequent_layer_name.format(slot=slot_name)] = _make_layer(
                data_source,
                band_sets,
                end_offset - infrequent_end_before,
                infrequent_duration,
                infrequent_period,
            )
    return layers
