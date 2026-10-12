"""Dataset layer configurations for change alert time series.

Each training window is a point in time at the date when a change first becomes
observable. A "slot" is a time series that ends at a fixed offset after that date. With
several slots ending at different offsets, the change appears at different positions
within the time series, which teaches the model to detect both very recent changes and
changes that are a few weeks or months old.

Each slot has a frequent layer (e.g. 7-day mosaics) that ends at the slot end. The slots
can optionally share one infrequent layer (e.g. 90-day mosaics) that ends where the
latest slot's frequent layer starts and extends back far enough to provide history for
every slot; ChangeTimeSeriesSampler only uses the infrequent images that end before the
selected frequent images.

Each frequent layer spans more periods than the number of mosaics it holds, so that
when the most recent periods have no images, it falls back to older ones.
"""

import copy
import math
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
    max_matches: int,
    min_matches: int | None = None,
) -> dict[str, Any]:
    """Make a raster layer with one mosaic per period, ending at end_offset.

    The layer holds the latest max_matches periods that have images, skipping empty
    periods. If min_matches is set, it overrides the data source's min_matches.
    """
    if duration < period:
        raise ValueError("layer duration must be at least one period")
    if not 1 <= max_matches <= duration // period:
        raise ValueError(
            f"max_matches must be between 1 and the number of periods "
            f"({duration // period}), got {max_matches}"
        )
    data_source = copy.deepcopy(data_source)
    query_config = data_source.get("query_config", {})
    query_config.update(
        {
            "space_mode": "MOSAIC",
            "period_duration": _format_timedelta(period),
            "max_matches": max_matches,
            "per_period_mosaic_reverse_time_order": False,
        }
    )
    if min_matches is not None:
        query_config["min_matches"] = min_matches
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
    frequent_max_matches: int,
    infrequent_period: timedelta | None = None,
    infrequent_duration: timedelta | None = None,
    frequent_layer_name: str = "frequent_{slot}",
    infrequent_layer_name: str = "infrequent",
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
            slot end. This is how far back to look for frequent images when the
            most recent periods have none.
        frequent_max_matches: the number of frequent images that the model uses,
            i.e. the latest this many periods within frequent_duration that have
            images are materialized per slot. Frequent layers also get
            min_matches=1, so slots without any frequent images are reported as
            rejected during prepare.
        infrequent_period: the duration of each infrequent mosaic. If None, no
            infrequent layer is created.
        infrequent_duration: the duration of infrequent history to provide for each
            slot. The shared infrequent layer ends at the start of the latest slot's
            frequent layer, and is extended by whole periods so that the earliest
            slot also has this much history.
        frequent_layer_name: format string for frequent layer names, with a {slot}
            placeholder.
        infrequent_layer_name: the name of the shared infrequent layer.
        slot_names: the names to substitute for {slot}, defaults to the slot index.

    Returns:
        map from layer name to layer config dict.
    """
    if not end_offsets:
        raise ValueError("at least one slot end offset is required")
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
            data_source,
            band_sets,
            end_offset,
            frequent_duration,
            frequent_period,
            frequent_max_matches,
            min_matches=1,
        )
    if infrequent_period is not None and infrequent_duration is not None:
        # We use one shared infrequent layer for all of the frequent periods. So it
        # should have enough images so that we can take a subset of it to match with
        # any selected frequent option. That means its duration should be the given
        # infrequent duration, plus additional infrequent periods corresponding to the
        # spread between the earliest and latest frequent options.
        spread = max(end_offsets) - min(end_offsets)
        num_extra = math.ceil(spread / infrequent_period)
        duration = infrequent_duration + num_extra * infrequent_period
        layers[infrequent_layer_name] = _make_layer(
            data_source,
            band_sets,
            max(end_offsets) - frequent_duration,
            duration,
            infrequent_period,
            duration // infrequent_period,
        )
    return layers
