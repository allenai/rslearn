"""Window time ranges and dataset layer configurations for change alerts.

The same dataset config is used for training and prediction, with windows whose time
ranges are computed by ChangeAlertTiming:

- A training window spans (change date - (num_frequent - 1) * frequent period, change
  date + detection range). Any num_frequent consecutive frequent images within it end
  between one frequent period and the detection range after the change date, so
  ChangeTimeSeriesSampler can pick any of them.
- A prediction window spans the latest num_frequent frequent periods before the time
  to alert on.

The frequent layer (e.g. 7-day mosaics) covers the window time range. The optional
infrequent layer (e.g. 90-day mosaics) ends at the window start and extends back by the
infrequent history duration.
"""

import copy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

DAY = timedelta(days=1)


@dataclass(frozen=True)
class ChangeAlertTiming:
    """The timing of the change alert time series.

    Attributes:
        frequent_period: the duration of each frequent image.
        num_frequent: the number of frequent images that the model uses.
        detection_range: how long after the change date a change should still be
            detected, i.e. the latest offset from the change date to the end of the
            time series. It must be a multiple of frequent_period, so that a frequent
            period starts exactly at the change date.
    """

    frequent_period: timedelta
    num_frequent: int
    detection_range: timedelta

    def __post_init__(self) -> None:
        """Validate the timing."""
        if self.frequent_period <= timedelta(0):
            raise ValueError("frequent_period must be positive")
        if self.num_frequent < 1:
            raise ValueError(f"num_frequent must be positive, got {self.num_frequent}")
        if self.detection_range < self.frequent_period:
            raise ValueError("detection_range must be at least one frequent period")
        if self.detection_range % self.frequent_period != timedelta(0):
            raise ValueError(
                f"detection_range ({self.detection_range}) must be a multiple of "
                f"frequent_period ({self.frequent_period})"
            )

    @property
    def frequent_max_matches(self) -> int:
        """The number of frequent periods in a training window."""
        duration = self.detection_range + (self.num_frequent - 1) * self.frequent_period
        return duration // self.frequent_period

    def train_time_range(self, change_date: datetime) -> tuple[datetime, datetime]:
        """Get the time range of a training window.

        Args:
            change_date: the date when the change first becomes observable. For
                windows without a change (negatives), any date can be used.

        Returns:
            the window time range.
        """
        return (
            change_date - (self.num_frequent - 1) * self.frequent_period,
            change_date + self.detection_range,
        )

    def predict_time_range(self, time: datetime) -> tuple[datetime, datetime]:
        """Get the time range of a prediction window.

        Args:
            time: the time to alert on, e.g. now.

        Returns:
            the window time range, covering the latest num_frequent frequent periods.
        """
        return (time - self.num_frequent * self.frequent_period, time)


def _format_timedelta(td: timedelta) -> str:
    """Format a timedelta for the dataset config (parsed with pytimeparse)."""
    if td % DAY == timedelta(0):
        return f"{td // DAY}d"
    return f"{int(td.total_seconds())}s"


def _make_layer(
    data_source: dict[str, Any],
    band_sets: list[dict[str, Any]],
    period: timedelta,
    max_matches: int,
    time_offset: timedelta | None = None,
    duration: timedelta | None = None,
    min_matches: int | None = None,
) -> dict[str, Any]:
    """Make a raster layer with one mosaic per period.

    The layer holds the latest max_matches periods that have images, skipping empty
    periods. If time_offset or duration is set, it overrides the data source's
    time_offset or duration. If min_matches is set, it overrides the data source's
    min_matches.
    """
    if max_matches < 1:
        raise ValueError(f"max_matches must be positive, got {max_matches}")
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
    if time_offset is not None:
        data_source["time_offset"] = _format_timedelta(time_offset)
    if duration is not None:
        data_source["duration"] = _format_timedelta(duration)
    return {
        "type": "raster",
        "band_sets": copy.deepcopy(band_sets),
        "data_source": data_source,
    }


def make_layer_configs(
    timing: ChangeAlertTiming,
    data_source: dict[str, Any],
    band_sets: list[dict[str, Any]],
    infrequent_period: timedelta | None = None,
    infrequent_duration: timedelta | None = None,
    frequent_layer_name: str = "frequent",
    infrequent_layer_name: str = "infrequent",
) -> dict[str, dict[str, Any]]:
    """Make the frequent and infrequent dataset layer configs.

    The layers are used for both training and prediction windows, whose time ranges
    should come from timing.train_time_range and timing.predict_time_range.

    Args:
        timing: the timing of the time series.
        data_source: the data source config (as in the dataset config.json) to use for
            each layer. time_offset, duration, and some query_config options are
            overwritten.
        band_sets: the band sets for each layer.
        infrequent_period: the duration of each infrequent mosaic. If None, no
            infrequent layer is created.
        infrequent_duration: the infrequent history duration, i.e. how far back the
            infrequent layer extends from the window start. It must be a multiple of
            infrequent_period.
        frequent_layer_name: the name of the frequent layer.
        infrequent_layer_name: the name of the infrequent layer.

    Returns:
        map from layer name to layer config dict.
    """
    if (infrequent_period is None) != (infrequent_duration is None):
        raise ValueError(
            "infrequent_period and infrequent_duration must be set together"
        )

    data_source = copy.deepcopy(data_source)
    # The frequent layer uses the window time range itself.
    data_source.pop("time_offset", None)
    data_source.pop("duration", None)

    layers: dict[str, dict[str, Any]] = {}
    layers[frequent_layer_name] = _make_layer(
        data_source,
        band_sets,
        timing.frequent_period,
        timing.frequent_max_matches,
        min_matches=1,
    )
    if infrequent_period is not None and infrequent_duration is not None:
        if infrequent_duration < infrequent_period:
            raise ValueError("infrequent_duration must be at least one period")
        if infrequent_duration % infrequent_period != timedelta(0):
            raise ValueError(
                f"infrequent_duration ({infrequent_duration}) must be a multiple of "
                f"infrequent_period ({infrequent_period})"
            )
        layers[infrequent_layer_name] = _make_layer(
            data_source,
            band_sets,
            infrequent_period,
            infrequent_duration // infrequent_period,
            time_offset=-infrequent_duration,
            duration=infrequent_duration,
        )
    return layers
