"""Sample one change alert time series per example."""

import random
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import torch

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.transform import Transform

UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def midpoint_day(time_range: tuple[datetime, datetime]) -> int:
    """Get the days since 1970-01-01 of the midpoint of a time range.

    This matches how PerPixelTimestepHead dates its input timesteps.
    """
    start, end = time_range
    return (start + (end - start) / 2 - UNIX_EPOCH) // timedelta(days=1)


def _clear_valid(target: dict[str, Any], mask: torch.Tensor) -> None:
    """Clear the valid mask of a target at the HW mask."""
    valid = target["valid"]
    new_valid = valid.image.clone()
    new_valid[:, :, mask] = 0
    target["valid"] = RasterImage(new_valid, timestamps=valid.timestamps)


class ChangeTimeSeriesSampler(Transform):
    """Build one time series from consecutive frequent images and infrequent history.

    The frequent input (e.g. 7-day mosaics) and the optional infrequent input (e.g.
    90-day mosaics) typically correspond to the layers from
    rslearn.change_alerts.layers.make_layer_configs. They should be loaded with
    load_all_item_groups and load_all_layers so that each is a RasterImage with one
    timestep per item group.

    The time series is built as follows:

    1. Pick num_frequent consecutive frequent images (all of them if there are fewer):
       starting at a random image with select="random" (training), or the latest ones
       with select="latest" (prediction).
    2. With select="random", keep all of the picked images with probability
       full_frequent_probability, and otherwise only the latest n of them, with n
       uniform between 1 and num_frequent - 1. This simulates prediction windows where
       some frequent periods have no images.
    3. Take the latest num_infrequent infrequent images that end at or before the start
       of the pick. If stand_in_min_gap is set and the pick starts at least that long
       after the first frequent image, the frequent image just before the pick is used
       as a stand-in for the most recent infrequent image, replacing the oldest one.
       The frequent images dropped in step 2 are never used as the stand-in.
    4. Concatenate the infrequent images (and stand-in) and the kept frequent images
       chronologically into input_dict[output_key].

    The timestep at which the change appears is predicted with PerPixelTimestepTask
    and PerPixelTimestepHead (with mode AFTER), which map the change day labels to the
    timesteps of the time series built here.

    If change dates vary within a window, set category_target and change_day_target
    so that pixels whose change is outside the time series (e.g. after the latest
    image) are marked invalid for both the category and the timestep targets, since
    the change cannot be observed in the input. Pixels without a change day (e.g.
    negatives) are not affected, so they keep their category target.
    """

    def __init__(
        self,
        frequent_key: str,
        num_frequent: int,
        infrequent_key: str | None = None,
        num_infrequent: int = 0,
        output_key: str = "image",
        select: Literal["random", "latest"] = "random",
        full_frequent_probability: float = 0.5,
        stand_in_min_gap: timedelta | None = None,
        category_target: str | None = None,
        change_day_target: str | None = None,
    ) -> None:
        """Create a new ChangeTimeSeriesSampler.

        Args:
            frequent_key: the input key of the frequent input.
            num_frequent: the number of frequent images to pick.
            infrequent_key: the input key of the infrequent input, required if
                num_infrequent > 0.
            num_infrequent: the number of infrequent images to use.
            output_key: the input key to write the time series to.
            select: "random" to pick consecutive frequent images at a random position
                (for training and evaluation), or "latest" to pick the latest ones
                (for prediction).
            full_frequent_probability: with select="random", the probability of
                keeping all of the picked frequent images. Otherwise fewer are kept.
                Set to 1 to always keep all of them.
            stand_in_min_gap: if set, use a stand-in for the most recent infrequent
                image when the pick starts at least this long after the first frequent
                image, e.g. half of the infrequent period. The gap is measured from the
                first frequent image rather than from the infrequent images, since
                mosaic timestamps are the time of a single scene within the period, and
                the frequent layer starts where the infrequent layer ends. If None, no
                stand-in is used.
            category_target: the category target whose valid mask is cleared at
                pixels whose change is outside the time series. Must be set together
                with change_day_target. This is only needed if change dates vary
                within a window.
            change_day_target: the PerPixelTimestepTask target containing the change
                day labels ("days" and "valid"). Its valid mask is also cleared at
                pixels whose change is outside the time series.
        """
        super().__init__()
        if num_frequent < 1:
            raise ValueError(f"num_frequent must be positive, got {num_frequent}")
        if num_infrequent > 0 and infrequent_key is None:
            raise ValueError("infrequent_key is required since num_infrequent > 0")
        if select not in ("random", "latest"):
            raise ValueError(f"select must be random or latest, got {select}")
        if not 0 <= full_frequent_probability <= 1:
            raise ValueError("full_frequent_probability must be between 0 and 1")
        if (category_target is None) != (change_day_target is None):
            raise ValueError(
                "category_target and change_day_target must be set together"
            )
        self.frequent_key = frequent_key
        self.num_frequent = num_frequent
        self.infrequent_key = infrequent_key
        self.num_infrequent = num_infrequent
        self.output_key = output_key
        self.select = select
        self.full_frequent_probability = full_frequent_probability
        self.stand_in_min_gap = stand_in_min_gap
        self.category_target = category_target
        self.change_day_target = change_day_target

    def _num_keep(self, num_picked: int) -> int:
        """Sample how many of the picked frequent images to keep."""
        if (
            self.select == "latest"
            or self.num_frequent == 1
            or random.random() < self.full_frequent_probability
        ):
            return num_picked
        return min(random.randint(1, self.num_frequent - 1), num_picked)

    def _select_history(
        self, frequent: RasterImage, pick_start_idx: int, input_dict: dict[str, Any]
    ) -> list[tuple[RasterImage, int]]:
        """Select the infrequent images (and stand-in) before the pick.

        Args:
            frequent: the frequent input.
            pick_start_idx: the index of the first picked frequent image.
            input_dict: the input dict, containing the infrequent input.

        Returns:
            the chronological list of (image, timestep index) to use as history.
        """
        if self.num_infrequent == 0:
            return []
        assert self.infrequent_key is not None
        assert frequent.timestamps is not None
        infrequent = input_dict[self.infrequent_key]
        if infrequent.timestamps is None:
            raise ValueError(
                f"infrequent input {self.infrequent_key} has no timestamps"
            )

        pick_start = frequent.timestamps[pick_start_idx][0]
        eligible = [
            idx
            for idx, (_, end) in enumerate(infrequent.timestamps)
            if end <= pick_start
        ]
        history = [(infrequent, idx) for idx in eligible[-self.num_infrequent :]]
        if (
            self.stand_in_min_gap is None
            or pick_start_idx == 0
            or pick_start - frequent.timestamps[0][0] < self.stand_in_min_gap
        ):
            return history

        stand_in_idx = pick_start_idx - 1
        # Keep the time series chronological.
        if eligible and (
            frequent.timestamps[stand_in_idx][0]
            < infrequent.timestamps[eligible[-1]][1]
        ):
            return history
        if len(history) == self.num_infrequent:
            history = history[1:]
        history.append((frequent, stand_in_idx))
        return history

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build the time series and optionally mask the targets.

        Args:
            input_dict: the input dict, containing the frequent and infrequent inputs.
            target_dict: the target dict.

        Returns:
            the updated (input_dict, target_dict).
        """
        frequent = input_dict.get(self.frequent_key)
        if frequent is None or not frequent.timestamps:
            raise ValueError(
                f"no images available for frequent input {self.frequent_key}"
            )

        count = len(frequent.timestamps)
        num_picked = min(self.num_frequent, count)
        if self.select == "latest":
            pick_start_idx = count - num_picked
        else:
            pick_start_idx = random.randint(0, count - num_picked)
        pick_end_idx = pick_start_idx + num_picked
        num_keep = self._num_keep(num_picked)

        selected = self._select_history(frequent, pick_start_idx, input_dict)
        selected += [
            (frequent, idx) for idx in range(pick_end_idx - num_keep, pick_end_idx)
        ]

        image = torch.cat(
            [raster.image[:, idx : idx + 1] for raster, idx in selected], dim=1
        )
        timestamps = [raster.timestamps[idx] for raster, idx in selected]  # type: ignore[index]
        input_dict[self.output_key] = RasterImage(image, timestamps=timestamps)

        # During prediction with MultiTask, the per-task target dicts are present but
        # empty.
        if (
            self.category_target is not None
            and self.change_day_target is not None
            and target_dict.get(self.category_target)
            and target_dict.get(self.change_day_target)
        ):
            self._mask_outside(timestamps, target_dict)

        return input_dict, target_dict

    def _mask_outside(
        self,
        timestamps: list[tuple[datetime, datetime]],
        target_dict: dict[str, Any],
    ) -> None:
        """Mark pixels whose change is outside the time series invalid.

        This uses the same day granularity as PerPixelTimestepHead with mode AFTER: a
        change is inside the time series if it is on or after the start day of the
        first image, and on or before the midpoint day of the last image.

        Args:
            timestamps: the chronological time ranges of the images in the time series
                built by the sampler (the timestamps of input_dict[output_key]).
            target_dict: the target dict, whose category_target and change_day_target
                valid masks are updated in place.
        """
        assert self.category_target is not None
        assert self.change_day_target is not None
        change_target = target_dict[self.change_day_target]
        days = change_target["days"].get_hw_tensor().long()
        has_change = change_target["valid"].get_hw_tensor() > 0

        first_start_day = (timestamps[0][0] - UNIX_EPOCH) // timedelta(days=1)
        last_day = max(midpoint_day(time_range) for time_range in timestamps)
        outside = has_change & ((days < first_start_day) | (days > last_day))
        if not outside.any():
            return

        _clear_valid(target_dict[self.category_target], outside)
        _clear_valid(change_target, outside)
