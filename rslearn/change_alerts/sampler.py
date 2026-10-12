"""Sample one change alert time series per example."""

import random
from datetime import UTC, datetime, timedelta
from typing import Any

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
    """Build one time series from a choice of frequent inputs.

    The frequent inputs (e.g. 7-day mosaics) typically correspond to the per-slot
    frequent layers from make_slot_layer_configs, and the optional infrequent input
    (e.g. 90-day mosaics) to the infrequent layer shared by all slots. The inputs
    should be loaded with load_all_item_groups and load_all_layers so that each is a
    RasterImage with one timestep per item group.

    The sampler picks the frequent input with the most images, randomly among ties (or
    frequent_index if set). Frequent inputs that are missing from the input dict (e.g.
    non-required inputs whose layer has no images) count as having zero images. It
    then takes all of the selected frequent images, preceded by the latest
    num_infrequent infrequent images that end before the first frequent image. These
    are concatenated chronologically into input_dict[output_key].

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
        frequent_keys: list[str],
        infrequent_key: str | None = None,
        num_infrequent: int = 0,
        output_key: str = "image",
        category_target: str | None = None,
        change_day_target: str | None = None,
        frequent_index: int | None = None,
    ) -> None:
        """Create a new ChangeTimeSeriesSampler.

        Args:
            frequent_keys: the input keys of the frequent inputs to choose from.
            infrequent_key: the input key of the infrequent input, required if
                num_infrequent > 0.
            num_infrequent: the number of infrequent images to use.
            output_key: the input key to write the time series to.
            category_target: the category target whose valid mask is cleared at
                pixels whose change is outside the time series. Must be set together
                with change_day_target. This is only needed if change dates vary
                within a window.
            change_day_target: the PerPixelTimestepTask target containing the change
                day labels ("days" and "valid"). Its valid mask is also cleared at
                pixels whose change is outside the time series.
            frequent_index: if set, always use this frequent input (e.g. for
                evaluation or prediction) instead of picking one. An error is raised
                if it has no images.
        """
        super().__init__()
        if not frequent_keys:
            raise ValueError("at least one frequent key is required")
        if num_infrequent > 0 and infrequent_key is None:
            raise ValueError("infrequent_key is required since num_infrequent > 0")
        if frequent_index is not None and not 0 <= frequent_index < len(frequent_keys):
            raise ValueError(f"frequent_index {frequent_index} is out of range")
        if (category_target is None) != (change_day_target is None):
            raise ValueError(
                "category_target and change_day_target must be set together"
            )
        self.frequent_keys = frequent_keys
        self.infrequent_key = infrequent_key
        self.num_infrequent = num_infrequent
        self.output_key = output_key
        self.category_target = category_target
        self.change_day_target = change_day_target
        self.frequent_index = frequent_index

    def _select(
        self, frequent_key: str, input_dict: dict[str, Any]
    ) -> tuple[list[tuple[RasterImage, int]], int]:
        """Select the images to use with one frequent input.

        Returns:
            a tuple (selected, frequent_count) where selected is the chronological
            list of (image, timestep index) to use and frequent_count is the number
            of frequent images among them.
        """
        frequent = input_dict.get(frequent_key)
        if frequent is None or not frequent.timestamps:
            return [], 0

        selected = [(frequent, idx) for idx in range(len(frequent.timestamps))]

        if self.num_infrequent > 0:
            assert self.infrequent_key is not None
            infrequent = input_dict[self.infrequent_key]
            if infrequent.timestamps is None:
                raise ValueError(
                    f"infrequent input {self.infrequent_key} has no timestamps"
                )
            # Strict inequality so that a scene shared with the first frequent image
            # is not duplicated, since OlmoEarth requires distinct timesteps.
            first_start = frequent.timestamps[0][0]
            infrequent_idxs = [
                idx
                for idx, (_, end) in enumerate(infrequent.timestamps)
                if end < first_start
            ][-self.num_infrequent :]
            selected = [(infrequent, idx) for idx in infrequent_idxs] + selected

        return selected, len(frequent.timestamps)

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
        if self.frequent_index is not None:
            frequent_key = self.frequent_keys[self.frequent_index]
            selected, _ = self._select(frequent_key, input_dict)
            if not selected:
                raise ValueError(
                    f"no images available for frequent input {frequent_key}"
                )
        else:
            # Randomly choose from only the frequent options that have the most images.
            # Normally all frequent options would have num_frequent images, but some
            # could end up with fewer images if there wasn't coverage, and we don't use
            # those options when possible. But other windows might only have frequent
            # options with fewer than num_frequent images, so if we have to we still
            # train on those.
            candidates = [self._select(key, input_dict) for key in self.frequent_keys]
            max_count = max(count for _, count in candidates)
            if max_count == 0:
                raise ValueError(
                    f"no images available for any frequent input among {self.frequent_keys}"
                )
            selected = random.choice(
                [sel for sel, count in candidates if count == max_count]
            )

        image = torch.cat(
            [raster.image[:, idx : idx + 1] for raster, idx in selected], dim=1
        )
        timestamps = [raster.timestamps[idx] for raster, idx in selected]  # type: ignore[index]
        input_dict[self.output_key] = RasterImage(image, timestamps=timestamps)

        if (
            self.category_target is not None
            and self.change_day_target is not None
            and self.category_target in target_dict
            and self.change_day_target in target_dict
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
