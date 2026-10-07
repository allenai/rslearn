"""Sample one change alert time series per example and derive its targets."""

import random
from datetime import UTC, datetime, timedelta
from typing import Any

import torch

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.transform import Transform

UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

# Value in the change day raster for pixels without a change date.
CHANGE_DAY_NODATA = 0


def _as_utc(ts: datetime) -> datetime:
    """Treat naive datetimes as UTC."""
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


def change_timestep(
    timestamps: list[tuple[datetime, datetime]], change_time: datetime
) -> int | None:
    """Get the index of the first image in which a change is observable.

    This is the first image whose time range ends after the change time, i.e. the
    image whose time range contains the change time, or if the change time falls
    between two images, the image after it. An image ending exactly at the change time
    (e.g. a weekly period ending at midnight of the change day) only covers the time
    before the change, so it is not selected.

    Args:
        timestamps: the chronological time ranges of the images.
        change_time: the time at which the change first becomes observable.

    Returns:
        the timestep index, or None if the change is not within the time series
        (it is before the first image or after the last image).
    """
    if not timestamps or change_time < _as_utc(timestamps[0][0]):
        return None
    for idx, (_, end) in enumerate(timestamps):
        if _as_utc(end) > change_time:
            return idx
    return None


class ChangeTimeSeriesSampler(Transform):
    """Build one time series from a choice of options and derive change targets.

    Each option is a frequent input (e.g. 7-day mosaics) optionally paired with an
    infrequent input (e.g. 90-day mosaics), typically corresponding to the layers of
    one slot from make_slot_layer_configs. The inputs should be loaded with
    load_all_item_groups and load_all_layers so that each is a RasterImage with one
    timestep per item group.

    The sampler picks one option (randomly, or option_index if set), then takes the
    latest num_frequent frequent images within frequent_lookback_days of the latest
    frequent image, and the latest num_infrequent infrequent images that end before
    the first selected frequent image. These are concatenated chronologically into
    input_dict[output_key]. Options that lack enough images are only picked if no
    option has enough, in which case the time series has fewer timesteps.

    If the change day raster is present (it is a target, so it is not loaded during
    prediction), the sampler also computes the timestep target: at each pixel with a
    change day, the index of the first image in which the change is observable (see
    change_timestep). Pixels whose change is outside the time series are marked
    invalid for both the timestep and the category targets, since the change cannot be
    observed in the input. Pixels without a change day (e.g. negatives) are invalid
    for the timestep target but keep their category target.

    The change day raster should contain the number of days since 1970-01-01 (UTC) at
    which the change first becomes observable, with 0 at pixels without a change.
    """

    def __init__(
        self,
        options: list[dict[str, str | None]],
        num_frequent: int,
        frequent_lookback_days: int | None = None,
        num_infrequent: int = 0,
        change_day_key: str = "change_day",
        output_key: str = "image",
        category_target: str | None = "category",
        timestep_target: str = "timestep",
        option_index: int | None = None,
        drop_keys: list[str] = [],
    ) -> None:
        """Create a new ChangeTimeSeriesSampler.

        Args:
            options: list of options, each a dict with a "frequent" input key and an
                optional "infrequent" input key.
            num_frequent: the number of frequent images to use.
            frequent_lookback_days: only use frequent images within this many days of
                the end of the latest frequent image. If None, all frequent images are
                candidates.
            num_infrequent: the number of infrequent images to use.
            change_day_key: the input key of the change day raster.
            output_key: the input key to write the time series to.
            category_target: the target whose valid mask should be cleared at pixels
                whose change is outside the time series. If None, no category target
                is updated.
            timestep_target: the target to write the timestep target to.
            option_index: if set, always use this option (e.g. for evaluation or
                prediction) instead of a random one.
            drop_keys: other input keys to remove, e.g. inputs that are only used by
                the sampler of a different split.
        """
        super().__init__()
        if not options:
            raise ValueError("at least one option is required")
        for option in options:
            if "frequent" not in option or option["frequent"] is None:
                raise ValueError(f"option {option} must have a frequent input key")
            if num_infrequent > 0 and option.get("infrequent") is None:
                raise ValueError(
                    f"option {option} needs an infrequent input key since num_infrequent > 0"
                )
        if option_index is not None and not 0 <= option_index < len(options):
            raise ValueError(f"option_index {option_index} is out of range")
        self.options = options
        self.num_frequent = num_frequent
        self.frequent_lookback = (
            timedelta(days=frequent_lookback_days)
            if frequent_lookback_days is not None
            else None
        )
        self.num_infrequent = num_infrequent
        self.change_day_key = change_day_key
        self.output_key = output_key
        self.category_target = category_target
        self.timestep_target = timestep_target
        self.option_index = option_index
        self.drop_keys = drop_keys

    def _select(
        self, option: dict[str, str | None], input_dict: dict[str, Any]
    ) -> tuple[list[tuple[RasterImage, int]], bool]:
        """Select the images to use from one option.

        Returns:
            a tuple (selected, complete) where selected is the chronological list of
            (image, timestep index) to use and complete is whether there were enough
            images.
        """
        frequent_key = option["frequent"]
        assert frequent_key is not None
        frequent = input_dict.get(frequent_key)
        if frequent is None or not frequent.timestamps:
            return [], False

        frequent_idxs = list(range(len(frequent.timestamps)))
        if self.frequent_lookback is not None:
            latest_end = max(_as_utc(end) for _, end in frequent.timestamps)
            min_start = latest_end - self.frequent_lookback
            frequent_idxs = [
                idx
                for idx in frequent_idxs
                if _as_utc(frequent.timestamps[idx][0]) >= min_start
            ]
        frequent_idxs = frequent_idxs[-self.num_frequent :]
        complete = len(frequent_idxs) == self.num_frequent
        selected = [(frequent, idx) for idx in frequent_idxs]
        if not selected:
            return [], False

        if self.num_infrequent > 0:
            infrequent_key = option.get("infrequent")
            assert infrequent_key is not None
            infrequent = input_dict.get(infrequent_key)
            infrequent_idxs: list[int] = []
            if infrequent is not None and infrequent.timestamps:
                # Strict inequality so that a scene shared with the first frequent
                # image is not duplicated, since OlmoEarth requires distinct timesteps.
                first_start = _as_utc(frequent.timestamps[frequent_idxs[0]][0])
                infrequent_idxs = [
                    idx
                    for idx, (_, end) in enumerate(infrequent.timestamps)
                    if _as_utc(end) < first_start
                ][-self.num_infrequent :]
            complete = complete and len(infrequent_idxs) == self.num_infrequent
            selected = [(infrequent, idx) for idx in infrequent_idxs] + selected

        return selected, complete

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build the time series and targets.

        Args:
            input_dict: the input dict, containing the option inputs and optionally
                the change day raster.
            target_dict: the target dict.

        Returns:
            the updated (input_dict, target_dict).
        """
        if self.option_index is not None:
            selected, _ = self._select(self.options[self.option_index], input_dict)
        else:
            candidates = [self._select(option, input_dict) for option in self.options]
            complete = [sel for sel, is_complete in candidates if is_complete]
            if complete:
                selected = random.choice(complete)
            else:
                # Fall back to the option with the most images.
                selected = max((sel for sel, _ in candidates), key=len)
        if not selected:
            raise ValueError(f"no images available for any option among {self.options}")

        image = torch.cat(
            [raster.image[:, idx : idx + 1] for raster, idx in selected], dim=1
        )
        timestamps = [raster.timestamps[idx] for raster, idx in selected]  # type: ignore[index]

        for option in self.options:
            for key in (option.get("frequent"), option.get("infrequent")):
                if key is not None:
                    input_dict.pop(key, None)
        for key in self.drop_keys:
            input_dict.pop(key, None)
        input_dict[self.output_key] = RasterImage(image, timestamps=timestamps)

        change_day = input_dict.pop(self.change_day_key, None)
        if change_day is not None:
            self._make_targets(change_day, timestamps, target_dict)

        return input_dict, target_dict

    def _make_targets(
        self,
        change_day: RasterImage,
        timestamps: list[tuple[datetime, datetime]],
        target_dict: dict[str, Any],
    ) -> None:
        """Compute the timestep target and update the category valid mask."""
        days = change_day.get_hw_tensor().long()
        classes = torch.zeros(days.shape, dtype=torch.long)
        valid = torch.zeros(days.shape, dtype=torch.float32)
        outside = torch.zeros(days.shape, dtype=torch.bool)

        for day in torch.unique(days).tolist():
            if day == CHANGE_DAY_NODATA:
                continue
            mask = days == day
            idx = change_timestep(timestamps, UNIX_EPOCH + timedelta(days=day))
            if idx is None:
                outside |= mask
            else:
                classes[mask] = idx
                valid[mask] = 1

        target_dict[self.timestep_target] = {
            "classes": RasterImage(classes[None, None, :, :]),
            "valid": RasterImage(valid[None, None, :, :]),
        }

        if self.category_target is not None and outside.any():
            category_valid = target_dict[self.category_target]["valid"]
            new_valid = category_valid.image.clone()
            new_valid[:, :, outside] = 0
            target_dict[self.category_target]["valid"] = RasterImage(
                new_valid, timestamps=category_valid.timestamps
            )
