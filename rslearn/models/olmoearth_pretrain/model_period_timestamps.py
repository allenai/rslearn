"""OlmoEarthPeriodTimestamps: aligns modality images to a fixed period grid."""

import logging
from datetime import datetime, timedelta
from typing import Any

import torch
from einops import rearrange
from olmoearth_pretrain_minimal import ModelID
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
)

from rslearn.models.olmoearth_pretrain.model import MODALITY_NAMES, OlmoEarth
from rslearn.train.model_context import ModelContext, RasterImage

logger = logging.getLogger(__name__)


class OlmoEarthPeriodTimestamps(OlmoEarth):
    """OlmoEarth model that aligns images to periods derived from the window time range.

    This subclass assumes that all modalities are configured with the same
    period_duration and max_matches, with no time_offset/duration modifying the window
    time range, and no fallback periods. Then, we align modalities by computing the
    periods for each sample, and aligning images against those periods.
    """

    def __init__(
        self,
        patch_size: int,
        period_duration: timedelta,
        max_matches: int,
        model_id: ModelID | None = None,
        model_path: str | None = None,
        checkpoint_path: str | None = None,
        selector: list[str | int] = ["encoder"],
        forward_kwargs: dict[str, Any] = {},
        random_initialization: bool = False,
        embedding_size: int | None = None,
        autocast_dtype: str | None = "bfloat16",
        token_pooling: bool = True,
        token_instance_pooling: bool = False,
    ):
        """Create a new OlmoEarthPeriodTimestamps model.

        Args:
            patch_size: token spatial patch size to use.
            period_duration: length of each period (e.g. timedelta(days=30)).
            max_matches: total number of periods to create from the window time range.
            model_id: the model ID to load.
            model_path: the path to load the model from.
            checkpoint_path: the checkpoint directory to load from.
            selector: sub-module selector path.
            forward_kwargs: additional forward-pass arguments.
            random_initialization: skip loading weights.
            embedding_size: optional embedding size override.
            autocast_dtype: dtype for autocasting, or None to disable.
            token_pooling: whether to pool tokens (BxCxHxW) or keep them (BxCxHxWxN).
            token_instance_pooling: whether to pool all valid tokens into a BxC
                FeatureVector instead of returning spatial feature maps.
        """
        super().__init__(
            patch_size=patch_size,
            model_id=model_id,
            model_path=model_path,
            checkpoint_path=checkpoint_path,
            selector=selector,
            forward_kwargs=forward_kwargs,
            random_initialization=random_initialization,
            embedding_size=embedding_size,
            autocast_dtype=autocast_dtype,
            token_pooling=token_pooling,
            token_instance_pooling=token_instance_pooling,
            use_legacy_timestamps=False,
        )
        self.period_duration = period_duration
        self.max_matches = max_matches

    @staticmethod
    def _compute_periods(
        time_range: tuple[datetime, datetime],
        period_duration: timedelta,
        max_matches: int,
    ) -> list[tuple[datetime, datetime]]:
        """Compute period timestamps from a time range.

        Periods are created from the end of the time range backwards, matching the
        behavior in rslearn.data_sources.utils.match_candidate_items_to_window.
        Up to max_matches periods are created. The returned list is in chronological
        order.

        Args:
            time_range: (start, end) of the window.
            period_duration: length of each period.
            max_matches: total number of periods to create.

        Returns:
            List of (period_start, period_end) tuples in chronological order.
        """
        periods: list[tuple[datetime, datetime]] = []
        period_end = time_range[1]
        while (
            period_end - period_duration >= time_range[0] and len(periods) < max_matches
        ):
            period_start = period_end - period_duration
            periods.append((period_start, period_end))
            period_end = period_start
        periods.reverse()
        return periods

    @staticmethod
    def _find_period_position(
        actual_ts: tuple[datetime, datetime],
        periods: list[tuple[datetime, datetime]],
        excluded: set[int] | None = None,
    ) -> int | None:
        """Find the closest period for an actual image timestamp using midpoint matching.

        Args:
            actual_ts: the actual (start, end) timestamp of the image.
            periods: list of period (start, end) tuples.
            excluded: set of period indices to skip (already filled).

        Returns:
            Index into periods of the best match, or None if no match.
        """
        actual_mid = actual_ts[0] + (actual_ts[1] - actual_ts[0]) / 2
        if excluded is None:
            excluded = set()

        best_idx = None
        best_distance = None

        for idx, period in enumerate(periods):
            if idx in excluded:
                continue
            period_mid = period[0] + (period[1] - period[0]) / 2
            distance = abs((actual_mid - period_mid).total_seconds())

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_idx = idx

        return best_idx

    def _align_tensor_to_periods(
        self,
        tensor: torch.Tensor,
        actual_timestamps: list[tuple[datetime, datetime]] | None,
        periods: list[tuple[datetime, datetime]],
        max_timesteps: int,
        num_band_sets: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align a CTHW tensor to the computed period slots.

        Args:
            tensor: input tensor of shape (C, T, H, W).
            actual_timestamps: timestamps for each timestep in tensor, or None.
            periods: the period slots to align to.
            max_timesteps: total output timesteps (for padding across the batch).
            num_band_sets: number of band sets for this modality.
            device: torch device.

        Returns:
            (aligned_tensor, valid_mask) where aligned_tensor has shape
            (C, max_timesteps, H, W) and valid_mask has shape
            (max_timesteps, H, W, num_band_sets).
        """
        num_periods = len(periods)
        c, t, h, w = tensor.shape

        aligned = torch.zeros(
            (c, max_timesteps, h, w), dtype=tensor.dtype, device=device
        )
        mask = torch.full(
            (max_timesteps, h, w, num_band_sets),
            fill_value=MaskValue.MISSING.value,
            dtype=torch.int32,
            device=device,
        )

        if actual_timestamps is None:
            copy_count = min(t, num_periods)
            aligned[:, :copy_count, :, :] = tensor[:, :copy_count, :, :]
            mask[:copy_count, :, :, :] = MaskValue.ONLINE_ENCODER.value
        else:
            sorted_indices = sorted(
                range(len(actual_timestamps)), key=lambda i: actual_timestamps[i][0]
            )

            filled: set[int] = set()
            for orig_idx in sorted_indices:
                actual_ts = actual_timestamps[orig_idx]
                period_idx = self._find_period_position(
                    actual_ts, periods, excluded=filled
                )
                if period_idx is not None and period_idx < max_timesteps:
                    aligned[:, period_idx, :, :] = tensor[:, orig_idx, :, :]
                    mask[period_idx, :, :, :] = MaskValue.ONLINE_ENCODER.value
                    filled.add(period_idx)
                else:
                    logger.warning(
                        "Image %d (timestamp %s) could not be assigned to a period.",
                        orig_idx,
                        actual_ts,
                    )

        return aligned, mask

    def _prepare_modality_inputs(
        self, context: ModelContext
    ) -> tuple[MaskedOlmoEarthSample, list[str], torch.device]:
        """Prepare modality tensors aligned to period timestamps.

        Computes periods from each sample's SampleMetadata.time_range, then aligns
        each modality's images to those periods via midpoint matching.

        Args:
            context: the model context with input tensors.

        Returns:
            tuple of (sample, present_modalities, device)
        """
        device = None

        present_modalities: list[str] = []
        for modality in MODALITY_NAMES:
            for inp in context.inputs:
                if modality not in inp:
                    continue
                assert isinstance(inp[modality], RasterImage)
                present_modalities.append(modality)
                if device is None:
                    device = inp[modality].image.device
                break

        if device is None:
            raise ValueError("No modalities present in context.inputs")

        # Compute periods per sample from SampleMetadata.time_range.
        periods_per_sample: list[list[tuple[datetime, datetime]]] = []
        for metadata in context.metadatas:
            if metadata.time_range is None:
                raise ValueError(
                    "OlmoEarthPeriodTimestamps requires SampleMetadata.time_range "
                    "to be set for all samples."
                )
            periods = self._compute_periods(
                metadata.time_range, self.period_duration, self.max_matches
            )
            if len(periods) < self.max_matches:
                logger.warning(
                    "Window %s/%s: time range fits %d periods but "
                    "max_matches=%d. The window time range is shorter than "
                    "max_matches * period_duration.",
                    metadata.window_group,
                    metadata.window_name,
                    len(periods),
                    self.max_matches,
                )
            periods_per_sample.append(periods)

        max_timesteps = max(len(p) for p in periods_per_sample)

        crop_bounds = context.metadatas[0].crop_bounds
        width = crop_bounds[2] - crop_bounds[0]
        height = crop_bounds[3] - crop_bounds[1]

        kwargs: dict[str, torch.Tensor] = {}
        for modality in present_modalities:
            num_channels = sum(len(bs.bands) for bs in Modality.get(modality).band_sets)
            num_band_sets = len(Modality.get(modality).band_sets)

            aligned_tensors = []
            valid_masks = []

            for sample_idx, input_dict in enumerate(context.inputs):
                sample_periods = periods_per_sample[sample_idx]

                if modality in input_dict:
                    raster_image = input_dict[modality]
                    aligned, mask = self._align_tensor_to_periods(
                        raster_image.image,
                        raster_image.timestamps,
                        sample_periods,
                        max_timesteps,
                        num_band_sets,
                        device,
                    )
                    aligned_tensors.append(aligned)
                    valid_masks.append(mask)
                else:
                    aligned_tensors.append(
                        torch.zeros(
                            (num_channels, max_timesteps, height, width),
                            dtype=torch.float32,
                            device=device,
                        )
                    )
                    valid_masks.append(
                        torch.full(
                            (max_timesteps, height, width, num_band_sets),
                            fill_value=MaskValue.MISSING.value,
                            dtype=torch.int32,
                            device=device,
                        )
                    )

            cur = torch.stack(aligned_tensors, dim=0)  # B, C, T, H, W
            cur = rearrange(cur, "b c t h w -> b h w t c")
            kwargs[modality] = cur

            mask = torch.stack(valid_masks, dim=0)  # B, T, H, W, S
            mask = rearrange(mask, "b t h w s -> b h w t s")
            kwargs[f"{modality}_mask"] = mask

        # Encode period midpoints as timestamps for position encoding.
        period_midpoints_per_sample = [
            [p[0] + (p[1] - p[0]) / 2 for p in periods]
            for periods in periods_per_sample
        ]
        kwargs["timestamps"] = torch.stack(
            [
                self.datetimes_to_timestamps(midpoints, max_timesteps, device)
                for midpoints in period_midpoints_per_sample
            ],
            dim=0,
        )

        return MaskedOlmoEarthSample(**kwargs), present_modalities, device
