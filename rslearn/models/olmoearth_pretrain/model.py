"""OlmoEarth model wrapper for fine-tuning in rslearn."""

import json
import warnings
from contextlib import nullcontext
from datetime import datetime
from typing import Any

import torch
from einops import rearrange
from olmo_core.config import Config
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.model_loader import (
    ModelID,
    load_model_from_id,
    load_model_from_path,
)
from olmoearth_pretrain.nn.flexihelios import Encoder, TokensAndMasks
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue
from upath import UPath

from rslearn.log_utils import get_logger
from rslearn.models.component import FeatureExtractor, FeatureMaps, TokenFeatureMaps
from rslearn.train.model_context import ModelContext, RasterImage

logger = get_logger(__name__)

MODALITY_NAMES = [
    "sentinel2_l2a",
    "sentinel1",
    "worldcover",
    "openstreetmap_raster",
    "landsat",
]

AUTOCAST_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

EMBEDDING_SIZES = {
    ModelID.OLMOEARTH_V1_NANO: 128,
    ModelID.OLMOEARTH_V1_TINY: 192,
    ModelID.OLMOEARTH_V1_BASE: 768,
    ModelID.OLMOEARTH_V1_LARGE: 1024,
}


class OlmoEarth(FeatureExtractor):
    """A wrapper to support the OlmoEarth model."""

    def __init__(
        self,
        patch_size: int,
        model_id: ModelID | None = None,
        model_path: str | None = None,
        checkpoint_path: str | None = None,
        selector: list[str | int] = ["encoder"],
        forward_kwargs: dict[str, Any] = {},
        random_initialization: bool = False,
        embedding_size: int | None = None,
        autocast_dtype: str | None = "bfloat16",
        token_pooling: bool = True,
        use_legacy_timestamps: bool = True,
    ):
        """Create a new OlmoEarth model.

        Args:
            patch_size: token spatial patch size to use.
            model_id: the model ID to load. One of model_id or model_path or checkpoint_path must be
                set.
            model_path: the path to load the model from. One of model_id or model_path or checkpoint_path must be
                set. Same structure as the HF-hosted `model_id` models: bundle with a config.json and weights.pth.
            checkpoint_path: the checkpoint directory to load from, if model_id or model_path is not
                set. It should contain a distributed checkpoint with a config.json file as well as model_and_optim
                folder.
            selector: an optional sequence of attribute names or list indices to select
                the sub-module that should be applied on the input images. Defaults to
                ["encoder"] to select only the transformer encoder.
            forward_kwargs: additional arguments to pass to forward pass besides the
                 MaskedOlmoEarthSample.
            random_initialization: whether to skip loading the checkpoint so the
                weights are randomly initialized. In this case, the checkpoint is only
                used to define the model architecture.
            embedding_size: optional embedding size to report via
                get_backbone_channels (if model_id is not set).
            autocast_dtype: which dtype to use for autocasting, or set None to disable.
            token_pooling: whether or not to pool the tokens. If True, the output will be BxCxHxW. If False,
                there will be an extra dimension, N, (BxCxHxWxN) representing the temporal and channel
                dimensions.
            use_legacy_timestamps: In our original implementation of OlmoEarth, we applied timestamps starting
                from 0 (instead of the actual timestamps of the input). The option to do this is preserved
                for backwards compatability with finetuned models which were trained against this implementation.
        """
        if use_legacy_timestamps:
            warnings.warn(
                "For new projects, don't use legacy timesteps.",
                FutureWarning,
            )

        if (
            sum(
                [
                    model_id is not None,
                    model_path is not None,
                    checkpoint_path is not None,
                ]
            )
            != 1
        ):
            raise ValueError(
                "exactly one of model_id, model_path, or checkpoint_path must be set"
            )

        super().__init__()
        self.patch_size = patch_size
        self.forward_kwargs = forward_kwargs
        self.embedding_size = embedding_size

        if autocast_dtype is not None:
            self.autocast_dtype = AUTOCAST_DTYPE_MAP[autocast_dtype]
        else:
            self.autocast_dtype = None

        if model_id is not None:
            # Load from Hugging Face.
            model = load_model_from_id(model_id, load_weights=not random_initialization)
            if self.embedding_size is None and model_id in EMBEDDING_SIZES:
                self.embedding_size = EMBEDDING_SIZES[model_id]

        elif model_path is not None:
            # Load from path.
            model = load_model_from_path(
                UPath(model_path), load_weights=not random_initialization
            )

        else:
            # Load the distributed model checkpoint by path through Olmo Core
            model = self._load_model_from_checkpoint(
                UPath(checkpoint_path), random_initialization
            )

        # Select just the portion of the model that we actually want to use.
        for part in selector:
            if isinstance(part, str):
                model = getattr(model, part)
            else:
                model = model[part]
        self.model = model
        self.token_pooling = token_pooling
        self.use_legacy_timestamps = use_legacy_timestamps

    def _load_model_from_checkpoint(
        self, checkpoint_upath: UPath, random_initialization: bool
    ) -> torch.nn.Module:
        """Load the OlmoEarth pre-trained model from a distributed checkpoint folder.

        The folder should contain config.json as well as the model_and_optim folder
        that contains the distributed checkpoint. This is the format produced by
        pre-training runs in olmoearth_pretrain.
        """
        with (checkpoint_upath / "config.json").open() as f:
            config_dict = json.load(f)
            model_config = Config.from_dict(config_dict["model"])

        model = model_config.build()

        # Load the checkpoint (requires olmo_core for distributed checkpoint loading).
        if not random_initialization:
            from olmo_core.distributed.checkpoint import load_model_and_optim_state

            train_module_dir = checkpoint_upath / "model_and_optim"
            load_model_and_optim_state(str(train_module_dir), model)
            logger.info(f"loaded OlmoEarth encoder from {train_module_dir}")

        return model

    @staticmethod
    def time_ranges_to_timestamps(
        time_ranges: list[tuple[datetime, datetime]],
        max_timestamps: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Turn the time ranges stored in a RasterImage to timestamps accepted by OlmoEarth.

        OlmoEarth only uses the month associated with each timestamp, so we take the midpoint
        the time range. For some inputs (e.g. Sentinel 2) we take an image from a specific
        time so that start_time == end_time == mid_time.
        """
        timestamps = torch.zeros((max_timestamps, 3), dtype=torch.int32, device=device)
        mid_ranges = [t[0] + ((t[1] - t[0]) / 2) for t in time_ranges]
        timestamps[: len(time_ranges), 0] = torch.tensor(
            [d.day for d in mid_ranges], dtype=torch.int32
        )
        # months are indexed 0-11
        timestamps[: len(time_ranges), 1] = torch.tensor(
            [d.month - 1 for d in mid_ranges], dtype=torch.int32
        )
        timestamps[: len(time_ranges), 2] = torch.tensor(
            [d.year for d in mid_ranges], dtype=torch.int32
        )
        return timestamps

    @staticmethod
    def _get_sample_expected_timestamps(
        inp: dict[str, torch.Tensor | RasterImage],
        present_modalities: list[str],
    ) -> list[tuple[datetime, datetime]] | None:
        """Get expected_timestamps for a single sample from any of its modalities.

        Args:
            inp: the input dict for this sample.
            present_modalities: list of modality names to check.

        Returns:
            The expected_timestamps if found in any modality, or None.
        """
        for modality in present_modalities:
            if modality not in inp:
                continue
            raster_img = inp[modality]
            if isinstance(raster_img, RasterImage) and raster_img.expected_timestamps:
                return raster_img.expected_timestamps
        return None

    @staticmethod
    def _find_timestamp_position(
        actual_ts: tuple[datetime, datetime],
        expected_timestamps: list[tuple[datetime, datetime]],
    ) -> int | None:
        """Find the position of an actual timestamp in expected timestamps.

        Uses midpoint matching - finds the expected timestamp whose midpoint is closest
        to the actual timestamp's midpoint.

        Args:
            actual_ts: the actual timestamp (start, end) tuple.
            expected_timestamps: list of expected timestamps.

        Returns:
            Index in expected_timestamps, or None if no good match found.
        """
        actual_mid = actual_ts[0] + (actual_ts[1] - actual_ts[0]) / 2

        best_idx = None
        best_distance = None

        for idx, expected_ts in enumerate(expected_timestamps):
            expected_mid = expected_ts[0] + (expected_ts[1] - expected_ts[0]) / 2
            distance = abs((actual_mid - expected_mid).total_seconds())

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_idx = idx

        return best_idx

    def _align_tensor_to_expected_timestamps(
        self,
        tensor: torch.Tensor,
        actual_timestamps: list[tuple[datetime, datetime]] | None,
        expected_timestamps: list[tuple[datetime, datetime]],
        max_timesteps: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, list[bool]]:
        """Align a tensor's timesteps to expected timestamp positions.

        The rest of the tensor (from len(expected_timestamps) to max_timesteps) is filled with zeros.

        Args:
            tensor: input tensor of shape (C, T, H, W).
            actual_timestamps: list of actual timestamps for each timestep in tensor.
            expected_timestamps: list of expected timestamps to align to.
            max_timesteps: final output timesteps (for padding to batch-compatible size).
            device: the torch device.

        Returns:
            Tuple of (aligned_tensor, valid_mask) where:
            - aligned_tensor has shape (C, max_timesteps, H, W)
            - valid_mask is a list of booleans indicating which positions have data
        """
        num_expected = len(expected_timestamps)
        c, t, h, w = tensor.shape

        # Create output tensor filled with zeros (sized to max_timesteps for batching)
        aligned = torch.zeros(
            (c, max_timesteps, h, w), dtype=tensor.dtype, device=device
        )
        valid_mask = [False] * max_timesteps

        if actual_timestamps is None:
            # No timestamps - can't align, just copy data sequentially
            copy_count = min(t, num_expected)
            aligned[:, :copy_count, :, :] = tensor[:, :copy_count, :, :]
            valid_mask[:copy_count] = [True] * copy_count
        else:
            # Sort actual data by timestamp
            sorted_indices = sorted(
                range(len(actual_timestamps)), key=lambda i: actual_timestamps[i][0]
            )

            for orig_idx in sorted_indices:
                actual_ts = actual_timestamps[orig_idx]
                expected_idx = self._find_timestamp_position(
                    actual_ts, expected_timestamps
                )
                if (
                    expected_idx is not None
                    and expected_idx < max_timesteps
                    and not valid_mask[expected_idx]
                ):
                    aligned[:, expected_idx, :, :] = tensor[:, orig_idx, :, :]
                    valid_mask[expected_idx] = True

        return aligned, valid_mask

    def _prepare_modality_inputs(
        self, context: ModelContext
    ) -> tuple[MaskedOlmoEarthSample, list[str], torch.device]:
        """Prepare modality tensors and masks for the OlmoEarth model.

        Uses expected_timestamps for temporal alignment when available, ensuring that:
        1. Missing timestamps are inserted at their correct temporal positions
        2. Completely missing modalities are handled (all-zero tensor with all-MISSING mask)
        3. Data is chronologically ordered (oldest first)

        Each sample is aligned to its own expected_timestamps, then padded to the
        maximum expected_timestamps length across all samples in the batch.

        Args:
            context: the model context with input tensors.

        Returns:
            tuple of (sample, present_modalities, device)
        """
        kwargs = {}
        device = None
        batch_size = len(context.inputs)

        # Determine which modalities are present in any sample
        present_modalities: list[str] = []
        for modality in MODALITY_NAMES:
            for inp in context.inputs:
                if modality in inp:
                    present_modalities.append(modality)
                    # Get device from first available tensor
                    if device is None:
                        raster_img = inp[modality]
                        if isinstance(raster_img, RasterImage):
                            device = raster_img.image.device
                    break

        if device is None:
            raise ValueError("No modality tensors found in context.inputs")

        # Collect per-sample expected_timestamps
        expected_timestamps_per_sample: list[
            list[tuple[datetime, datetime]] | None
        ] = []
        for inp in context.inputs:
            sample_expected_ts = self._get_sample_expected_timestamps(
                inp, present_modalities
            )
            expected_timestamps_per_sample.append(sample_expected_ts)

        # Determine max_timesteps:
        # - If any sample has expected_timestamps, use max of expected_timestamps lengths
        # - Otherwise, fall back to max actual timesteps
        has_any_expected = any(ts is not None for ts in expected_timestamps_per_sample)
        if has_any_expected:
            max_timesteps = max(
                len(ts) if ts is not None else 0
                for ts in expected_timestamps_per_sample
            )
            # Ensure at least 1 timestep
            if max_timesteps == 0:
                max_timesteps = 1
        else:
            # Fallback to original behavior: find max actual timesteps
            max_timesteps = 1
            for modality in present_modalities:
                for inp in context.inputs:
                    if modality in inp:
                        raster_img = inp[modality]
                        if isinstance(raster_img, RasterImage):
                            max_timesteps = max(
                                max_timesteps, raster_img.image.shape[1]
                            )

        # Track timestamps per instance (aka sample) for position encoding
        # Using expected_timestamps when available, otherwise actual timestamps
        timestamps_per_instance: list[list[tuple[datetime, datetime]]] = []
        if has_any_expected:
            # All samples have expected_timestamps (batch-level invariant)
            timestamps_per_instance = expected_timestamps_per_sample  # type: ignore[assignment]
        else:
            # No samples have expected_timestamps - use actual timestamps
            for inp in context.inputs:
                best_ts: list[tuple[datetime, datetime]] = []
                for modality in present_modalities:
                    if modality in inp:
                        raster_img = inp[modality]
                        if (
                            isinstance(raster_img, RasterImage)
                            and raster_img.timestamps
                        ):
                            if len(raster_img.timestamps) > len(best_ts):
                                best_ts = list(raster_img.timestamps)
                timestamps_per_instance.append(best_ts)

        # Compute spatial dimensions once from any non-missing sample
        spatial_h, spatial_w = None, None
        for inp in context.inputs:
            for modality in present_modalities:
                if modality in inp:
                    raster_img = inp[modality]
                    if isinstance(raster_img, RasterImage):
                        spatial_h, spatial_w = (
                            raster_img.image.shape[2],
                            raster_img.image.shape[3],
                        )
                        break
            if spatial_h is not None:
                break

        if spatial_h is None or spatial_w is None:
            raise ValueError("Cannot determine spatial dimensions from any input")

        # Process each modality
        for modality in present_modalities:
            num_band_sets = len(Modality.get(modality).band_sets)
            num_channels = sum(len(bs.bands) for bs in Modality.get(modality).band_sets)

            aligned_tensors = []
            valid_masks_per_sample = []

            for sample_idx, inp in enumerate(context.inputs):
                if modality in inp:
                    raster_img = inp[modality]
                    assert isinstance(raster_img, RasterImage)
                    tensor = raster_img.image
                    actual_timestamps = raster_img.timestamps

                    sample_expected_ts = expected_timestamps_per_sample[sample_idx]
                    if has_any_expected and sample_expected_ts is not None:
                        # Align to this sample's expected timestamps
                        aligned, valid_mask = self._align_tensor_to_expected_timestamps(
                            tensor,
                            actual_timestamps,
                            sample_expected_ts,
                            max_timesteps,
                            device,
                        )
                    else:
                        # No expected timestamps - pad at end (original behavior)
                        c, t, h, w = tensor.shape
                        if t < max_timesteps:
                            pad = torch.zeros(
                                (c, max_timesteps - t, h, w),
                                dtype=tensor.dtype,
                                device=device,
                            )
                            aligned = torch.cat([tensor, pad], dim=1)
                        else:
                            aligned = tensor
                        valid_mask = [True] * t + [False] * (max_timesteps - t)

                    aligned_tensors.append(aligned)
                    valid_masks_per_sample.append(valid_mask)
                else:
                    # Modality completely missing for this sample
                    aligned = torch.zeros(
                        (num_channels, max_timesteps, spatial_h, spatial_w),
                        dtype=torch.float32,
                        device=device,
                    )
                    aligned_tensors.append(aligned)
                    # All timesteps are missing
                    valid_masks_per_sample.append([False] * max_timesteps)

            # Stack tensors and rearrange
            cur = torch.stack(aligned_tensors, dim=0)  # B, C, T, H, W
            cur = rearrange(cur, "b c t h w -> b h w t c")
            kwargs[modality] = cur

            # Create mask based on valid_masks_per_sample
            b, h, w = cur.shape[0], cur.shape[1], cur.shape[2]
            mask = torch.full(
                (b, h, w, max_timesteps, num_band_sets),
                fill_value=MaskValue.MISSING.value,
                dtype=torch.int32,
                device=device,
            )
            for sample_idx, valid_mask in enumerate(valid_masks_per_sample):
                for t_idx, is_valid in enumerate(valid_mask):
                    if is_valid:
                        mask[sample_idx, :, :, t_idx, :] = (
                            MaskValue.ONLINE_ENCODER.value
                        )
            kwargs[f"{modality}_mask"] = mask

        if self.use_legacy_timestamps:
            # Note that only months (0 to 11) are used in OlmoEarth position encoding.
            timestamps = torch.zeros(
                (batch_size, max_timesteps, 3),
                dtype=torch.int32,
                device=device,
            )
            timestamps[:, :, 0] = 1  # day
            timestamps[:, :, 1] = torch.arange(max_timesteps, device=device)[
                None, :
            ]  # month
            timestamps[:, :, 2] = 2024  # year
            kwargs["timestamps"] = timestamps
        else:
            if max([len(t) for t in timestamps_per_instance]) == 0:
                # Timestamps is required.
                raise ValueError("No inputs had timestamps.")
            # Note that only months (0 to 11) are used in OlmoEarth position encoding.
            kwargs["timestamps"] = torch.stack(
                [
                    self.time_ranges_to_timestamps(time_range, max_timesteps, device)
                    for time_range in timestamps_per_instance
                ],
                dim=0,
            )

        return MaskedOlmoEarthSample(**kwargs), present_modalities, device

    def forward(self, context: ModelContext) -> FeatureMaps | TokenFeatureMaps:
        """Compute feature maps from the OlmoEarth backbone.

        Args:
            context: the model context. Input dicts should include keys corresponding
                to the modalities that should be passed to the OlmoEarth model.

        Returns:
            a FeatureMaps consisting of one feature map, at 1/patch_size of the input
                resolution. Embeddings will be pooled across modalities and timesteps.
        """
        sample, present_modalities, device = self._prepare_modality_inputs(context)

        # Decide context based on self.autocast_dtype.
        if self.autocast_dtype is None:
            torch_context = nullcontext()
        else:
            assert device is not None
            torch_context = torch.amp.autocast(
                device_type=device.type, dtype=self.autocast_dtype
            )

        # Check if we can bypass masks (fast_pass=True)
        missing_tokens = False
        for modality in present_modalities:
            modality_mask = getattr(sample, f"{modality}_mask")
            if torch.any(modality_mask == MaskValue.MISSING.value):
                missing_tokens = True
                break

        with torch_context:
            # Currently we assume the provided model always returns a TokensAndMasks object.
            tokens_and_masks: TokensAndMasks
            if isinstance(self.model, Encoder):
                # Encoder has a fast_pass argument to indicate mask is not needed.
                tokens_and_masks = self.model(
                    sample,
                    fast_pass=not missing_tokens,
                    patch_size=self.patch_size,
                    **self.forward_kwargs,
                )["tokens_and_masks"]
            else:
                # Other models like STEncoder do not have this option supported.
                tokens_and_masks = self.model(
                    sample, patch_size=self.patch_size, **self.forward_kwargs
                )["tokens_and_masks"]

        # Apply temporal/modality pooling so we just have one feature per patch.
        features = []
        if self.token_pooling:
            for modality in present_modalities:
                modality_features = getattr(tokens_and_masks, modality)  # BHWTSC
                # If fast_pass is False, we need to mask the missing tokens before pooling.
                if missing_tokens:
                    modality_masks = getattr(
                        tokens_and_masks, f"{modality}_mask"
                    )  # BHWTS
                    modality_masks_bool = (
                        modality_masks != MaskValue.MISSING.value
                    ).unsqueeze(-1)
                    count = modality_masks_bool.sum(dim=[3, 4])
                    # Masked average over band sets and timesteps (BHWTSC -> BHWC).
                    pooled = (modality_features * modality_masks_bool).sum(
                        dim=[3, 4]
                    ) / count.clamp(min=1)
                else:
                    # Pool over band sets and timesteps (BHWTSC -> BHWC).
                    pooled = modality_features.mean(dim=[3, 4])
                # We want BHWC -> BCHW.
                pooled = rearrange(pooled, "b h w c -> b c h w")
                features.append(pooled)
            # Pool over the modalities, so we get one BCHW feature map.
            pooled = torch.stack(features, dim=0).mean(dim=0)
            return FeatureMaps([pooled])
        else:
            for modality in present_modalities:
                modality_features = getattr(tokens_and_masks, modality)
                # Combine band sets and timesteps into last dim (BHWTSC -> BHWCN).
                modality_features = rearrange(
                    modality_features, "b h w t s c -> b c h w (t s)"
                )
                features.append(modality_features)
            pooled = torch.cat(features, dim=-1)
            return TokenFeatureMaps([pooled])

    def get_backbone_channels(self) -> list:
        """Returns the output channels of this model when used as a backbone.

        The output channels is a list of (downsample_factor, depth) that corresponds
        to the feature maps that the backbone returns. For example, an element [2, 32]
        indicates that the corresponding feature map is 1/2 the input resolution and
        has 32 channels.

        Returns:
            the output channels of the backbone as a list of (downsample_factor, depth)
            tuples.
        """
        return [(self.patch_size, self.embedding_size)]
