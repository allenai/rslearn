"""OlmoEarth model wrapper for fine-tuning in rslearn."""

import copy
import json
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import torch
from einops import rearrange
from olmoearth_pretrain_minimal import ModelID, load_model_from_id, load_model_from_path
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.config import Config
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
)
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

TOKENS_IN_BATCH_KEY = "tokens_in_batch"

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
        timestamp_error_tolerance: timedelta = timedelta(days=15),
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
            use_legacy_timestamps: set timestamps to dummy values [1 January 2024, 1 February 2024, ...]
                instead of the actual timestamps of the input. The option to do this is preserved
                for backwards compatability with finetuned models which were trained against this
                original implementation. If set False, then we determine the timesteps across modalities
                based on the actual image timestamps with a greedy algorithm.
            timestamp_error_tolerance: when use_legacy_timestamps=False, if there are multiple modalities,
                we try to align timesteps across modalities. To do so, we use a greedy algorithm, where
                if the closest previous timestamp to a new image is within timestamp_error_tolerance of
                the new image's capture time, then we reuse that timestamp; otherwise, we create a new
                timestep.
        """
        if use_legacy_timestamps:
            warnings.warn(
                "For new projects, don't use legacy timesteps. "
                "Support will be removed after 2026-04-01.",
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
        self.timestamp_error_tolerance = timestamp_error_tolerance

    def _patch_legacy_encoder_config(self, config_dict: dict) -> dict:
        """Patch checkpoint config dicts that predate use_linear_patch_embed.

        Old checkpoints used Conv2d for patch projection and have no use_linear_patch_embed
        key. Without this patch they would incorrectly default to True (Linear) and fail
        to load. Call this on the raw config dict before passing to Config.from_dict.
        """
        enc = config_dict.get("model", {}).get("encoder_config", {})
        if isinstance(enc, dict) and "use_linear_patch_embed" not in enc:
            config_dict = copy.deepcopy(config_dict)
            config_dict["model"]["encoder_config"]["use_linear_patch_embed"] = False
        return config_dict

    def _load_model_from_checkpoint(
        self, checkpoint_upath: UPath, random_initialization: bool
    ) -> torch.nn.Module:
        """Load the OlmoEarth pre-trained model from a distributed checkpoint folder.

        The folder should contain config.json as well as the model_and_optim folder
        that contains the distributed checkpoint. This is the format produced by
        pre-training runs in olmoearth_pretrain.

        Uses the full olmoearth_pretrain package if available (to pick up architecture
        updates), otherwise falls back to olmoearth_pretrain_minimal.
        """
        try:
            from olmoearth_pretrain.config import Config as FullConfig
        except ImportError:
            FullConfig = Config

        with (checkpoint_upath / "config.json").open() as f:
            config_dict = json.load(f)
            config_dict = self._patch_legacy_encoder_config(config_dict)
            model_config = FullConfig.from_dict(config_dict["model"])

        model = model_config.build()

        if not random_initialization:
            try:
                from olmo_core.distributed.checkpoint import (
                    load_model_and_optim_state,
                )
            except ImportError:
                raise ImportError(
                    "olmo-core is required for loading distributed checkpoints. "
                    "Install it with: pip install olmo-core"
                )

            train_module_dir = checkpoint_upath / "model_and_optim"
            load_model_and_optim_state(str(train_module_dir), model)
            logger.info(f"loaded OlmoEarth encoder from {train_module_dir}")

        return model

    @staticmethod
    def datetimes_to_timestamps(
        datetimes: list[datetime],
        max_timestamps: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Turn datetimes to timestamps accepted by OlmoEarth."""
        timestamps = torch.zeros((max_timestamps, 3), dtype=torch.int32, device=device)
        timestamps[: len(datetimes), 0] = torch.tensor(
            [d.day for d in datetimes], dtype=torch.int32, device=device
        )
        # months are indexed 0-11
        timestamps[: len(datetimes), 1] = torch.tensor(
            [d.month - 1 for d in datetimes], dtype=torch.int32, device=device
        )
        timestamps[: len(datetimes), 2] = torch.tensor(
            [d.year for d in datetimes], dtype=torch.int32, device=device
        )
        return timestamps

    def _prepare_modality_inputs_legacy(
        self, context: ModelContext
    ) -> tuple[MaskedOlmoEarthSample, list[str], torch.device]:
        """Legacy modality tensor and mask preparation.

        We compute the maximum number of timesteps across samples and modalities, and
        pad all modalities to those timesteps. The time ranges are set as
        [1 January 2024, 1 February 2024, ...].

        Args:
            context: the model context with input tensors.

        Returns:
            tuple of (sample, present_modalities, device)
        """
        device = None
        batch_size = len(context.inputs)

        # Determine which modalities are present in any sample
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

        # Compute the max_timesteps across samples and modalities.
        max_timesteps = 1
        for input_dict in context.inputs:
            for modality in present_modalities:
                if modality not in input_dict:
                    continue

                raster_img = input_dict[modality]
                assert isinstance(raster_img, RasterImage)
                max_timesteps = max(max_timesteps, raster_img.image.shape[1])

        # Determine height/width from the first available input tensor rather than
        # crop_bounds, since transforms like Pad may have changed the tensor's spatial
        # dimensions without updating the metadata.
        height: int | None = None
        width: int | None = None
        for input_dict in context.inputs:
            for modality in present_modalities:
                if modality in input_dict:
                    height = input_dict[modality].image.shape[-2]
                    width = input_dict[modality].image.shape[-1]
                    break
            if height is not None:
                break

        if height is None or width is None:
            raise ValueError(
                "Could not determine spatial dimensions from any input tensor"
            )

        # Process each modality.
        kwargs = {}
        for modality in present_modalities:
            num_channels = sum(len(bs.bands) for bs in Modality.get(modality).band_sets)
            num_band_sets = len(Modality.get(modality).band_sets)

            padded_tensors = []
            valid_masks = []

            for input_dict in context.inputs:
                if modality in input_dict:
                    raster_img = input_dict[modality]
                    assert isinstance(raster_img, RasterImage)
                    tensor = raster_img.image

                    # We pad the T dimension of CTHW tensor to max_timesteps.
                    cur_timesteps = tensor.shape[1]
                    padded = torch.zeros(
                        (num_channels, max_timesteps, height, width),
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                    padded[:, :cur_timesteps, :, :] = tensor
                    padded_tensors.append(padded)

                    valid_mask = [MaskValue.ONLINE_ENCODER.value] * cur_timesteps + [
                        MaskValue.MISSING.value
                    ] * (max_timesteps - cur_timesteps)
                    valid_mask_tensor = torch.tensor(
                        valid_mask, dtype=torch.int32, device=tensor.device
                    )
                    valid_mask_tensor = valid_mask_tensor.view(max_timesteps, 1, 1, 1)
                    valid_mask_tensor = valid_mask_tensor.expand(
                        max_timesteps, height, width, num_band_sets
                    )
                    valid_masks.append(valid_mask_tensor)
                else:
                    # Modality completely missing for this sample
                    padded = torch.zeros(
                        (num_channels, max_timesteps, height, width),
                        dtype=torch.float32,
                        device=device,
                    )
                    padded_tensors.append(padded)
                    valid_masks.append(
                        torch.full(
                            (max_timesteps, height, width, num_band_sets),
                            fill_value=MaskValue.MISSING.value,
                            dtype=torch.int32,
                            device=device,
                        )
                    )

            # Stack tensors and rearrange
            cur = torch.stack(padded_tensors, dim=0)  # B, C, T, H, W
            cur = rearrange(cur, "b c t h w -> b h w t c")
            kwargs[modality] = cur

            mask = torch.stack(valid_masks, dim=0)  # B, T, H, W, S
            mask = rearrange(mask, "b t h w s -> b h w t s")
            kwargs[f"{modality}_mask"] = mask

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

        return MaskedOlmoEarthSample(**kwargs), present_modalities, device

    def _prepare_modality_inputs(
        self, context: ModelContext
    ) -> tuple[MaskedOlmoEarthSample, list[str], torch.device]:
        """Prepare modality tensors and masks for the OlmoEarth model.

        We compute timesteps for each sample greedily: for each image input, if the
        closest existing timestep (that is not already used by another image in the same
        modality) is within timestamp_error_tolerance of the image's timestamp, then we
        slot the image there. Otherwise, we add another timestep based on the image's
        timestamp. We sort the computed timesteps chronologically.

        Args:
            context: the model context with input tensors.

        Returns:
            tuple of (sample, present_modalities, device)
        """
        # Determine which modalities are present in any sample
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

        # For each sample, greedily assign images to timesteps.
        # We iterate over images across modalities while maintaining a timestamps list.
        # If the image is within timestamp_error_tolerance of an existing timestamp in
        # the list, and no other image in the same modality has already been assigned
        # to that timestamp, then we assign the image to that timestamp. Otherwise, we
        # add a new timestamp to the list corresponding to the image's capture time.

        @dataclass
        class SampleTimestamps:
            # List of timestamps greedily chosen for this sample.
            timestamps: list[datetime]
            # For each modality, the timestamp assigned to each image present in that modality.
            modality_timestamps: dict[str, list[datetime]]

        all_sample_timestamps: list[SampleTimestamps] = []

        for input_dict in context.inputs:
            sample_timestamps = SampleTimestamps(
                timestamps=[],
                modality_timestamps={},
            )
            for modality in MODALITY_NAMES:
                if modality not in input_dict:
                    continue

                raster_image = input_dict[modality]
                if raster_image.timestamps is None or any(
                    ts is None for ts in raster_image.timestamps
                ):
                    raise ValueError(
                        f"modality {modality} has no timestamps or has some images without a timestamp. "
                        "Enable use_legacy_timestamps if timestamps are unavailable."
                    )

                cur_modality_timestamps = []
                for cur_timestamp in raster_image.timestamps:
                    center_time = (
                        cur_timestamp[0] + (cur_timestamp[1] - cur_timestamp[0]) / 2
                    )
                    # Check for the best existing timestamp that we can assign this image to.
                    best_timestamp: datetime | None = None
                    best_timedelta = self.timestamp_error_tolerance
                    for existing_timestamp in sample_timestamps.timestamps:
                        if existing_timestamp in cur_modality_timestamps:
                            # Already assigned to another image in this modality.
                            continue
                        diff = abs(center_time - existing_timestamp)
                        if (
                            diff >= self.timestamp_error_tolerance
                            or diff >= best_timedelta
                        ):
                            continue
                        best_timestamp = existing_timestamp
                        best_timedelta = diff

                    if best_timestamp is not None:
                        cur_modality_timestamps.append(best_timestamp)
                        continue

                    # No suitable existing timestamp, so add a new one.
                    if center_time in sample_timestamps.timestamps:
                        # This implies another image in the same modality has the same timestamp,
                        # otherwise we should have matched to it.
                        raise ValueError(
                            f"modality {modality} has multiple images with the same timestamp"
                        )
                    sample_timestamps.timestamps.append(center_time)
                    cur_modality_timestamps.append(center_time)

                sample_timestamps.modality_timestamps[modality] = (
                    cur_modality_timestamps
                )

            # Sort the timestamps chronologically.
            # Note that this means the user cannot pass in images in a different
            # temporal order. But we need to sort since we are adding timestamps from
            # different modalities interleaved.
            sample_timestamps.timestamps.sort()

            all_sample_timestamps.append(sample_timestamps)

        max_timesteps = max(
            len(sample_timestamps.timestamps)
            for sample_timestamps in all_sample_timestamps
        )

        # Determine height/width from the first available input tensor rather than
        # crop_bounds, since transforms like Pad may have changed the tensor's spatial
        # dimensions without updating the metadata.
        height: int | None = None
        width: int | None = None
        for input_dict in context.inputs:
            for modality in present_modalities:
                if modality in input_dict:
                    height = input_dict[modality].image.shape[-2]
                    width = input_dict[modality].image.shape[-1]
                    break
            if height is not None:
                break

        if height is None or width is None:
            raise ValueError(
                "Could not determine spatial dimensions from any input tensor"
            )

        # Process each modality
        kwargs = {}
        for modality in present_modalities:
            num_channels = sum(len(bs.bands) for bs in Modality.get(modality).band_sets)
            num_band_sets = len(Modality.get(modality).band_sets)

            aligned_tensors = []
            valid_masks = []

            for input_dict, sample_timestamps in zip(
                context.inputs, all_sample_timestamps
            ):
                if modality not in input_dict:
                    # Modality completely missing for this sample
                    aligned = torch.zeros(
                        (num_channels, max_timesteps, height, width),
                        dtype=torch.float32,
                        device=device,
                    )
                    aligned_tensors.append(aligned)
                    valid_masks.append(
                        torch.full(
                            (max_timesteps, height, width, num_band_sets),
                            fill_value=MaskValue.MISSING.value,
                            dtype=torch.int32,
                            device=device,
                        )
                    )
                    continue

                raster_image = input_dict[modality]

                # Get dict from timestamp index -> index in raster_image.
                timestamp_index_to_raster_index = {}
                for raster_index, timestamp in enumerate(
                    sample_timestamps.modality_timestamps.get(modality, [])
                ):
                    timestamp_index = sample_timestamps.timestamps.index(timestamp)
                    timestamp_index_to_raster_index[timestamp_index] = raster_index

                # Now get raster tensors or missing tensors based on the index dict.
                aligned = []
                valid = []
                for timestamp_index in range(max_timesteps):
                    if timestamp_index not in timestamp_index_to_raster_index:
                        aligned.append(
                            torch.zeros(
                                (num_channels, height, width),
                                dtype=torch.float32,
                                device=device,
                            )
                        )
                        valid.append(
                            torch.full(
                                (height, width, num_band_sets),
                                fill_value=MaskValue.MISSING.value,
                                dtype=torch.int32,
                                device=device,
                            )
                        )
                        continue

                    raster_index = timestamp_index_to_raster_index[timestamp_index]
                    # Select timestep from CTHW tensor.
                    aligned.append(raster_image.image[:, raster_index, :, :])
                    valid.append(
                        torch.full(
                            (height, width, num_band_sets),
                            fill_value=MaskValue.ONLINE_ENCODER.value,
                            dtype=torch.int32,
                            device=device,
                        )
                    )

                aligned_tensors.append(torch.stack(aligned, dim=1))
                valid_masks.append(torch.stack(valid, dim=0))

            # Stack tensors and rearrange
            cur = torch.stack(aligned_tensors, dim=0)  # B, C, T, H, W
            cur = rearrange(cur, "b c t h w -> b h w t c")
            kwargs[modality] = cur

            mask = torch.stack(valid_masks, dim=0)  # B, T, H, W, S
            mask = rearrange(mask, "b t h w s -> b h w t s")
            kwargs[f"{modality}_mask"] = mask

        # Note that only months (0 to 11) are used in OlmoEarth position encoding.
        kwargs["timestamps"] = torch.stack(
            [
                self.datetimes_to_timestamps(
                    sample_timestamps.timestamps, max_timesteps, device
                )
                for sample_timestamps in all_sample_timestamps
            ],
            dim=0,
        )

        return MaskedOlmoEarthSample(**kwargs), present_modalities, device

    @staticmethod
    def compute_tokens_in_batch(
        tokens_and_masks: TokensAndMasks, present_modalities: list[str]
    ) -> int:
        """Count the total tokens in the batch from the encoder output shapes.

        Args:
            tokens_and_masks: encoder output with BHWTSC tensors per modality.
            present_modalities: modality names that were fed to the encoder.

        Returns:
            total token count (B * H * W * T * S summed across modalities).
        """
        total = 0
        for modality in present_modalities:
            b, h, w, t, s, _ = getattr(tokens_and_masks, modality).shape
            total += b * h * w * t * s
        return total

    def forward(self, context: ModelContext) -> FeatureMaps | TokenFeatureMaps:
        """Compute feature maps from the OlmoEarth backbone.

        Args:
            context: the model context. Input dicts should include keys corresponding
                to the modalities that should be passed to the OlmoEarth model.

        Returns:
            a FeatureMaps consisting of one feature map, at 1/patch_size of the input
                resolution. Embeddings will be pooled across modalities and timesteps.
        """
        if self.use_legacy_timestamps:
            sample, present_modalities, device = self._prepare_modality_inputs_legacy(
                context
            )
        else:
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
            tokens_and_masks = self.model(
                sample,
                fast_pass=not missing_tokens,
                patch_size=self.patch_size,
                **self.forward_kwargs,
            )["tokens_and_masks"]

        context.context_dict[TOKENS_IN_BATCH_KEY] = self.compute_tokens_in_batch(
            tokens_and_masks, present_modalities
        )

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
