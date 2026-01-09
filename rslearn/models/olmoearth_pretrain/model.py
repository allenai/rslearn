"""OlmoEarth model wrapper for fine-tuning in rslearn."""

import json
import warnings
from contextlib import nullcontext
from datetime import datetime
from typing import Any

import torch
from einops import rearrange
from olmoearth_pretrain.config import Config, require_olmo_core
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import (
    ModelID,
    load_model_from_id,
    load_model_from_path,
)
from olmoearth_pretrain.nn.flexihelios import Encoder, TokensAndMasks
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
                "For new projects, don't use legacy timesteps.", DeprecationWarning
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
        # We avoid loading the train module here because it depends on running within
        # olmo_core.
        # Only pull in olmo_core when trying to load a distributed checkpoint to avoid dependency.
        require_olmo_core("_load_model_from_checkpoint")
        from olmo_core.distributed.checkpoint import load_model_and_optim_state

        with (checkpoint_upath / "config.json").open() as f:
            config_dict = json.load(f)
            model_config = Config.from_dict(config_dict["model"])

        model = model_config.build()

        # Load the checkpoint.
        if not random_initialization:
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

    def _prepare_modality_inputs(
        self, context: ModelContext
    ) -> tuple[MaskedOlmoEarthSample, list[str], torch.device]:
        """Prepare modality tensors and masks for the OlmoEarth model.

        Uses a two-pass approach to ensure all modalities have consistent timestep
        dimensions for position encoding.

        Args:
            context: the model context with input tensors.

        Returns:
            tuple of (sample, present_modalities, device)
        """
        kwargs = {}
        present_modalities = []
        device = None

        # First pass: find global max_timesteps across all modalities and samples
        # TODO: currently we assume all modalities have the same number of timesteps,
        # which is not true for all cases, and time series time steps are assumed to
        # be 1-month apart. It also assumes continuity between available timesteps.
        # We'll have to fix all that.
        max_timesteps = 1
        modality_data = {}
        # we will just store the longest time range
        # per instance in the batch. This means it may not be
        # aligned per modality
        timestamps_per_instance: list[list[tuple[datetime, datetime]]] = [[]] * len(
            context.inputs
        )
        for modality in MODALITY_NAMES:
            if modality not in context.inputs[0]:
                continue
            present_modalities.append(modality)
            tensors = []
            for idx, inp in enumerate(context.inputs):
                assert isinstance(inp, RasterImage)
                tensors.append(inp[modality].image)
                cur_timestamps = inp[modality].timestamps
                if cur_timestamps is not None and len(cur_timestamps) > len(
                    timestamps_per_instance[idx]
                ):
                    timestamps_per_instance[idx] = cur_timestamps
            tensors = [inp[modality].image for inp in context.inputs]
            device = tensors[0].device
            max_t = max(t.shape[1] for t in tensors)
            max_timesteps = max(max_timesteps, max_t)
            modality_data[modality] = (
                tensors,
                len(Modality.get(modality).band_sets),
            )

        # Second pass: pad and process each modality with global max_timesteps
        for modality in present_modalities:
            tensors, num_band_sets = modality_data[modality]

            # Pad tensors to target_ch and track original timesteps for masking
            padded = []
            original_timesteps = []
            for t in tensors:
                orig_t = t.shape[1]
                original_timesteps.append(orig_t)
                if orig_t < max_timesteps:
                    pad = torch.zeros(
                        t.shape[:1] + (max_timesteps - orig_t,) + t.shape[2:],
                        dtype=t.dtype,
                        device=device,
                    )
                    t = torch.cat([t, pad], dim=1)
                padded.append(t)

            cur = torch.stack(padded, dim=0)
            cur = rearrange(cur, "b c t h w -> b h w t c")
            kwargs[modality] = cur

            # Create mask: ONLINE_ENCODER for valid, MISSING for padded timesteps
            b, h, w = cur.shape[0], cur.shape[1], cur.shape[2]
            mask = torch.full(
                (b, h, w, max_timesteps, num_band_sets),
                fill_value=MaskValue.ONLINE_ENCODER.value,
                dtype=torch.int32,
                device=device,
            )
            for sample_idx, orig_t in enumerate(original_timesteps):
                if orig_t < max_timesteps:
                    mask[sample_idx, :, :, orig_t:, :] = MaskValue.MISSING.value
            kwargs[f"{modality}_mask"] = mask

        if self.use_legacy_timestamps:
            # Note that only months (0 to 11) are used in OlmoEarth position encoding.
            timestamps = torch.zeros(
                (len(context.inputs), max_timesteps, 3),
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
