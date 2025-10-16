"""OlmoEarth model wrapper for fine-tuning in rslearn."""

import json
from contextlib import nullcontext
from typing import Any

import torch
from einops import rearrange
from olmo_core.config import Config
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexihelios import Encoder, TokensAndMasks
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue
from upath import UPath

from rslearn.log_utils import get_logger

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


class OlmoEarth(torch.nn.Module):
    """A wrapper to support the OlmoEarth model."""

    def __init__(
        self,
        # TODO: we should accept model ID instead of checkpoint_path once we are closer
        # to being ready for release.
        checkpoint_path: str,
        selector: list[str | int] = [],
        forward_kwargs: dict[str, Any] = {},
        random_initialization: bool = False,
        embedding_size: int | None = None,
        patch_size: int | None = None,
        autocast_dtype: str | None = "bfloat16",
    ):
        """Create a new OlmoEarth model.

        Args:
            checkpoint_path: the checkpoint directory to load. It should contain
                config.json file as well as model_and_optim folder.
            selector: an optional sequence of attribute names or list indices to select
                the sub-module that should be applied on the input images.
            forward_kwargs: additional arguments to pass to forward pass besides the
                 MaskedOlmoEarthSample.
            random_initialization: whether to skip loading the checkpoint so the
                weights are randomly initialized. In this case, the checkpoint is only
                used to define the model architecture.
            embedding_size: optional embedding size to report via
                get_backbone_channels.
            patch_size: optional patch size to report via get_backbone_channels.
            autocast_dtype: which dtype to use for autocasting, or set None to disable.
        """
        super().__init__()
        _checkpoint_path = UPath(checkpoint_path)
        self.forward_kwargs = forward_kwargs
        self.embedding_size = embedding_size
        self.patch_size = patch_size

        if autocast_dtype is not None:
            self.autocast_dtype = AUTOCAST_DTYPE_MAP[autocast_dtype]
        else:
            self.autocast_dtype = None

        # Load the model config and initialize it.
        # We avoid loading the train module here because it depends on running within
        # olmo_core.
        with (_checkpoint_path / "config.json").open() as f:
            config_dict = json.load(f)
            model_config = Config.from_dict(config_dict["model"])

        model = model_config.build()

        # Load the checkpoint.
        if not random_initialization:
            train_module_dir = _checkpoint_path / "model_and_optim"
            if train_module_dir.exists():
                load_model_and_optim_state(str(train_module_dir), model)
                logger.info(f"loaded OlmoEarth encoder from {train_module_dir}")
            else:
                logger.info(f"could not find OlmoEarth encoder at {train_module_dir}")
        else:
            logger.info("skipping loading OlmoEarth encoder")

        # Select just the portion of the model that we actually want to use.
        for part in selector:
            if isinstance(part, str):
                model = getattr(model, part)
            else:
                model = model[part]
        self.model = model

    def forward(self, inputs: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Compute feature maps from the OlmoEarth backbone.

        Inputs:
            inputs: input dicts. It should include keys corresponding to the modalities
                that should be passed to the OlmoEarth model.
        """
        kwargs = {}
        present_modalities = []
        device = None
        # Handle the case where some modalities are multitemporal and some are not.
        # We assume all multitemporal modalities have the same number of timesteps.
        max_timesteps = 1
        for modality in MODALITY_NAMES:
            if modality not in inputs[0]:
                continue
            present_modalities.append(modality)
            cur = torch.stack([inp[modality] for inp in inputs], dim=0)
            device = cur.device
            # Check if it's single or multitemporal, and reshape accordingly
            num_bands = Modality.get(modality).num_bands
            num_timesteps = cur.shape[1] // num_bands
            max_timesteps = max(max_timesteps, num_timesteps)
            cur = rearrange(cur, "b (t c) h w -> b h w t c", t=num_timesteps)
            kwargs[modality] = cur
            # Create mask array which is BHWTS (without channels but with band sets).
            num_band_sets = len(Modality.get(modality).band_sets)
            mask_shape = cur.shape[0:4] + (num_band_sets,)
            mask = (
                torch.ones(mask_shape, dtype=torch.int32, device=device)
                * MaskValue.ONLINE_ENCODER.value
            )
            kwargs[f"{modality}_mask"] = mask

        # Timestamps is required.
        # Note that only months (0 to 11) are used in OlmoEarth position encoding.
        # For now, we assign same timestamps to all inputs, but later we should handle varying timestamps per input.
        timestamps = torch.zeros(
            (len(inputs), max_timesteps, 3), dtype=torch.int32, device=device
        )
        timestamps[:, :, 0] = 1  # day
        timestamps[:, :, 1] = torch.arange(max_timesteps, device=device)[
            None, :
        ]  # month
        timestamps[:, :, 2] = 2024  # year
        kwargs["timestamps"] = timestamps

        sample = MaskedOlmoEarthSample(**kwargs)

        # Decide context based on self.autocast_dtype.
        if self.autocast_dtype is None:
            context = nullcontext()
        else:
            assert device is not None
            context = torch.amp.autocast(
                device_type=device.type, dtype=self.autocast_dtype
            )

        with context:
            # Currently we assume the provided model always returns a TokensAndMasks object.
            tokens_and_masks: TokensAndMasks
            if isinstance(self.model, Encoder):
                # Encoder has a fast_pass argument to indicate mask is not needed.
                tokens_and_masks = self.model(
                    sample, fast_pass=True, **self.forward_kwargs
                )["tokens_and_masks"]
            else:
                # Other models like STEncoder do not have this option supported.
                tokens_and_masks = self.model(sample, **self.forward_kwargs)[
                    "tokens_and_masks"
                ]

        # Apply temporal/modality pooling so we just have one feature per patch.
        features = []
        for modality in present_modalities:
            modality_features = getattr(tokens_and_masks, modality)
            # Pool over band sets and timesteps (BHWTSC -> BHWC).
            pooled = modality_features.mean(dim=[3, 4])
            # We want BHWC -> BCHW.
            pooled = rearrange(pooled, "b h w c -> b c h w")
            features.append(pooled)
        # Pool over the modalities, so we get one BCHW feature map.
        pooled = torch.stack(features, dim=0).mean(dim=0)
        return [pooled]

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
