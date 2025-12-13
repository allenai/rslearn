"""OlmoEarth model wrapper for fine-tuning in rslearn."""

import json
from contextlib import nullcontext
from typing import Any

import torch
from einops import rearrange
from olmo_core.config import Config
from olmo_core.distributed.checkpoint import load_model_and_optim_state
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
from rslearn.models.component import FeatureExtractor, FeatureMaps
from rslearn.train.model_context import ModelContext

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
        """
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

    def _load_model_from_checkpoint(
        self, checkpoint_upath: UPath, random_initialization: bool
    ) -> torch.nn.Module:
        """Load the OlmoEarth pre-trained model from a distributed checkpoint folder.

        The folder should contain config.json as well as the model_and_optim folder
        that contains the distributed checkpoint. This is the format produced by
        pre-training runs in olmoearth_pretrain.
        """
        # Load the model config and initialize it.
        # We avoid loading the train module here because it depends on running within
        # olmo_core.
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

    def forward(self, context: ModelContext) -> FeatureMaps:
        """Compute feature maps from the OlmoEarth backbone.

        Args:
            context: the model context. Input dicts should include keys corresponding
                to the modalities that should be passed to the OlmoEarth model.

        Returns:
            a FeatureMaps consisting of one feature map, at 1/patch_size of the input
                resolution. Embeddings will be pooled across modalities and timesteps.
        """
        kwargs = {}
        present_modalities = []
        device = None
        # Handle the case where some modalities are multitemporal and some are not.
        # We assume all multitemporal modalities have the same number of timesteps.
        max_timesteps = 1
        for modality in MODALITY_NAMES:
            if modality not in context.inputs[0]:
                continue
            present_modalities.append(modality)
            cur = torch.stack([inp[modality] for inp in context.inputs], dim=0)
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
            (len(context.inputs), max_timesteps, 3), dtype=torch.int32, device=device
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
            torch_context = nullcontext()
        else:
            assert device is not None
            torch_context = torch.amp.autocast(
                device_type=device.type, dtype=self.autocast_dtype
            )

        with torch_context:
            # Currently we assume the provided model always returns a TokensAndMasks object.
            tokens_and_masks: TokensAndMasks
            if isinstance(self.model, Encoder):
                # Encoder has a fast_pass argument to indicate mask is not needed.
                tokens_and_masks = self.model(
                    sample,
                    fast_pass=True,
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
                modality_features = getattr(tokens_and_masks, modality)
                # Pool over band sets and timesteps (BHWTSC -> BHWC).
                pooled = modality_features.mean(dim=[3, 4])
                # We want BHWC -> BCHW.
                pooled = rearrange(pooled, "b h w c -> b c h w")
                features.append(pooled)
            # Pool over the modalities, so we get one BCHW feature map.
            pooled = torch.stack(features, dim=0).mean(dim=0)
        else:
            for modality in present_modalities:
                modality_features = getattr(tokens_and_masks, modality)
                # Aggregate over band sets and timesteps (BHWTSC -> BHWC).
                modality_features = rearrange(
                    modality_features, "b h w t s c -> b c h w (t s)"
                )
                features.append(modality_features)
            pooled = torch.cat(features, dim=-1)
        return FeatureMaps([pooled])

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
