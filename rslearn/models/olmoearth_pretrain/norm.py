"""Normalization transforms."""

import json
from typing import Any

from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize import (
    load_computed_config,
)

from rslearn.log_utils import get_logger
from rslearn.train.transforms.transform import Transform, selector_exists

logger = get_logger(__file__)


class OlmoEarthNormalize(Transform):
    """Normalize using OlmoEarth JSON config.

    For Sentinel-1 data, the values should be converted to decibels before being passed
    to this transform.
    """

    def __init__(
        self,
        band_names: dict[str, list[str]],
        std_multiplier: float | None = 2,
        config_fname: str | None = None,
        skip_missing: bool = False,
    ) -> None:
        """Initialize a new OlmoEarthNormalize.

        Args:
            band_names: map from modality name to the list of bands in that modality in
                the order they are being loaded. Note that this order must match the
                expected order for the OlmoEarth model.
            std_multiplier: the std multiplier matching the one used for the model
                training in OlmoEarth.
            config_fname: load the normalization configuration from this file, instead
                of getting it from OlmoEarth.
            skip_missing: if True, skip modalities that don't exist in the input dict.
                Useful when working with optional inputs.
        """
        super().__init__()
        self.band_names = band_names
        self.std_multiplier = std_multiplier
        self.skip_missing = skip_missing

        if config_fname is None:
            self.norm_config = load_computed_config()
        else:
            logger.warning(
                f"Loading normalization config from {config_fname}. This argument is deprecated and will be removed in a future version."
            )
            with open(config_fname) as f:
                self.norm_config = json.load(f)

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply normalization over the inputs and targets.

        Args:
            input_dict: the input
            target_dict: the target

        Returns:
            normalized (input_dicts, target_dicts) tuple
        """
        for modality_name, cur_band_names in self.band_names.items():
            # Skip missing modalities if skip_missing is enabled
            if self.skip_missing and not selector_exists(
                input_dict, target_dict, modality_name
            ):
                continue

            band_norms = self.norm_config[modality_name]
            image = input_dict[modality_name]
            # Keep a set of indices to make sure that we normalize all of them.
            needed_band_indices = set(range(image.image.shape[0]))
            num_timesteps = image.image.shape[0] // len(cur_band_names)

            for band, norm_dict in band_norms.items():
                # If multitemporal, normalize each timestep separately.
                for t in range(num_timesteps):
                    band_idx = cur_band_names.index(band) + t * len(cur_band_names)
                    min_val = norm_dict["mean"] - self.std_multiplier * norm_dict["std"]
                    max_val = norm_dict["mean"] + self.std_multiplier * norm_dict["std"]
                    image.image[band_idx] = (image.image[band_idx] - min_val) / (
                        max_val - min_val
                    )
                    needed_band_indices.remove(band_idx)

            if len(needed_band_indices) > 0:
                raise ValueError(
                    f"for modality {modality_name}, bands {needed_band_indices} were unexpectedly not normalized"
                )

        return input_dict, target_dict
