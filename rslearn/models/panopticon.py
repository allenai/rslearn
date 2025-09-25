"""Wrapper for the Panopticon model."""

import torch
from torch import nn
from einops import rearrange, repeat
import yaml
import torch.nn.functional as F
from importlib import resources
import math
from enum import StrEnum
from typing import Any
from rslearn.log_utils import get_logger

logger = get_logger(__name__)


class PanopticonModalities(StrEnum):
    """Modalities supported by Panopticon."""
    SENTINEL2 = "sentinel2"
    LANDSAT8 = "landsat8"
    SENTINEL1 = "sentinel1"

class Panopticon(nn.Module):
    """Class containing the Panopticon model that can ingest MaskedHeliosSample objects."""

    supported_modalities: list[str] = [PanopticonModalities.SENTINEL2, PanopticonModalities.LANDSAT8,PanopticonModalities.SENTINEL1] # also supports others but not set up yet
    patch_size: int = 14
    base_image_size: int = 224

    def __init__(
        self,
        band_order: dict[str, list[str]],
        torchhub_id: str = "panopticon_vitb14",
    ):
        """Initialize the Panopticon wrapper.

        Args:
            torchhub_id: The torch hub model ID for panopticon
        """
        super().__init__()
        # Load the panopticon model
        self._load_model(torchhub_id)
        self.output_dim = self.model.embed_dim
        self.band_order = band_order
    def _load_model(self, torchhub_id: str) -> None:
        """Load the panopticon model from torch hub."""
        import time

        # Hack to get around https://discuss.pytorch.org/t/torch-hub-load-gives-httperror-rate-limit-exceeded/124769
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        for attempt in range(2):
            try:
                self.model = torch.hub.load(
                    "panopticon-FM/panopticon",
                    torchhub_id,
                )
                break
            except Exception as e:
                logger.warning(
                    f"Error loading panopticon model: {e}. Retrying in 5 seconds..."
                )
                time.sleep(5)
        else:
            raise RuntimeError(
                f"Failed to load panopticon model {torchhub_id} after retrying."
            )

    def _process_modality_data(self, data: torch.Tensor) -> torch.Tensor:
        """Process individual modality data.

        Args:
            data: Input tensor of shape [B, C, H, W]
        Returns:
            Processed tensor of shape [B, C, H, W]
        """

        original_height = data.shape[2]
        new_height = self.patch_size if original_height == 1 else self.base_image_size

        data = F.interpolate(
            data,
            size=(new_height, new_height),
            mode="bilinear",
            align_corners=False,
        )
        return data

    def _create_channel_ids(
        self, modality: str, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Create channel IDs for the panopticon model."""
        with resources.open_text(
            "rslearn.models.panopticon_data.sensors", f"{modality}.yaml"
        ) as f:
            sensor_config = yaml.safe_load(f)

        band_order = self.band_order[modality]
        chn_ids = [sensor_config["bands"][band.upper()]["gaussian"]["mu"] for band in band_order]
        chn_ids = torch.tensor(chn_ids, dtype=torch.float32, device=device)
        chn_ids = repeat(chn_ids, "c -> b c", b=batch_size)
        return chn_ids

    def prepare_input(
        self, input_data: dict[str, torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        """Prepare input for the panopticon model from MaskedHeliosSample."""
        channel_ids_list: list[torch.Tensor] = []
        processed_data_list: list[torch.Tensor] = []
        for modality in self.supported_modalities:
            if modality not in input_data.keys():
                logger.debug(f"Modality {modality} not found in input data")
                continue
            data = input_data[modality]
            device = data.device
            processed_data = self._process_modality_data(data)
            processed_data_list.append(processed_data)
            batch_size = processed_data.shape[0]
            chn_ids = self._create_channel_ids(modality, batch_size, device)
            channel_ids_list.append(chn_ids)

        processed_data = torch.cat(processed_data_list, dim=1)
        chn_ids = torch.cat(channel_ids_list, dim=1)
        return {
            "imgs": processed_data,
            "chn_ids": chn_ids,
        }

    def forward(self, inputs: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Forward pass through the panopticon model."""
        batch_inputs = {key: torch.stack([inp[key] for inp in inputs], dim=0) for key in inputs[0].keys()}
        panopticon_inputs = self.prepare_input(batch_inputs)
        output_features = self.model.forward_features(panopticon_inputs)[
                "x_norm_patchtokens"
            ]

        num_tokens = output_features.shape[1]
        height = int(math.sqrt(num_tokens))
        output_features = rearrange(
                output_features, "b (h w) d -> b d h w", h=height, w=height
            )
        return [output_features]

    def get_backbone_channels(self) -> list:
        """Returns the output channels of this model when used as a backbone.

        The output channels is a list of (downsample_factor, depth) that corresponds
        to the feature maps that the backbone returns. For example, an element [2, 32]
        indicates that the corresponding feature map is 1/2 the input resolution and
        has 32 channels.
        """
        return [(self.patch_size, self.output_dim)]