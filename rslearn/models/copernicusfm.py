"""Copernicus FM model."""

from enum import Enum
import logging
import math

import torch
import torch.nn.functional as F
from einops import rearrange
from upath import UPath

from .copernicusfm.model_vit import vit_base_patch16


logger = logging.getLogger(__name__)
class CopernicusFMModality(Enum):
    """Modality for Copernicus FM."""
    SENTINEL2_L2A = "sentinel2_l2a"
    SENTINEL1 = "sentinel1"

MODALITY_TO_WAVELENGTH_BANDWIDTHS: dict[str, dict[str, list]] = {
    # https://github.com/zhu-xlab/Copernicus-FM/blob/main/Copernicus-Bench/src/configs/dataset/cobench_eurosat_s2.yaml
    CopernicusFMModality.SENTINEL2_L2A.value: {
        "band_names": [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B10",
            "B11",
            "B12",
        ],
        "band_wavelengths": [
            440,
            490,
            560,
            665,
            705,
            740,
            783,
            842,
            860,
            940,
            1370,
            1610,
            2190,
        ],
        "band_bandwidths": [20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 30, 90, 180],
    },
    # https://github.com/zhu-xlab/Copernicus-FM/blob/main/Copernicus-Bench/src/configs/dataset/cobench_eurosat_s1.yaml
    CopernicusFMModality.SENTINEL1.value: {
        "band_names": ["vv", "vh"],
        "band_wavelengths": [50000000, 50000000],
        "band_bandwidths": [1e9, 1e9],
    },
}


class CopernicusFM(torch.nn.Module):
    """Wrapper for Copernicus FM to ingest Masked Helios Sample."""

    image_resolution = 224
    patch_size = 16
    input_mode = "spectral"
    supported_modalities = [CopernicusFMModality.SENTINEL2_L2A.value, CopernicusFMModality.SENTINEL1.value]

    def __init__(
        self, band_order: dict[str, list[str]], load_directory: str = "/weka/dfive-default/helios/models/copernicusfm"
    ) -> None:
        """Initialize the Copernicus FM wrapper.

        Args:
            load_directory: The directory to load from
        """
        super().__init__()

        # global_pool=True so that we initialize the fc_norm layer
        self.band_order = band_order
        self.model = vit_base_patch16(num_classes=10, global_pool=True)
        check_point = torch.load(
            UPath(load_directory) / "CopernicusFM_ViT_base_varlang_e100.pth"
        )
        if "model" in check_point:
            state_dict = check_point["model"]
        else:
            state_dict = check_point
        self.model.load_state_dict(state_dict, strict=False)

        # take MODALITY_TO_WAVELENGTH_BANDWIDTHS and rearrage it so that it has the same
        # ordering as the Helios band orders, defined by Modality.band_order
        self.modality_to_wavelength_bandwidths = {}
        for modality in self.supported_modalities:
            wavelength_bandwidths = MODALITY_TO_WAVELENGTH_BANDWIDTHS[modality]
            band_order = MODALITY_TO_WAVELENGTH_BANDWIDTHS[modality]["band_names"]
            wavelengths = []
            bandwidths = []
            band_order = self.band_order.get(modality, None)
            if band_order is None:
                logger.warning(f"Band order for modality {modality} not found in band_order dictionary, unable to use this modality unless specified")
                continue
            for b in band_order:
                cfm_idx = wavelength_bandwidths["band_names"].index(b)
                wavelengths.append(wavelength_bandwidths["band_wavelengths"][cfm_idx])
                bandwidths.append(wavelength_bandwidths["band_bandwidths"][cfm_idx])
            self.modality_to_wavelength_bandwidths[modality] = {
                "band_bandwidths": bandwidths,
                "band_wavelengths": wavelengths,
            }

    def _resize_data(
        self, data: torch.Tensor
    ) -> list[torch.Tensor]:
        """Process individual modality data.

        Args:
            data: Input tensor of shape [B, C, H, W]

        Returns:
            list of tensors of shape [B, C, H, W]
        """
        # Get original dimensions
        original_height = data.shape[2]
        new_height = (
            self.patch_size if original_height == 1 else self.image_resolution
        )
        data = F.interpolate(
            data,
            size=(new_height, new_height),
            mode="bilinear",
            align_corners=False,
        )
        return data

    def prepare_input(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[int], list[int]]:
        """Prepare input for the CopernicusFM model from MaskedHeliosSample."""
        wavelengths: list[int] = []
        bandwidths: list[int] = []
        all_processed_data: list[list[torch.Tensor]] = []
        for modality in inputs.keys():
            if modality not in self.supported_modalities:
                logger.warning(
                    f"Skipping modality {modality} as it is not in the supported "
                    f"modalities list {self.supported_modalities}"
                )
                continue

            data = inputs[modality]

            if data is None:
                continue

            all_processed_data.append(self._resize_data(data))
            wavelengths.extend(
                self.modality_to_wavelength_bandwidths[modality]["band_wavelengths"]
            )
            bandwidths.extend(
                self.modality_to_wavelength_bandwidths[modality]["band_bandwidths"]
            )

        concatenated_processed_data = torch.cat(all_processed_data, dim=1)
        return concatenated_processed_data, wavelengths, bandwidths

    def forward(
        self,
        inputs: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Forward pass through CopernicusFM model."""
        batch_inputs = {key: torch.stack([inp[key] for inp in inputs], dim=0) for key in inputs[0].keys()}
        # Prepare input
        data, wavelengths, bandwidths = self.prepare_input(
            batch_inputs
        )
        meta = torch.full(
            (1, 4), float("nan"), device=data.device
        )  # [lon, lat, delta_time, patch_token_area], assume unknown
        # "The embed tensor contains the encoded image features, which can be used for downstream tasks."
        _, timestep_output = self.model(
            data,
            meta,
            wavelengths,
            bandwidths,
            None,
            self.input_mode,
            self.patch_size,
            )
        # no norm, following
        # https://github.com/zhu-xlab/Copernicus-FM/blob/main/Copernicus-Bench/src/foundation_models/CopernicusFM/models_dwv_seg.py
        side = math.isqrt(timestep_output.shape[1])
        output_features = rearrange(
            timestep_output, "b (h w) c -> b c h w ", h=side, w=side
        )
        return [output_features]

    def get_backbone_channels(self) -> list[tuple[int, int]]:
        """Returns the output channels of this model when used as a backbone."""
        # TODO: load this from a constant depending on the model size
        return [(self.patch_size, 768)]
