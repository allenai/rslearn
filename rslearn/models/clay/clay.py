"""Clay models."""

from __future__ import annotations

import math
from enum import Enum
from importlib.resources import files
from typing import Any

import torch
import yaml
from einops import rearrange
from huggingface_hub import hf_hub_download

# from claymodel.module import ClayMAEModule
from terratorch.models.backbones.clay_v15.module import ClayMAEModule

from rslearn.train.transforms.transform import Transform


class ClaySize(str, Enum):
    """Size of the Clay model."""

    BASE = "base"
    LARGE = "large"


PATCH_SIZE = 8
CLAY_MODALITIES = ["sentinel-2-l2a", "sentinel-1-rtc", "landsat-c2l1", "naip"]
CONFIG_DIR = files("rslearn.models.clay.configs")
CLAY_METADATA_PATH = str(CONFIG_DIR / "metadata.yaml")


def get_clay_checkpoint_path(
    filename: str = "v1.5/clay-v1.5.ckpt",
    repo_id: str = "made-with-clay/Clay",
) -> str:
    """Return a cached local path to the Clay ckpt from the Hugging Face Hub."""
    return hf_hub_download(repo_id=repo_id, filename=filename)  # nosec B615


class Clay(torch.nn.Module):
    """Clay backbones."""

    def __init__(
        self,
        model_size: ClaySize,
        modality: str = "sentinel-2-l2a",
        checkpoint_path: str | None = None,
        metadata_path: str = CLAY_METADATA_PATH,
    ) -> None:
        """Initialize the Clay model.

        Args:
            model_size: The size of the Clay model.
            modality: The modality to use (subset of CLAY_MODALITIES).
            checkpoint_path: Path to clay-v1.5.ckpt, if None, fetch from HF Hub.
            metadata_path: Path to metadata.yaml.
        """
        super().__init__()

        # Clay only supports single modality input
        if modality not in CLAY_MODALITIES:
            raise ValueError(f"Invalid modality: {modality}")

        ckpt = checkpoint_path or get_clay_checkpoint_path()
        if model_size == ClaySize.LARGE:
            self.model = ClayMAEModule.load_from_checkpoint(
                checkpoint_path=ckpt,
                model_size="large",
                metadata_path=metadata_path,
                dolls=[16, 32, 64, 128, 256, 768, 1024],
                doll_weights=[1, 1, 1, 1, 1, 1, 1],
                mask_ratio=0.0,
                shuffle=False,
            )
        elif model_size == ClaySize.BASE:
            # Failed to load Base model in Clay v1.5
            raise ValueError("Clay BASE model currently not supported in v1.5.")
            self.model = ClayMAEModule.load_from_checkpoint(
                checkpoint_path=ckpt,
                model_size="base",
                metadata_path=metadata_path,
                dolls=[16, 32, 64, 128, 256, 768],
                doll_weights=[1, 1, 1, 1, 1, 1],
                mask_ratio=0.0,
                shuffle=False,
            )
        else:
            raise ValueError(f"Invalid model size: {model_size}")

        with open(metadata_path) as f:
            self.metadata = yaml.safe_load(f)

        self.model_size = model_size
        self.modality = modality

    def forward(self, inputs: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Forward pass for the Clay model.

        Args:
            inputs: input dicts that must include `self.modality` as a key

        Returns:
            List[torch.Tensor]: Single-scale feature tensors from the encoder.
        """
        if self.modality not in inputs[0]:
            raise ValueError(f"Missing modality {self.modality} in inputs.")

        param = next(self.model.parameters())
        device = param.device

        chips = torch.stack(
            [inp[self.modality] for inp in inputs], dim=0
        )  # (B, C, H, W)

        order = self.metadata[self.modality]["band_order"]
        wavelengths = []
        for band in self.metadata[self.modality]["band_order"]:
            wavelengths.append(
                self.metadata[self.modality]["bands"]["wavelength"][band] * 1000
            )  # Convert to nm
        # Check channel count matches Clay expectation
        if chips.shape[1] != len(order):
            raise ValueError(
                f"Channel count {chips.shape[1]} does not match expected {len(order)} for {self.modality}"
            )

        # Time & latlon zeros are valid per Clay doc
        # https://clay-foundation.github.io/model/getting-started/basic_use.html
        datacube = {
            "platform": self.modality,
            "time": torch.zeros(chips.shape[0], 4).to(device),
            "latlon": torch.zeros(chips.shape[0], 4).to(device),
            "pixels": chips.to(device),
            "gsd": torch.tensor(self.metadata[self.modality]["gsd"]).to(device),
            "waves": torch.tensor(wavelengths).to(device),
        }

        tokens, *_ = self.model.model.encoder(datacube)  # (B, 1 + N, D)

        # Remove CLS token
        spatial = tokens[:, 1:, :]  # (B, N, D)
        n_tokens = spatial.shape[1]
        side = int(math.isqrt(n_tokens))
        if chips.shape[2] != side * PATCH_SIZE or chips.shape[3] != side * PATCH_SIZE:
            raise ValueError(
                f"Input spatial size {(chips.shape[2], chips.shape[3])} is not compatible with patch size {PATCH_SIZE}"
            )

        features = rearrange(spatial, "b (h w) d -> b d h w", h=side, w=side)
        return [features]

    def get_backbone_channels(self) -> list:
        """Return output channels of this model when used as a backbone."""
        if self.model_size == ClaySize.LARGE:
            depth = 1024
        elif self.model_size == ClaySize.BASE:
            depth = 768
        else:
            raise ValueError(f"Invalid model size: {self.model_size}")
        return [(PATCH_SIZE, depth)]


class ClayNormalize(Transform):
    """Normalize inputs using Clay metadata.

    For Sentinel-1, the intensities should be converted to decibels.
    """

    def __init__(self, metadata_path: str = CLAY_METADATA_PATH) -> None:
        """Initialize ClayNormalize."""
        super().__init__()
        with open(metadata_path) as f:
            self.metadata = yaml.safe_load(f)

    def apply_image(
        self, image: torch.Tensor, means: list[float], stds: list[float]
    ) -> torch.Tensor:
        """Normalize the specified image with Clay normalization."""
        x = image.float()
        if x.shape[0] != len(means):
            raise ValueError(
                f"channel count {x.shape[0]} does not match provided band stats {len(means)}"
            )
        for c in range(x.shape[0]):
            x[c] = (x[c] - means[c]) / stds[c]
        return x

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Normalize the specified image with Clay normalization."""
        for modality in CLAY_MODALITIES:
            if modality not in input_dict or modality not in self.metadata:
                continue
            modality_metadata = self.metadata[modality]
            means = [
                modality_metadata["bands"]["mean"][b]
                for b in modality_metadata["band_order"]
            ]
            stds = [
                modality_metadata["bands"]["std"][b]
                for b in modality_metadata["band_order"]
            ]
            input_dict[modality] = self.apply_image(input_dict[modality], means, stds)
        return input_dict, target_dict
