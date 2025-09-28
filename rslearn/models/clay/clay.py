"""Clay models."""

from __future__ import annotations

from enum import Enum
from typing import Any

import torch
import yaml
from claymodel.module import ClayMAEModule
from huggingface_hub import hf_hub_download

from rslearn.train.transforms.transform import Transform


class ClaySize(str, Enum):
    """Size of the Clay model."""

    BASE = "base"
    LARGE = "large"


PATCH_SIZE = 8
CLAY_MODALITIES = ["sentinel-2-l2a", "sentinel-1-rtc", "landsat-c2l1", "naip"]


def get_clay_checkpoint_path(
    filename: str = "v1.5/clay-v1.5.ckpt",
    repo_id: str = "made-with-clay/Clay",
) -> str:
    """Return a cached local path to the Clay ckpt from the Hugging Face Hub."""
    return hf_hub_download(repo_id=repo_id, filename=filename)  # nosec B615


def get_clay_metadata_path(
    filename: str = "configs/metadata.yaml",
    repo_id: str = "Clay-foundation/model",
) -> str:
    """Return a cached local path to Clay's metadata.yaml from the Hugging Face Hub."""
    return hf_hub_download(repo_id=repo_id, filename=filename)  # nosec B615


class Clay(torch.nn.Module):
    """Clay backbones."""

    def __init__(
        self,
        model_size: ClaySize,
        modalities: list[str] = ["sentinel-2-l2a"],
        checkpoint_path: str | None = None,
        metadata_path: str | None = None,
    ) -> None:
        """Initialize the Clay model.

        Args:
            model_size: The size of the Clay model.
            modalities: The modalities to use (subset of CLAY_MODALITIES).
            checkpoint_path: Path to clay-v1.5.ckpt; if None, fetch from HF Hub.
            metadata_path: Path to metadata.yaml; if None, fetch from HF Hub.
        """
        super().__init__()

        for m in modalities:
            if m not in CLAY_MODALITIES:
                raise ValueError(f"Invalid modality: {m}")

        if model_size == ClaySize.LARGE:
            ckpt = checkpoint_path or get_clay_checkpoint_path()
        elif model_size == ClaySize.BASE:
            raise ValueError("Clay BASE is not supported for v1.5 in this wrapper.")
        else:
            raise ValueError(f"Invalid model size: {model_size}")

        md_path = metadata_path or get_clay_metadata_path()

        # Load model + metadata
        module = ClayMAEModule.load_from_checkpoint(
            ckpt,
            model_size="large",
            metadata_path=md_path,
            dolls=[16, 32, 64, 128, 256, 768, 1024],
            doll_weights=[1, 1, 1, 1, 1, 1, 1],
            mask_ratio=0.0,
            shuffle=False,
        )
        module.eval()
        self.model = module

        with open(md_path) as f:
            self.metadata = yaml.safe_load(f)

        self.model_size = model_size
        self.modalities = modalities
        self._depth = 1024  # Clay v1.5 CLS embedding size

    def forward(self, inputs: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Forward pass for the Clay model.

        Args:
            inputs: input dicts that must include modalities as keys which are defined in the self.modalities list

        Returns:
            List[torch.Tensor]: Single-scale feature tensors from the encoder.
                                For Clay, this is a pooled map with shape (B, D, 1, 1).
        """
        if len(inputs) == 0:
            raise ValueError("Empty inputs list.")

        embeddings = []

        for modality in self.modalities:
            if modality not in inputs[0]:
                continue

            chips = torch.stack([inp[modality] for inp in inputs], dim=0).float()
            device, dtype = chips.device, chips.dtype

            order = self.metadata[modality]["band_order"]
            wavelengths = torch.tensor(
                [[self.metadata[modality]["bands"]["wavelength"][b] for b in order]],
                device=device,
                dtype=dtype,
            )

            # Fill in minimal datacube (time/latlon zeros are valid placeholders)
            datacube = {
                "platform": modality,
                "time": torch.zeros(chips.shape[0], 4, device=device, dtype=dtype),
                "latlon": torch.zeros(chips.shape[0], 4, device=device, dtype=dtype),
                "pixels": chips,
                "gsd": torch.tensor(
                    self.metadata[modality]["gsd"], device=device, dtype=dtype
                ),
                "waves": wavelengths,
            }

            with torch.no_grad():
                tokens, *_ = self.model.model.encoder(datacube)  # (B, 1+N, D)
                emb = tokens[:, 0, :]  # CLS token
            embeddings.append(emb)

        if not embeddings:
            raise ValueError(
                f"No valid modalities present in inputs. Expected one of {self.modalities}."
            )

        fused = torch.stack(embeddings, dim=0).mean(dim=0)  # (B, D)
        return [fused[:, :, None, None]]

    def get_backbone_channels(self) -> list:
        """Return output channels of this model when used as a backbone."""
        if self.model_size == ClaySize.LARGE:
            depth = self._depth
        elif self.model_size == ClaySize.BASE:
            depth = 768
        else:
            raise ValueError(f"Invalid model size: {self.model_size}")
        return [(PATCH_SIZE, depth)]


class ClayNormalize(Transform):
    """Normalize inputs using Clay metadata."""

    def __init__(self, metadata: dict) -> None:
        """Initialize a new ClayNormalize."""
        super().__init__()
        self.metadata = metadata

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
            meta = self.metadata[modality]
            means = [meta["bands"]["mean"][b] for b in meta["band_order"]]
            stds = [meta["bands"]["std"][b] for b in meta["band_order"]]
            input_dict[modality] = self.apply_image(input_dict[modality], means, stds)
        return input_dict, target_dict
