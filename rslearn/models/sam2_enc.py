"""SegmentAnything2 encoders."""

from typing import Any

import torch
import torch.nn as nn
from sam2.build_sam import build_sam2
from upath import UPath


class SAM2Encoder(nn.Module):
    """SAM2's image encoder."""

    def __init__(self, model_identifier: str) -> None:
        """Initializes the SAM2Encoder with a specific model configuration and checkpoint.

        Args:
            model_identifier: Identifier for model type.
        """
        super().__init__()

        if "tiny" in model_identifier:
            model_cfg = "sam2_hiera_t.yaml"
            checkpoint_path = UPath(
                "gcs://rslearn-eai/artifacts/sam2/sam2_hiera_tiny.pt"
            )
            self.backbone_channels = [
                [4, 96],
                [8, 192],
                [16, 384],
                [32, 768],
            ]
        elif "small" in model_identifier:
            model_cfg = "sam2_hiera_s.yaml"
            checkpoint_path = UPath(
                "gcs://rslearn-eai/artifacts/sam2/sam2_hiera_small.pt"
            )
            self.backbone_channels = [
                [4, 96],
                [8, 192],
                [16, 384],
                [32, 768],
            ]
        elif "base" in model_identifier:
            model_cfg = "sam2_hiera_b+.yaml"
            checkpoint_path = UPath(
                "gcs://rslearn-eai/artifacts/sam2/sam2_hiera_base_plus.pt"
            )
            self.backbone_channels = [
                [4, 112],
                [8, 224],
                [16, 448],
                [32, 896],
            ]
        elif "large" in model_identifier:
            model_cfg = "sam2_hiera_l.yaml"
            checkpoint_path = UPath(
                "gcs://rslearn-eai/artifacts/sam2/sam2_hiera_large.pt"
            )
            self.backbone_channels = [
                [4, 144],
                [8, 288],
                [16, 576],
                [32, 1152],
            ]
        else:
            raise ValueError(f"Invalid model identifier: {model_identifier}")

        # Build the model and remove unnecessary components
        with checkpoint_path.open("rb") as f:
            self.model = build_sam2(model_cfg, f)
        self._remove_unused_modules()

        self.encoder = self.model.image_encoder.trunk

    def _remove_unused_modules(self) -> None:
        """Removes unused modules from the SAM2 model."""
        del self.model.sam_mask_decoder
        del self.model.sam_prompt_encoder
        del self.model.memory_encoder
        del self.model.memory_attention
        del self.model.mask_downsample
        del self.model.obj_ptr_tpos_proj
        del self.model.obj_ptr_proj
        del self.model.image_encoder.neck

    def forward(self, inputs: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Extract multi-scale features from a batch of images.

        Args:
            inputs: List of dictionaries, each containing the input image under the key 'image'.

        Returns:
            List[torch.Tensor]: Multi-scale feature tensors from the encoder.
        """
        images = torch.stack([inp["image"] for inp in inputs], dim=0)
        features = self.encoder(images)
        return features

    def get_backbone_channels(self) -> list[list[int]]:
        """Returns the output channels of the encoder at different scales.

        Returns:
            List[List[int]]: List of downsample factors and corresponding channel counts at each scale.
        """
        return self.backbone_channels
