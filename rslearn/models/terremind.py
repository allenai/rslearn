"""Terremind models."""

from enum import Enum
from typing import Any

import torch
from terratorch.registry import BACKBONE_REGISTRY


# create size for terramind as enum
class TerramindSize(str, Enum):
    """Size of the Terramind model."""

    BASE = "base"
    LARGE = "large"


PATCH_SIZE = 16
DEFAULT_IMAGE_SIZE = 264

# TerraMind modalities
TERRAMIND_MODALITIES = ["S2L1C", "S2L2A", "S1GRD", "S1RTC", "RGB", "DEM"]

# TerraMind standardization values
TERRAMIND_MEANS = {
    "S2L1C": [
        2357.089,
        2137.385,
        2018.788,
        2082.986,
        2295.651,
        2854.537,
        3122.849,
        3040.560,
        3306.481,
        1473.847,
        506.070,
        2472.825,
        1838.929,
    ],
    "S2L2A": [
        1390.458,
        1503.317,
        1718.197,
        1853.910,
        2199.100,
        2779.975,
        2987.011,
        3083.234,
        3132.220,
        3162.988,
        2424.884,
        1857.648,
    ],
    "S1GRD": [-12.599, -20.293],
    "S1RTC": [-10.93, -17.329],
    "RGB": [87.271, 80.931, 66.667],
    "DEM": [670.665],
}

TERRAMIND_STDS = {
    "S2L1C": [
        1624.683,
        1675.806,
        1557.708,
        1833.702,
        1823.738,
        1733.977,
        1732.131,
        1679.732,
        1727.26,
        1024.687,
        442.165,
        1331.411,
        1160.419,
    ],
    "S2L2A": [
        2106.761,
        2141.107,
        2038.973,
        2134.138,
        2085.321,
        1889.926,
        1820.257,
        1871.918,
        1753.829,
        1797.379,
        1434.261,
        1334.311,
    ],
    "S1GRD": [5.195, 5.890],
    "S1RTC": [4.391, 4.459],
    "RGB": [58.767, 47.663, 42.631],
    "DEM": [951.272],
}


class Terramind(torch.nn.Module):
    """Terremind backbones."""

    def __init__(
        self,
        model_size: TerramindSize,
        image_size: int = DEFAULT_IMAGE_SIZE,
        modalities: list[str] = ["S2L2A"],
    ) -> None:
        """Initialize the Terramind model.

        Args:
            model_size: The size of the Terramind model.
            image_size: The size of the input image.
            modalities: The modalities to use.
        """
        super().__init__()

        # Check if all modalities are valid
        for modality in modalities:
            if modality not in TERRAMIND_MODALITIES:
                raise ValueError(f"Invalid modality: {modality}")

        # Check if image size is valid
        if image_size < PATCH_SIZE:
            raise ValueError(f"Image size must be at least {PATCH_SIZE}x{PATCH_SIZE}")

        if model_size == TerramindSize.BASE:
            self.model = BACKBONE_REGISTRY.build(
                "terramind_v1_base", modalities=modalities, pretrained=True
            )
        elif model_size == TerramindSize.LARGE:
            self.model = BACKBONE_REGISTRY.build(
                "terramind_v1_large", modalities=modalities, pretrained=True
            )
        else:
            raise ValueError(f"Invalid model size: {model_size}")

        self.image_size = image_size
        self.modalities = modalities
        self.height, self.width = image_size // PATCH_SIZE, image_size // PATCH_SIZE

    def forward(self, inputs: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Forward pass for the Terramind model.

        Args:
            inputs: input dicts that must include modalities as keys which are defined in the self.modalities list

        Returns:
            List[torch.Tensor]: Single-scale feature tensors from the encoder.
        """
        model_inputs = {}
        for modality in self.modalities:
            # We assume the all the inputs include the same modalities
            if modality not in inputs[0]:
                continue
            cur = torch.stack([inp[modality] for inp in inputs], dim=0)  # (B, C, H, W)
            model_inputs[modality] = cur

        # By default, the patch embeddings are averaged over all modalities to reduce output tokens
        # So the output shape is (B, N, D), where N is the number of patches and D is the embedding dimension
        image_features = self.model(model_inputs)
        batch_size = image_features.shape[0]
        # Image features are (B, D, H, W)
        return [
            image_features.reshape(batch_size, self.height, self.width, -1).permute(
                0, 3, 1, 2
            )
        ]

    def get_backbone_channels(self) -> list:
        """Returns the output channels of this model when used as a backbone.

        The output channels is a list of (patch_size, depth) that corresponds
        to the feature maps that the backbone returns.

        Returns:
            the output channels of the backbone as a list of (patch_size, depth) tuples.
        """
        # We need to
        if self.model_size == TerramindSize.BASE:
            depth = 768
        elif self.model_size == TerramindSize.LARGE:
            depth = 1024
        else:
            raise ValueError(f"Invalid model size: {self.model_size}")
        return [(PATCH_SIZE, depth)]
