"""CROMA models."""

import shutil
import tempfile
import urllib.request
from enum import Enum
from typing import Any

import torch
from einops import rearrange
from upath import UPath

from rslearn.log_utils import get_logger
from rslearn.train.transforms.transform import Transform
from rslearn.utils.fsspec import open_atomic

from .use_croma import PretrainedCROMA

logger = get_logger(__name__)


class CromaSize(str, Enum):
    """CROMA model size."""

    BASE = "base"
    LARGE = "large"


class CromaModality(str, Enum):
    """CROMA model configured input modalities."""

    BOTH = "both"
    SENTINEL1 = "SAR"
    SENTINEL2 = "optical"


PATCH_SIZE = 8
DEFAULT_IMAGE_RESOLUTION = 120
PRETRAINED_URLS: dict[CromaSize, str] = {
    CromaSize.BASE: "https://huggingface.co/antofuller/CROMA/resolve/main/CROMA_base.pt",
    CromaSize.LARGE: "https://huggingface.co/antofuller/CROMA/resolve/main/CROMA_large.pt",
}
MEAN_AND_STD_BY_BAND: dict[tuple[str, str], tuple[float, float]] = {
    ("sentinel1", "vv"): (0.15, 0.82),
    ("sentinel1", "vh"): (0.03, 0.15),
    ("sentinel2", "B01"): (1116, 1956),
    ("sentinel2", "B02"): (1189, 1859),
    ("sentinel2", "B03"): (1408, 1728),
    ("sentinel2", "B04"): (1513, 1741),
    ("sentinel2", "B05"): (1891, 1755),
    ("sentinel2", "B06"): (2484, 1622),
    ("sentinel2", "B07"): (2723, 1622),
    ("sentinel2", "B08"): (2755, 1612),
    ("sentinel2", "B8A"): (2886, 1611),
    ("sentinel2", "B09"): (3270, 2651),
    ("sentinel2", "B11"): (2563, 1442),
    ("sentinel2", "B12"): (1914, 1329),
}
MODALITY_BANDS = {
    "sentinel1": ["vv", "vh"],
    "sentinel2": [
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
        "B11",
        "B12",
    ],
}


class Croma(torch.nn.Module):
    """CROMA backbones.

    There are two model sizes, base and large.

    The model can be applied with just Sentinel-1, just Sentinel-2, or both. The input
    must be defined a priori by passing the corresponding CromaModality. Sentinel-1
    images should be passed under the "sentinel1" key while Sentinel-2 images should be
    passed under the "sentinel2" key. Only a single timestep can be provided.

    The band order for Sentinel-1 is: vv, vh.

    The band order for Sentinel-2 is: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09,
    B11, B12. It is trained on L1C images with B10 removed.

    See https://github.com/antofuller/CROMA for more details.
    """

    def __init__(
        self,
        size: CromaSize,
        modality: CromaModality,
        pretrained_path: str | None = None,
        image_resolution: int = DEFAULT_IMAGE_RESOLUTION,
    ) -> None:
        """Instantiate a new Croma instance.

        Args:
            size: the model size, either base or large.
            modality: the modalities to configure the model to accept.
            pretrained_path: the local path to the pretrained weights. Otherwise it is
                downloaded and cached in temp directory.
            image_resolution: the width and height of the input images.
        """
        super().__init__()
        self.size = size
        self.modality = modality
        self.image_resolution = image_resolution

        # Cache the CROMA weights to a deterministic path in temporary directory if the
        # path is not provided by the user.
        if pretrained_path is None:
            pretrained_url = PRETRAINED_URLS[self.size]
            local_fname = UPath(
                tempfile.gettempdir(), "rslearn_cache", "croma", f"{self.size.value}.pt"
            )
            if not local_fname.exists():
                logger.info(
                    "caching CROMA weights from %s to %s", pretrained_url, local_fname
                )
                local_fname.parent.mkdir(parents=True, exist_ok=True)
                with urllib.request.urlopen(pretrained_url) as response:
                    with open_atomic(local_fname, "wb") as f:
                        shutil.copyfileobj(response, f)
            else:
                logger.info("using cached CROMA weights at %s", local_fname)
            pretrained_path = local_fname.path

        self.model = PretrainedCROMA(
            pretrained_path=pretrained_path,
            size=size.value,
            modality=modality.value,
            image_resolution=image_resolution,
        )

    def forward(self, inputs: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Compute feature maps from the Croma backbone.

        Inputs:
            inputs: input dicts that must include either/both of "sentinel2" or
                "sentinel1" keys depending on the configured modality.
        """
        sentinel1: torch.Tensor | None = None
        sentinel2: torch.Tensor | None = None
        if self.modality in [CromaModality.BOTH, CromaModality.SENTINEL1]:
            sentinel1 = torch.stack([inp["sentinel1"] for inp in inputs], dim=0)
        if self.modality in [CromaModality.BOTH, CromaModality.SENTINEL2]:
            sentinel2 = torch.stack([inp["sentinel2"] for inp in inputs], dim=0)
        outputs = self.model(
            SAR_images=sentinel1,
            optical_images=sentinel2,
        )

        # Pick which encoding to use.
        # If modality is both, then there are three options, we could concatenate the
        # SAR and optical encodings but for now we just use the joint encodings.
        if self.modality == CromaModality.BOTH:
            features = outputs["joint_encodings"]
        elif self.modality == CromaModality.SENTINEL1:
            features = outputs["SAR_encodings"]
        elif self.modality == CromaModality.SENTINEL2:
            features = outputs["optical_encodings"]

        # Rearrange from patch embeddings to 2D feature map.
        num_patches_per_dim = self.image_resolution // PATCH_SIZE
        features = rearrange(
            features,
            "b (h w) d -> b d h w",
            h=num_patches_per_dim,
            w=num_patches_per_dim,
        )

        return [features]

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
        if self.size == CromaSize.BASE:
            depth = 768
        elif self.size == CromaSize.LARGE:
            depth = 1024
        else:
            raise ValueError(f"unknown CromaSize {self.size}")
        return [(PATCH_SIZE, depth)]


class CromaNormalize(Transform):
    """Normalize inputs using CROMA normalization.

    It will apply normalization to the "sentinel1" and "sentinel2" input keys (if set).
    """

    def __init__(self) -> None:
        """Initialize a new CromaNormalize."""
        super().__init__()

    def apply_image(self, image: torch.Tensor, modality: str) -> torch.Tensor:
        """Normalize the specified image with CROMA normalization.

        CROMA normalized based on batch statistics, but we may apply the model with
        small batches, so we instead use preset statistics corresponding to the dataset
        distribution.

        The normalized value is based on clipping to [mean-2*std, mean+2*std] and then
        linear rescaling to [0, 1].

        Args:
            image: the image to transform.
            modality: the modality of the image.
            mean: the mean to use for the normalization.
            std: the standard deviation to use for the normalization.
        """
        image = image.float()

        # Number of channels must be a multiple of the expected number of bands for
        # this modality. It can be a multiple since we accept stacked time series.
        band_names = MODALITY_BANDS[modality]
        if image.shape[0] % len(band_names) != 0:
            raise ValueError(
                f"image has {image.shape[0]} channels for modality {modality} which is not a multiple of expected number of bands {len(band_names)}"
            )

        normalized_bands = []
        for band_idx in range(image.shape[0]):
            band_name = band_names[band_idx % len(band_names)]
            mean, std = MEAN_AND_STD_BY_BAND[(modality, band_name)]

            orig = image[band_idx, :, :]
            min_value = mean - 2 * std
            max_value = mean + 2 * std

            normalized = (orig - min_value) / (max_value - min_value)
            normalized = torch.clip(normalized, 0, 1)
            normalized_bands.append(normalized)

        return torch.stack(normalized_bands, dim=0)

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
        for modality in MODALITY_BANDS.keys():
            if modality not in input_dict:
                continue
            input_dict[modality] = self.apply_image(input_dict[modality], modality)
        return input_dict, target_dict
