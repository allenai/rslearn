"""DinoV3 model.

This code loads the DINOv3 model. You must obtain the model separately from Meta to use
it. See https://github.com/facebookresearch/dinov3 for applicable license and copyright
information.
"""

from enum import StrEnum
from pathlib import Path
from typing import Any

import torch
import torchvision
from einops import rearrange

from rslearn.train.transforms.normalize import Normalize
from rslearn.train.transforms.transform import Transform


class DinoV3Models(StrEnum):
    """Names for different DinoV3 images on torch hub."""

    SMALL_WEB = "dinov3_vits16"
    SMALL_PLUS_WEB = "dinov3_vits16plus"
    BASE_WEB = "dinov3_vitb16"
    LARGE_WEB = "dinov3_vitl16"
    HUGE_PLUS_WEB = "dinov3_vith16plus"
    FULL_7B_WEB = "dinov3_vit7b16"
    LARGE_SATELLITE = "dinov3_vitl16_sat"
    FULL_7B_SATELLITE = "dinov3_vit7b16_sat"


DINOV3_PTHS: dict[str, str] = {
    DinoV3Models.LARGE_SATELLITE: "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
    DinoV3Models.FULL_7B_SATELLITE: "dinov3_vit7b16_pretrain_sat493m-a6675841.pth",
    DinoV3Models.BASE_WEB: "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    DinoV3Models.LARGE_WEB: "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    DinoV3Models.HUGE_PLUS_WEB: "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    DinoV3Models.FULL_7B_WEB: "dinov3_vit7b16_pretrain_lvd1689m-a955f4.pth",
}


class DinoV3(torch.nn.Module):
    """DinoV3 Backbones.

    Must have the pretrained weights downloaded in checkpoint_dir for them to be loaded.
    See https://github.com/facebookresearch/dinov3?tab=readme-ov-file#pretrained-models

    Only takes RGB as input. Expects normalized data (use the below normalizer).

    Uses patch size 16. The input is resized to 256x256; when applying DinoV3 on
    segmentation or detection tasks with inputs larger than 256x256, it may be best to
    train and predict on 256x256 crops (using SplitConfig.patch_size argument).
    """

    image_size: int = 256
    patch_size: int = 16
    output_dim: int = 1024

    def _load_model(self, size: str, checkpoint_dir: str | None) -> torch.nn.Module:
        model_name = size.replace("_sat", "")
        if checkpoint_dir is not None:
            weights = str(Path(checkpoint_dir) / DINOV3_PTHS[size])
            return torch.hub.load(
                "facebookresearch/dinov3",
                model_name,
                weights=weights,
            )  # nosec
        return torch.hub.load("facebookresearch/dinov3", model_name, pretrained=False)  # nosec

    def __init__(
        self,
        checkpoint_dir: str | None,
        size: str = DinoV3Models.LARGE_SATELLITE,
        use_cls_token: bool = False,
        do_resizing: bool = True,
    ) -> None:
        """Instantiate a new DinoV3 instance.

        Args:
            checkpoint_dir: the local path to the pretrained weight dir. If None, we load the architecture
                only (randomly initialized).
            size: the model size, see class for various models.
            use_cls_token: use pooled class token (for classification), otherwise returns spatial feature map.
            do_resizing: whether to resize inputs to 256x256. Default true.
        """
        super().__init__()
        self.size = size
        self.checkpoint_dir = checkpoint_dir
        self.use_cls_token = use_cls_token
        self.do_resizing = do_resizing
        self.model = self._load_model(size, checkpoint_dir)

    def forward(self, inputs: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Forward pass for the dinov3 model.

        Args:
            inputs: input dicts that must include "image" key.

        Returns:
            List[torch.Tensor]: Single-scale feature tensors from the encoder.
        """
        cur = torch.stack([inp["image"] for inp in inputs], dim=0)  # (B, C, H, W)

        if self.do_resizing and (
            cur.shape[2] != self.image_size or cur.shape[3] != self.image_size
        ):
            cur = torchvision.transforms.functional.resize(
                cur,
                [self.image_size, self.image_size],
            )

        if self.use_cls_token:
            features = self.model(cur)
        else:
            features = self.model.forward_features(cur)["x_norm_patchtokens"]
            batch_size, num_patches, _ = features.shape
            height, width = int(num_patches**0.5), int(num_patches**0.5)
            features = rearrange(features, "b (h w) d -> b d h w", h=height, w=width)

        return [features]

    def get_backbone_channels(self) -> list:
        """Returns the output channels of this model when used as a backbone.

        The output channels is a list of (downsample_factor, depth) that corresponds
        to the feature maps that the backbone returns. For example, an element [2, 32]
        indicates that the corresponding feature map is 1/2 the input resolution and
        has 32 channels.
        """
        return [(self.patch_size, self.output_dim)]


class DinoV3Normalize(Transform):
    """Normalize inputs using DinoV3 normalization.

    Normalize "image" key in input according to Dino statistics from pretraining. Satellite pretraining has slightly different normalizing than the base image model so set 'satellite' depending on what pretrained model you are using.

    Input "image" should be RGB-like image between 0-255.
    """

    def __init__(self, satellite: bool = True):
        """Initialize a new DinoV3Normalize."""
        super().__init__()
        self.satellite = satellite
        if satellite:
            mean = [0.430, 0.411, 0.296]
            std = [0.213, 0.156, 0.143]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        self.normalize = Normalize(
            [value * 255 for value in mean],
            [value * 255 for value in std],
            num_bands=3,
        )

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Normalize the specified image with DinoV3 normalization.

        Args:
            input_dict: the input dictionary.
            target_dict: the target dictionary.

        Returns:
            normalized (input_dicts, target_dicts) tuple
        """
        return self.normalize(input_dict, target_dict)
