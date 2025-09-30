import torch
from pathlib import Path
from enum import StrEnum
from torchvision.transforms import v2

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

class DinoV3(torch.nn.Module):
    
    modalities:list[str] = ['rgb', 's2', 'landsat']
    image_size:int = 256

    def _load_model(model_size: str, checkpoint_dir: str):
        if model_size == DinoV3Models.LARGE_SATELLITE:
            return torch.hub.load("facebookresearch/dinov3", 'dinov3_vitl16', weights=Path(checkpoint_dir) / 'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth')
        elif model_size == DinoV3Models.FULL_7B_SATELLITE:
            return torch.hub.load("facebookresearch/dinov3", 'dinov3_vit7b16', weights=Path(checkpoint_dir) / 'dinov3_vit7b16_pretrain_sat493m-a6675841.pth')
         elif model_size == DinoV3Models.BASE_WEB:
             return torch.hub.load("facebookresearch/dinov3", 'dinov3_vitb16', weights=Path(checkpoint_dir) / 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
         elif model_size == DinoV3Models.LARGE_WEB:
             return torch.hub.load("facebookresearch/dinov3", 'dinov3_vitl16', weights=Path(checkpoint_dir) / 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
         elif model_size == DinoV3Models.HUGE_PLUS_WEB:
             return torch.hub.load("facebookresearch/dinov3", 'dinov3_vith16plus', weights=Path(checkpoint_dir) / 'dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth')
         elif model_size == DinoV3Models.FULL_7B_WEB:
             return torch.hub.load("facebookresearch/dinov3", 'dinov3_vit7b16', weights=Path(checkpoint_dir) / 'dinov3_vit7b16_pretrain_lvd1689m-a955f4.pth')
        # Add these if you want
        # elif model_size == DinoV3Models.SMALL_WEB:
        #     return torch.hub.load("facebookresearch/dinov3", 'dinov3_vits16', weights=Path(checkpoint_dir) / 'not downloaded')
        # elif model_size == DinoV3Models.SMALL_PLUS_WEB:
        #     return torch.hub.load("facebookresearch/dinov3", 'dinov3_vits16plus', weights=Path(checkpoint_dir) / 'not downloaded')
        else:
            raise ValueError(f'Unspported size: {model_size}')
            
    def __init__(
        self,
        model_size: str = DinoV3Models.LARGE_SATELLITE,
        checkpoint_dir: str = '/weka/dfive-default/helios/models/dinov3/repo/dinov3',
        use_cls_token: bool = False,
    ):
        self.model_size = model_size
        self.checkpoint_dir = checkpoint_dir
        self.use_cls_token = use_cls_token
        self.model = _load_model(model_size, checkpoint_dir)

    def forward(self, inputs: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Forward pass for the dinov3 model.

        Args:
            inputs: input dicts that must include modalities as keys which are defined in the self.modalities list

        Returns:
            List[torch.Tensor]: Single-scale feature tensors from the encoder.
        """

        output_features = []
        for modality in self.modalities:
            if modality not in inputs[0]:
                continue
            cur = torch.stack([inp[modality] for inp in inputs], dim=0)  # (B, C, H, W)
            features = self.model(cur)
            output_features.append(features)
        avg_features = torch.stack(output_features, dim=0).mean(dim=0)
        if not self.use_cls_features
            batch_size, num_patches, _ = avg_features.shape
            height, width = int(num_patches**0.5), int(num_patches**0.5)
            avg_features = rearrange(avg_features, "b (h w) d -> b d h w", h=height, w=width)
        return [avg_features]

class DinoV3Normalize(Transform):
    """Normalize inputs using DinoV3 normalization.

    It will apply normalization to the modalities that are specified in the model configuration.
    """

    def __init__(self, satellite:bool = True) -> None:
        """Initialize a new DinoV3Normalize."""
        self.satellite = satellite
        if satellite:
            self.normalize = v2.Normalize(
                mean=(0.430, 0.411, 0.296),
                std=(0.213, 0.156, 0.143),
            )
        else:
            self.normalize = v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        super().__init__()

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
        for modality in DinoV3.modalities:
            input_dict[modality] = self.normalize(input_dict[modality])
        return input_dict, target_dict
