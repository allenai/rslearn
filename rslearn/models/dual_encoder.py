"""Dual encoder for late fusion of multiple modalities."""

import torch
from einops import rearrange

from rslearn.models.component import FeatureExtractor, FeatureMaps
from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.train.model_context import ModelContext


class Era5Encoder(torch.nn.Module):
    """Simple CNN encoder for ERA5 weather data."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 4,
        downsample_factor: int = 8,
    ):
        """Create a new Era5Encoder.

        Args:
            in_channels: number of ERA5 input channels.
            hidden_channels: number of channels in hidden layers.
            out_channels: number of output embedding channels.
            num_layers: number of convolutional layers.
            downsample_factor: spatial downsample factor (unused, kept for config).
        """
        super().__init__()
        self.downsample_factor = downsample_factor

        layers: list[torch.nn.Module] = []
        for i in range(num_layers):
            c_in = in_channels if i == 0 else hidden_channels
            c_out = out_channels if i == num_layers - 1 else hidden_channels
            layers.extend(
                [
                    torch.nn.Conv2d(c_in, c_out, 3, padding=1),
                    torch.nn.BatchNorm2d(c_out),
                    torch.nn.ReLU(inplace=True),
                ]
            )
        self.encoder = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        """Encode ERA5 data and resize to match the target feature map size.

        Args:
            x: ERA5 input tensor of shape (B, C, H, W).
            target_size: (H, W) of the target feature map to match.

        Returns:
            encoded features of shape (B, out_channels, target_H, target_W).
        """
        features = self.encoder(x)
        # Resize to match OlmoEarth feature map size
        features = torch.nn.functional.interpolate(
            features, size=target_size, mode="bilinear", align_corners=False
        )
        return features


class OlmoEarthWithEra5(FeatureExtractor):
    """Dual encoder: OlmoEarth for S1/S2 + separate encoder for ERA5.

    OlmoEarth processes the standard satellite modalities (sentinel2_l2a,
    sentinel1, etc.) while a separate CNN processes ERA5 weather data.
    The two sets of features are fused (concatenated or added) at the
    embedding level before being passed to downstream decoders.
    """

    def __init__(
        self,
        # OlmoEarth config
        olmo_patch_size: int = 8,
        olmo_model_id: str = "OLMOEARTH_V1_BASE",
        olmo_embedding_size: int = 768,
        # ERA5 encoder config
        era5_in_channels: int = 37,
        era5_hidden_channels: int = 128,
        era5_out_channels: int = 128,
        era5_num_layers: int = 4,
        era5_key: str = "era5",
        # Fusion options
        fusion_method: str = "concat",
    ):
        """Create a new OlmoEarthWithEra5.

        Args:
            olmo_patch_size: token spatial patch size for OlmoEarth.
            olmo_model_id: OlmoEarth model ID string (e.g. "OLMOEARTH_V1_BASE").
            olmo_embedding_size: embedding dimension of the OlmoEarth encoder.
            era5_in_channels: number of ERA5 input channels.
            era5_hidden_channels: hidden channels in the ERA5 CNN encoder.
            era5_out_channels: output channels from the ERA5 encoder.
            era5_num_layers: number of conv layers in the ERA5 encoder.
            era5_key: key in the input dict that holds ERA5 data.
            fusion_method: how to fuse features, one of "concat" or "add".
        """
        super().__init__()

        from olmoearth_pretrain.model_loader import ModelID

        self.olmo = OlmoEarth(
            patch_size=olmo_patch_size,
            model_id=getattr(ModelID, olmo_model_id),
            embedding_size=olmo_embedding_size,
        )

        self.era5_encoder = Era5Encoder(
            in_channels=era5_in_channels,
            hidden_channels=era5_hidden_channels,
            out_channels=era5_out_channels,
            num_layers=era5_num_layers,
        )

        self.era5_key = era5_key
        self.fusion_method = fusion_method
        self.olmo_embedding_size = olmo_embedding_size
        self.era5_out_channels = era5_out_channels
        self.patch_size = olmo_patch_size

    def forward(self, context: ModelContext) -> FeatureMaps:
        """Compute fused feature maps from OlmoEarth and ERA5 encoders.

        OlmoEarth processes S1/S2 modalities from context while the ERA5
        encoder processes the ERA5 data. Features are fused according to
        the configured fusion_method.

        Args:
            context: the model context containing all modality inputs.

        Returns:
            a FeatureMaps with one fused feature map.
        """
        # Get OlmoEarth features (processes S1/S2 modalities)
        olmo_features = self.olmo(context)  # Returns FeatureMaps
        olmo_feat = olmo_features.feature_maps[0]  # BCHW

        # Get ERA5 data and encode
        era5_data = torch.stack(
            [
                rearrange(inp[self.era5_key].image, "c t h w -> (c t) h w")
                for inp in context.inputs
            ],
            dim=0,
        )

        target_size = (olmo_feat.shape[2], olmo_feat.shape[3])
        era5_feat = self.era5_encoder(era5_data, target_size)  # BCHW

        # Fuse features
        if self.fusion_method == "concat":
            fused = torch.cat([olmo_feat, era5_feat], dim=1)
        elif self.fusion_method == "add":
            fused = olmo_feat + era5_feat
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return FeatureMaps([fused])

    def get_backbone_channels(self) -> list:
        """Returns the output channels of this dual encoder.

        Returns:
            list of (downsample_factor, depth) tuples.
        """
        if self.fusion_method == "concat":
            return [
                (self.patch_size, self.olmo_embedding_size + self.era5_out_channels)
            ]
        return [(self.patch_size, self.olmo_embedding_size)]
