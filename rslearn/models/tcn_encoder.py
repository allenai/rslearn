"""Dual encoder for late fusion of multiple modalities."""

import torch
import torch.nn as nn
from einops import rearrange

from rslearn.models.component import (
    FeatureExtractor,
    FeatureMaps,
    FeatureVector,
)
from rslearn.train.model_context import ModelContext

MODALITY_NAMES = [
    "era5_daily",
]


def prepare_ts_modality(context: ModelContext, mod_key: str) -> torch.Tensor:
    """Extract and batch a time-series modality from model context.

    Handles variable-length sequences across batch items by zero-padding
    shorter sequences to the maximum length in the batch.

    Args:
        context: the model context containing all modality inputs.
        mod_key: key identifying the time-series modality in each input dict.

    Returns:
        a tensor of shape [B, HW, T, C] where T is the max sequence length
        in the batch and shorter sequences are zero-padded.
    """
    ts_list = [
        rearrange(inp[mod_key].image, "c t h w -> (h w) t c") for inp in context.inputs
    ]

    max_t = max(x.shape[1] for x in ts_list)
    padded = []
    for x in ts_list:
        if x.shape[1] < max_t:
            pad = torch.zeros(
                x.shape[0],
                max_t - x.shape[1],
                x.shape[2],
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat([x, pad], dim=1)
        padded.append(x)

    return torch.stack(padded, dim=0)  # [B, HW, T, C]


class SimpleTCNEncoder(FeatureExtractor):
    """Simple baseline CNN encoder for Time Series data.

    Simple X-layer 1D CNN with temporal downsampling after each conv layer.
    Can output either a FeatureVector (for classification) or FeatureMaps (for segmentation)
    by replicating the embedding across a spatial grid.
    """

    def __init__(
        self,
        num_conv_layers: int = 3,
        in_channels: int = 13,
        base_dim: int = 32,
        width_mult: int = 2,
        mlp_ratio: int = 2,
        output_dim: int = 256,
        start_kernel_size: int = 7,
        dropout: float = 0.1,
        mod_key: str = "era5_daily",
        output_spatial_size: int | None = None,
    ):
        """Create a new SimpleTCNEncoder.

        Args:
            num_conv_layers: number of convolutional layer groups.
            in_channels: number of input temporal variables
            base_dim: initial hidden dimension
            width_mult: width multiplier after each downsample
            mlp_ratio: expansion ratio for the final MLP
            output_dim: output embedding dimension
            start_kernel_size: initial convolutional kernel size (decreases by 2 each layer)
            dropout: dropout probability.
            mod_key: key in the input dict that holds the time series data
            output_spatial_size: if provided, upsample to a spatial grid of this size (e.g., 5 for 5x5).
                If None, outputs a FeatureVector. If set, outputs FeatureMaps with replicated embeddings.
        """
        super().__init__()

        self.mod_key = mod_key
        self.output_spatial_size = output_spatial_size
        self._warned_spatial = False

        # Front-end linear projection
        self.input_proj = nn.Linear(in_channels, base_dim)

        def conv1D_block(
            dim: int, kernel_size: int, dropout: float, batch_norm: bool = True
        ) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size, stride=1, padding="same"),
                nn.BatchNorm1d(dim) if batch_norm else nn.Identity(),
                nn.GELU(),
                nn.Dropout1d(dropout),  # channel dropout
            )

        def downsample_block(dim_in: int, dim_out: int, k_size: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(dim_in, dim_out, k_size, stride=2, padding=k_size // 2),
                nn.BatchNorm1d(dim_out),
                nn.GELU(),
            )

        # Conv layers
        conv_layers = []
        width = base_dim
        for layer_ix in range(num_conv_layers):
            k_size = max(3, start_kernel_size - 2 * layer_ix)
            conv_layers.append(
                nn.Sequential(
                    conv1D_block(width, k_size, dropout),
                    conv1D_block(width, k_size, dropout),
                    downsample_block(width, width_mult * width, k_size),
                )
            )
            width = width * width_mult
        self.conv_layers = nn.Sequential(*conv_layers)

        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(2 * width, 2 * width * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width * mlp_ratio, output_dim),
        )

    def forward(self, context: ModelContext) -> FeatureVector | FeatureMaps:
        """Extract time series embedding.

        Args:
            context: the model context containing all modality inputs.

        Returns:
            If output_spatial_size is None: a FeatureVector with shape [B, output_dim].
            If output_spatial_size is set: a FeatureMaps with shape [B, output_dim, H, W]
                where the embedding is replicated across all spatial locations.
        """
        # Extract TS data from context (pads variable-length sequences)
        TS_data = prepare_ts_modality(context, self.mod_key)  # [B, HW, T, C]

        # Average-pool spatial dimensions: [B, T, C]
        if TS_data.dim() == 4:
            if TS_data.shape[1] > 1 and not self._warned_spatial:
                import warnings

                warnings.warn(
                    f"SimpleTCNEncoder: spatial extent {TS_data.shape[1]} > 1, "
                    f"averaging over spatial dimensions. Consider loading the "
                    f"input at coarser resolution to avoid this.",
                    stacklevel=2,
                )
                self._warned_spatial = True
            TS_data = TS_data.mean(dim=1)

        x = self.input_proj(TS_data)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = torch.cat([x.mean(dim=2), x.max(dim=2).values], dim=1)
        x = self.mlp(x)  # [B, output_dim]

        # If output_spatial_size is specified, replicate across spatial dimensions
        if self.output_spatial_size is not None:
            # x: [B, output_dim] -> [B, output_dim, H, W]
            B = x.shape[0]
            x = x.view(B, -1, 1, 1)  # [B, output_dim, 1, 1]
            x = x.expand(B, -1, self.output_spatial_size, self.output_spatial_size)
            return FeatureMaps([x])

        return FeatureVector(feature_vector=x)


class TCNResidualBlock(nn.Module):
    """Residual block with dilated causal convolutions for TCN."""

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        num_groups: int = 8,
    ):
        """Create a TCN residual block.

        Args:
            d_model: number of channels.
            kernel_size: convolutional kernel size.
            dilation: dilation factor for the convolution.
            dropout: dropout probability.
            num_groups: number of groups for GroupNorm.
        """
        super().__init__()
        # Calculate padding for causal convolution
        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(num_groups, d_model)
        self.activation1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(num_groups, d_model)
        self.activation2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.norm_out = nn.GroupNorm(num_groups, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input tensor of shape [B, d_model, T].

        Returns:
            output tensor of shape [B, d_model, T].
        """
        residual = x

        # First conv layer
        out = self.conv1(x)
        # Remove extra padding to maintain sequence length (causal)
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        out = self.norm1(out)
        out = self.activation1(out)
        out = self.dropout1(out)

        # Second conv layer
        out = self.conv2(out)
        # Remove extra padding to maintain sequence length (causal)
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        out = self.norm2(out)
        out = self.activation2(out)
        out = self.dropout2(out)

        # Residual connection
        out = out + residual
        out = self.norm_out(out)

        return out


class AttentionPooling(nn.Module):
    """Attention-based pooling over temporal dimension."""

    def __init__(self, d_model: int):
        """Create attention pooling module.

        Args:
            d_model: number of channels.
        """
        super().__init__()
        self.attention = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool temporal dimension using attention.

        Args:
            x: input tensor of shape [B, d_model, T].

        Returns:
            pooled tensor of shape [B, d_model].
        """
        # x: [B, d_model, T]
        x_t = x.transpose(1, 2)  # [B, T, d_model]
        scores = self.attention(x_t)  # [B, T, 1]
        weights = torch.softmax(scores, dim=1)  # [B, T, 1]
        pooled = (x_t * weights).sum(dim=1)  # [B, d_model]
        return pooled


class TCNEncoder(FeatureExtractor):
    """Temporal Convolutional Network encoder for ERA5 weather data.

    This encoder processes daily weather sequences (e.g., 365 days × 12 variables)
    into a compact weather embedding using dilated causal convolutions and
    multi-scale pooling.
    """

    def __init__(
        self,
        in_channels: int = 12,
        d_model: int = 128,
        d_output: int = 256,
        seq_length: int = 365,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
        dropout: float = 0.1,
        num_groups: int = 8,
        era5_key: str = "era5_daily",
        pooling_windows: list[int] | None = None,
    ):
        """Create a new TCNEncoder.

        Args:
            in_channels: number of input weather variables (e.g., 12).
            d_model: hidden dimension for the TCN backbone (e.g., 128).
            d_output: output embedding dimension (e.g., 256).
            seq_length: expected sequence length in days (e.g., 365).
            kernel_size: convolutional kernel size (usually 3).
            dilations: list of dilation factors for residual blocks.
                Default: [1, 2, 4, 8, 16, 32, 64, 128].
            dropout: dropout probability.
            num_groups: number of groups for GroupNorm.
            era5_key: key in the input dict that holds ERA5 data.
            pooling_windows: list of window sizes for multi-scale pooling.
                Default: [30, 120, 365] for recent, mid-term, and full-year.
        """
        super().__init__()

        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 128]

        if pooling_windows is None:
            pooling_windows = [30, 120, 365]

        self.era5_key = era5_key
        self.seq_length = seq_length
        self.pooling_windows = pooling_windows

        # Front-end: normalize input variables and project to d_model
        self.input_norm = nn.LayerNorm(in_channels)
        self.input_proj = nn.Linear(in_channels, d_model)

        # TCN backbone: stack of dilated residual blocks
        self.tcn_blocks = nn.ModuleList(
            [
                TCNResidualBlock(
                    d_model=d_model,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                    num_groups=num_groups,
                )
                for d in dilations
            ]
        )

        # Multi-scale attention pooling
        self.pooling_modules = nn.ModuleList(
            [AttentionPooling(d_model) for _ in pooling_windows]
        )

        # Final MLP to produce output embedding
        # Input is concatenation of all pooled representations
        mlp_input_dim = d_model * len(pooling_windows)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, d_output),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_output, d_output),
        )

    def forward(self, context: ModelContext) -> FeatureVector:
        """Extract weather embedding from ERA5 daily data.

        Args:
            context: the model context containing all modality inputs.

        Returns:
            a FeatureVector with the weather embedding of shape [B, d_output].
        """
        # Extract ERA5 data from context (pads variable-length sequences)
        # Expected shape after: [B, HW, T, C] where T is days, C is variables
        era5_data = prepare_ts_modality(context, self.era5_key)

        # If spatial dimensions exist, we need to handle them
        # Assuming H=W=1 for daily ERA5 aggregated data, so shape is [B, 1, T, C]
        # Squeeze spatial dimensions: [B, T, C]
        if era5_data.dim() == 4:
            era5_data = era5_data.squeeze(1)

        B, T, C = era5_data.shape

        # Normalize input variables per timestep
        x = self.input_norm(era5_data)  # [B, T, C]

        # Project to d_model
        x = self.input_proj(x)  # [B, T, d_model]

        # Transpose for conv1d: [B, d_model, T]
        x = x.transpose(1, 2)

        # Apply TCN blocks
        for block in self.tcn_blocks:
            x = block(x)  # [B, d_model, T]

        # Multi-scale pooling
        pooled_features = []
        for window_size, pooling_module in zip(
            self.pooling_windows, self.pooling_modules
        ):
            # Extract last window_size timesteps
            if window_size >= T:
                x_window = x
            else:
                x_window = x[:, :, -window_size:]

            # Apply attention pooling
            pooled = pooling_module(x_window)  # [B, d_model]
            pooled_features.append(pooled)

        # Concatenate all pooled features
        combined = torch.cat(pooled_features, dim=1)  # [B, d_model * num_windows]

        # Final MLP to produce weather embedding
        z_wx = self.mlp(combined)  # [B, d_output]

        return FeatureVector(feature_vector=z_wx)


# Commented out - incomplete implementation that references undefined Era5Encoder
# TODO: Implement a proper dual encoder if needed, or use composition pattern
#
# class OlmoEarthWithEra5(FeatureExtractor):
#     # I think this would be better as a Concat(FeatureExtractor) which takes a list[list[torch.nn.Module]] (where the lists should specify
#     # #different encoder paths, each one is FeatureExtractor followed by IntermediateComponents), and it applies each one of them and expects
#     # #output across all of them to either be FeatureMaps or FeatureVector and it just concatenates them all. This way it doesn't need to
#     # be specific to one use case.
#     """Dual encoder: OlmoEarth for S1/S2 + separate encoder for ERA5.
#
#     OlmoEarth processes the standard satellite modalities (sentinel2_l2a,
#     sentinel1, etc.) while a separate CNN processes ERA5 weather data.
#     The two sets of features are fused (concatenated or added) at the
#     embedding level before being passed to downstream decoders.
#     """
#
#     def __init__(
#         self,
#         # OlmoEarth config
#         olmo_patch_size: int = 8,
#         olmo_model_id: str = "OLMOEARTH_V1_BASE",
#         olmo_embedding_size: int = 768,
#         # ERA5 encoder config
#         era5_in_channels: int = 37,
#         era5_hidden_channels: int = 128,
#         era5_out_channels: int = 128,
#         era5_num_layers: int = 4,
#         era5_key: str = "era5",
#         # Fusion options
#         fusion_method: str = "concat",
#     ):
#         """Create a new OlmoEarthWithEra5.
#
#         Args:
#             olmo_patch_size: token spatial patch size for OlmoEarth.
#             olmo_model_id: OlmoEarth model ID string (e.g. "OLMOEARTH_V1_BASE").
#             olmo_embedding_size: embedding dimension of the OlmoEarth encoder.
#             era5_in_channels: number of ERA5 input channels.
#             era5_hidden_channels: hidden channels in the ERA5 CNN encoder.
#             era5_out_channels: output channels from the ERA5 encoder.
#             era5_num_layers: number of conv layers in the ERA5 encoder.
#             era5_key: key in the input dict that holds ERA5 data.
#             fusion_method: how to fuse features, one of "concat" or "add".
#         """
#         super().__init__()
#
#         from olmoearth_pretrain.model_loader import ModelID
#
#         self.olmo = OlmoEarth(
#             patch_size=olmo_patch_size,
#             model_id=getattr(ModelID, olmo_model_id),
#             embedding_size=olmo_embedding_size,
#         )
#
#         self.era5_encoder = Era5Encoder(
#             in_channels=era5_in_channels,
#             hidden_channels=era5_hidden_channels,
#             out_channels=era5_out_channels,
#             num_layers=era5_num_layers,
#         )
#
#         self.era5_key = era5_key
#         self.fusion_method = fusion_method
#         self.olmo_embedding_size = olmo_embedding_size
#         self.era5_out_channels = era5_out_channels
#         self.patch_size = olmo_patch_size
#
#     def forward(self, context: ModelContext) -> FeatureMaps:
#         """TBD.
#
#         Args:
#             context: the model context containing all modality inputs.
#
#         Returns:
#             a FeatureMaps with one fused feature map.
#         """
#         # Get OlmoEarth features (processes S1/S2 modalities)
#         olmo_features = self.olmo(context)  # Returns FeatureMaps
#         olmo_feat = olmo_features.feature_maps[0]  # BCHW
#
#         # Get ERA5 data and encode
#         era5_data = torch.stack(
#             [
#                 rearrange(inp[self.era5_key].image, "c t h w -> (c t) h w")
#                 for inp in context.inputs
#             ],
#             dim=0,
#         )
#
#         target_size = (olmo_feat.shape[2], olmo_feat.shape[3])
#         era5_feat = self.era5_encoder(era5_data, target_size)  # BCHW
#
#         # Fuse features
#         if self.fusion_method == "concat":
#             fused = torch.cat([olmo_feat, era5_feat], dim=1)
#         elif self.fusion_method == "add":
#             fused = olmo_feat + era5_feat
#         else:
#             raise ValueError(f"Unknown fusion method: {self.fusion_method}")
#
#         return FeatureMaps([fused])
#
#     def get_backbone_channels(self) -> list:
#         """Returns the output channels of this dual encoder.
#
#         Returns:
#             list of (downsample_factor, depth) tuples.
#         """
#         if self.fusion_method == "concat":
#             return [
#                 (self.patch_size, self.olmo_embedding_size + self.era5_out_channels)
#             ]
#         return [(self.patch_size, self.olmo_embedding_size)]
