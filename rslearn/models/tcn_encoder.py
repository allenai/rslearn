"""Dual encoder for late fusion of multiple modalities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def prepare_ts_modality(
    context: ModelContext,
    mod_key: str,
    pad_value: float = 0.0,
    has_mask_channel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract, batch, and pre-process a time-series modality from model context.

    Handles variable-length sequences across batch items by padding shorter
    sequences to the maximum length in the batch.  Spatial dimensions are
    averaged so the output is always 3-D.

    The input is expected to be a proper ``[C, T, H, W]`` tensor where ``C``
    is the number of variables and ``T`` is the number of timesteps.  If the
    raw data stores all timesteps flattened into the channel dimension (e.g.
    ``[C*T, 1, H, W]`` with ``passthrough: true``), apply the
    :class:`~rslearn.train.transforms.unflatten_timesteps.UnflattenTimesteps`
    transform first to separate channels and timesteps.

    When ``has_mask_channel`` is True the first channel (index 0) of the raw
    CTHW tensor is treated as a binary validity mask (1 = valid, 0 = masked)
    produced by :class:`RandomTimeMasking` with ``append_mask_channel=True``.
    The mask channel is stripped from the data so that the returned *data*
    has the original number of features, and its information is merged into
    the returned *mask* tensor.

    Args:
        context: the model context containing all modality inputs.
        mod_key: key identifying the time-series modality in each input dict.
        pad_value: value used to fill padded timesteps (default 0).
        has_mask_channel: if True, expect a prepended binary mask channel and
            extract it.

    Returns:
        A tuple ``(data, mask)`` where *data* has shape ``[B, T, C]`` and
        *mask* is a bool tensor of shape ``[B, T]`` that is ``True`` for
        valid timesteps and ``False`` for masked / padded ones.  The mask
        unifies both batch-padding and augmentation masking.
    """
    raw_images = [inp[mod_key].image for inp in context.inputs]  # list of [C, T, H, W]

    # --- optionally extract and separate the mask channel ----
    aug_masks: list[torch.Tensor] | None = None
    if has_mask_channel:
        aug_masks = []
        stripped = []
        for img in raw_images:
            # Channel 0 is the binary mask: (1, T, H, W)
            mask_ch = img[0:1]  # [1, T, H, W]
            # Spatial-average then threshold → per-timestep bool [T]
            aug_masks.append(mask_ch.mean(dim=(0, 2, 3)) > 0.5)
            stripped.append(img[1:])  # remaining data channels
        raw_images = stripped

    ts_list = [rearrange(img, "c t h w -> (h w) t c") for img in raw_images]

    max_t = max(x.shape[1] for x in ts_list)
    padded = []
    lengths = []
    for x in ts_list:
        lengths.append(x.shape[1])
        if x.shape[1] < max_t:
            pad = torch.full(
                (x.shape[0], max_t - x.shape[1], x.shape[2]),
                pad_value,
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat([x, pad], dim=1)
        padded.append(x)

    data = torch.stack(padded, dim=0)  # [B, HW, T, C]

    # Average-pool spatial dimensions: [B, T, C]
    if data.dim() == 4:
        data = data.mean(dim=1)

    B, T, _ = data.shape

    # Build padding mask: True = real, False = padded.
    mask = torch.ones(B, T, dtype=torch.bool, device=data.device)
    for i, length in enumerate(lengths):
        mask[i, length:] = False

    # Merge augmentation mask (from RandomTimeMasking) into the padding mask.
    if aug_masks is not None:
        for i, am in enumerate(aug_masks):
            # am is [T_orig]; pad to max_t if needed.
            if am.shape[0] < T:
                am = F.pad(am.float(), (0, T - am.shape[0])).bool()
            elif am.shape[0] > T:
                am = am[:T]
            mask[i] &= am.to(mask.device)

    return data, mask


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
        has_mask_channel: bool = False,
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
            has_mask_channel: if True, expect a prepended binary mask channel
                (from :class:`RandomTimeMasking` with ``append_mask_channel=True``)
                and use it to ignore masked timesteps during temporal pooling.
        """
        super().__init__()

        self.mod_key = mod_key
        self.in_channels = in_channels
        self.output_spatial_size = output_spatial_size
        self.num_conv_layers = num_conv_layers
        self.has_mask_channel = has_mask_channel

        # Front-end: normalize input variables and project to base_dim
        self.input_norm = nn.LayerNorm(in_channels)
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
        # Extract and spatially pool TS data: [B, T, C]
        TS_data, mask = prepare_ts_modality(
            context,
            self.mod_key,
            has_mask_channel=self.has_mask_channel,
        )

        x = self.input_norm(TS_data)
        x = self.input_proj(x)

        # Zero out masked/padded timesteps so convolutions see neutral values.
        x = x * mask.unsqueeze(-1)  # [B, T, d_model]

        x = x.transpose(1, 2)  # [B, d_model, T]
        x = self.conv_layers(x)  # [B, width, T']

        # Downsample mask to match the conv output temporal dimension.
        T_out = x.shape[2]
        if T_out != mask.shape[1]:
            ds_mask = (
                F.adaptive_max_pool1d(mask.float().unsqueeze(1), T_out).squeeze(1) > 0.5
            )
        else:
            ds_mask = mask

        # Masked mean + max pooling over time.
        mask_expanded = ds_mask.unsqueeze(1)  # [B, 1, T']
        x_masked = x * mask_expanded  # zero out invalid positions
        valid_count = ds_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        x_mean = x_masked.sum(dim=2) / valid_count  # [B, width]
        x_max = x.masked_fill(~mask_expanded, float("-inf")).max(dim=2).values

        x = torch.cat([x_mean, x_max], dim=1)  # [B, 2*width]
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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Pool temporal dimension using attention.

        Args:
            x: input tensor of shape [B, d_model, T].
            mask: optional bool tensor of shape [B, T] where ``True`` marks
                valid timesteps and ``False`` marks masked/padded ones.
                Masked positions receive ``-inf`` attention scores so they
                cannot contribute to the pooled output.

        Returns:
            pooled tensor of shape [B, d_model].
        """
        # x: [B, d_model, T]
        x_t = x.transpose(1, 2)  # [B, T, d_model]
        scores = self.attention(x_t)  # [B, T, 1]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(scores, dim=1)  # [B, T, 1]
        pooled = (x_t * weights).sum(dim=1)  # [B, d_model]
        return pooled


class TCNEncoder(FeatureExtractor):
    """Temporal Convolutional Network encoder for time series data.

    This encoder processes time series into a compact embedding
    using dilated causal convolutions and multi-scale pooling.
    """

    def __init__(
        self,
        in_channels: int = 14,
        d_model: int = 128,
        d_output: int = 256,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
        dropout: float = 0.1,
        num_groups: int = 8,
        mod_key: str = "era5_daily",
        output_spatial_size: int | None = None,
        pooling_windows: list[int] | None = None,
        has_mask_channel: bool = False,
    ):
        """Create a new TCNEncoder.

        Args:
            in_channels: number of input variables per timestep.
            d_model: hidden dimension for the TCN backbone.
            d_output: output embedding dimension.
            kernel_size: convolutional kernel size.
            dilations: list of dilation factors for residual blocks.
                Default: [1, 2, 4, 8, 16, 32, 64, 128].
            dropout: dropout probability.
            num_groups: number of groups for GroupNorm.
            mod_key: key in the input dict that holds the time series data.
            output_spatial_size: if provided, upsample to a spatial grid of this size (e.g., 5 for 5x5).
                If None, outputs a FeatureVector. If set, outputs FeatureMaps with replicated embeddings.
            pooling_windows: list of pyramid levels where each entry is the
                number of bins at that level. Default: [1, 2, 4, 12] for
                whole-sequence, halves, quarters, and monthly bins.
            has_mask_channel: if True, expect a prepended binary mask channel
                (from :class:`RandomTimeMasking` with ``append_mask_channel=True``)
                and use it to ignore masked timesteps during attention pooling.
        """
        super().__init__()

        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 128]

        # Pooling_windows as pyramid levels: number of bins per level.
        # Example: [1, 2, 4] => pool over whole year, halves, and quarters.
        if pooling_windows is None:
            pooling_windows = [1, 2, 4, 12]

        self.mod_key = mod_key
        self.in_channels = in_channels
        self.output_spatial_size = output_spatial_size
        self.pooling_windows = pooling_windows
        self.has_mask_channel = has_mask_channel

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

        # Temporal pyramid attention pooling:
        # one pooling module per BIN across all pyramid levels
        self.num_pyramid_bins = sum(pooling_windows)
        self.pooling_modules = nn.ModuleList(
            [AttentionPooling(d_model) for _ in range(self.num_pyramid_bins)]
        )

        # Final MLP to produce output embedding
        mlp_input_dim = d_model * self.num_pyramid_bins
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, d_output),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_output, d_output),
        )

    def forward(self, context: ModelContext) -> FeatureVector | FeatureMaps:
        """Extract time series embedding via TCN and temporal pyramid pooling.

        Args:
            context: the model context containing all modality inputs.

        Returns:
            a FeatureVector with the embedding of shape [B, d_output].
        """
        # Extract and spatially pool TS data: [B, T, C]
        era5_data, mask = prepare_ts_modality(
            context,
            self.mod_key,
            has_mask_channel=self.has_mask_channel,
        )

        B, T, C = era5_data.shape

        max_bins = max(self.pooling_windows)
        if T < max_bins:
            raise ValueError(
                f"Sequence length T={T} is shorter than the largest pyramid "
                f"level n_bins={max_bins}. Each pyramid bin must contain at "
                f"least one timestep. Use shorter pooling_windows or longer "
                f"sequences."
            )

        x = self.input_norm(era5_data)  # [B, T, C]
        x = self.input_proj(x)  # [B, T, d_model]

        # Zero out masked/padded timesteps so convolutions see neutral values.
        x = x * mask.unsqueeze(-1)  # [B, T, d_model]

        x = x.transpose(1, 2)  # [B, d_model, T]

        for block in self.tcn_blocks:
            x = block(x)  # [B, d_model, T]

        # Temporal pyramid pooling (mask-aware)
        pooled_features = []
        pool_idx = 0
        for n_bins in self.pooling_windows:
            # Split into n_bins contiguous chunks (nearly equal lengths)
            chunks = torch.chunk(x, chunks=n_bins, dim=2)
            mask_chunks = torch.chunk(mask, chunks=n_bins, dim=1)

            for chunk, mask_chunk in zip(chunks, mask_chunks):
                pooled = self.pooling_modules[pool_idx](
                    chunk, mask_chunk
                )  # [B, d_model]
                pooled_features.append(pooled)
                pool_idx += 1

        combined = torch.cat(pooled_features, dim=1)  # [B, d_model * num_bins_total]
        x = self.mlp(combined)  # [B, d_output]

        # If output_spatial_size is specified, replicate across spatial dimensions
        if self.output_spatial_size is not None:
            # x: [B, output_dim] -> [B, output_dim, H, W]
            B = x.shape[0]
            x = x.view(B, -1, 1, 1)  # [B, output_dim, 1, 1]
            x = x.expand(B, -1, self.output_spatial_size, self.output_spatial_size)
            return FeatureMaps([x])
        return FeatureVector(feature_vector=x)
