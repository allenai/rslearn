"""Strided CNN encoder for aligned single-stack rasters (e.g., DEM / SRTM)."""

from __future__ import annotations

import math

import torch

from rslearn.train.model_context import ModelContext

from .component import FeatureExtractor, FeatureMaps


class RasterStrideEncoder(FeatureExtractor):
    """A stack of stride-2 convolutions producing one ``FeatureMaps`` tensor.

    Use ``downsample_factor`` equal to the primary backbone's token stride (e.g.
    OlmoEarth ``patch_size``) so multi-path late fusion sees the same grid size.
    """

    def __init__(
        self,
        input_key: str,
        in_channels: int = 1,
        hidden_channels: int = 32,
        out_channels: int = 128,
        downsample_factor: int = 4,
    ) -> None:
        super().__init__()
        if downsample_factor < 2:
            raise ValueError("downsample_factor must be >= 2")
        if downsample_factor & (downsample_factor - 1) != 0:
            raise ValueError("downsample_factor must be a power of 2")

        self.input_key = input_key
        self.downsample_factor = downsample_factor
        num_layers = int(math.log2(downsample_factor))

        layers: list[torch.nn.Module] = []
        c_in = in_channels
        for i in range(num_layers):
            c_out = out_channels if i == num_layers - 1 else hidden_channels
            layers.extend(
                [
                    torch.nn.Conv2d(
                        c_in,
                        c_out,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    torch.nn.BatchNorm2d(c_out),
                    torch.nn.ReLU(inplace=True),
                ]
            )
            c_in = c_out
        self.stem = torch.nn.Sequential(*layers)
        self._out_channels = out_channels

    def forward(self, context: ModelContext) -> FeatureMaps:
        """Encode ``input_key`` rasters to a single downsampled feature map."""
        x = torch.stack(
            [
                inp[self.input_key].single_ts_to_chw_tensor()
                for inp in context.inputs
            ],
            dim=0,
        )
        return FeatureMaps([self.stem(x.float())])

    def get_backbone_channels(self) -> list[tuple[int, int]]:
        """Return ``(downsample_factor, channels)`` for the output map."""
        return [(self.downsample_factor, self._out_channels)]
