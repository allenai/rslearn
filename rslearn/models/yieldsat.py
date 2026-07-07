"""Model components for YieldSAT preprocessed time-series tensors."""

from __future__ import annotations

from typing import Literal

import torch

from rslearn.models.component import FeatureExtractor, FeatureVector
from rslearn.train.model_context import ModelContext


class YieldSatMLP(FeatureExtractor):
    """Simple MLP encoder for YieldSAT ``sample(time_step, band)`` tensors."""

    def __init__(
        self,
        input_key: str = "sample",
        hidden_dims: list[int] | None = None,
        out_channels: int = 1,
        dropout: float = 0.1,
        temporal_mode: Literal["flatten", "mean"] = "flatten",
    ) -> None:
        super().__init__()
        self.input_key = input_key
        self.temporal_mode = temporal_mode

        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers: list[torch.nn.Module] = []
        prev_is_lazy = True
        for dim in hidden_dims:
            if prev_is_lazy:
                layers.append(torch.nn.LazyLinear(dim))
                prev_is_lazy = False
            else:
                layers.append(torch.nn.Linear(prev_dim, dim))
            layers.append(torch.nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            prev_dim = dim

        if hidden_dims:
            layers.append(torch.nn.Linear(hidden_dims[-1], out_channels))
        else:
            layers.append(torch.nn.LazyLinear(out_channels))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, context: ModelContext) -> FeatureVector:
        samples = torch.stack(
            [inp[self.input_key].to(dtype=torch.float32) for inp in context.inputs],
            dim=0,
        )

        if self.temporal_mode == "flatten":
            features = samples.flatten(start_dim=1)
        elif self.temporal_mode == "mean":
            features = samples.mean(dim=1)
        else:
            raise ValueError(f"unknown temporal_mode {self.temporal_mode}")

        return FeatureVector(self.net(features))
