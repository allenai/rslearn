"""Adapt the OlmoEarth daily ERA5 encoder to rslearn's model interface."""

import json
import math
import os
from datetime import timedelta
from pathlib import Path

import torch
from torch import nn

from rslearn.models.component import FeatureExtractor, FeatureMaps, FeatureVector
from rslearn.models.era5_encoder import Era5DailyEncoder, Era5DailyEncoderConfig
from rslearn.train.model_context import ModelContext, RasterImage


class Era5SWTEncoder(FeatureExtractor):
    """Encode raw daily ERA5, then predict a vector or a constant spatial map.

    Input must be a native single-cell CTHW raster with one timestamp per day.
    Raw normalization happens here, before the encoder's SWT normalization.
    The nested ``encoder`` preserves the OlmoEarth backbone's state-dict keys;
    ``head`` is a new supervised readout and is initialized independently.
    """

    def __init__(
        self,
        encoder_config: Era5DailyEncoderConfig,
        raw_stats_path: str,
        mod_key: str = "era5_daily",
        d_output: int = 1,
        output_spatial_size: int | None = None,
    ) -> None:
        """Configure the backbone, preprocessing, and supervised readout.

        Args:
            encoder_config: Architecture matching the pretrained ERA5 encoder.
            raw_stats_path: JSON with raw_mean/raw_std in input band order.
                Environment variables are expanded. For SWT, this must match
                the raw normalization used to compute its wavelet statistics.
            mod_key: Input raster key in ModelContext.
            d_output: Number of output channels (one for DFMC regression).
            output_spatial_size: Repeat the output across this square label
                grid. If omitted, return a FeatureVector.
        """
        super().__init__()
        stats = json.loads(Path(os.path.expandvars(raw_stats_path)).read_text())
        mean, std = stats["raw_mean"], stats["raw_std"]
        if len(mean) != encoder_config.in_channels or len(std) != len(mean):
            raise ValueError("Raw ERA5 statistics must match in_channels")
        if not all(math.isfinite(x) for x in mean + std) or any(x <= 0 for x in std):
            raise ValueError("Raw statistics must be finite with positive deviations")
        if d_output < 1 or (
            output_spatial_size is not None and output_spatial_size < 1
        ):
            raise ValueError("Output channel count and spatial size must be positive")
        self.mod_key = mod_key
        self.output_spatial_size = output_spatial_size
        self.register_buffer("raw_mean", torch.tensor(mean).view(1, 1, -1))
        self.register_buffer("raw_std", torch.tensor(std).view(1, 1, -1))
        self.nodata_value = float(stats.get("nodata_value", -9999))
        band_order = stats.get("band_order", [])
        if len(band_order) != len(mean):
            raise ValueError("Raw statistics must include band_order for every channel")
        self.encoder = Era5DailyEncoder(encoder_config)
        embedding_size = encoder_config.embedding_size
        if encoder_config.pooling == "cls_mean_concat":
            embedding_size *= 2
        self.head = nn.Linear(embedding_size, d_output)

    def _prepare_inputs(
        self, context: ModelContext
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the v1.2 exact-sentinel mask and zero-impute normalized no-data.

        As in pretraining, fully missing inputs remain zero-filled; the SWT
        encoder handles whole-variable masking from the returned validity mask.
        """
        config = self.encoder.config
        values, dates = [], []
        for inputs in context.inputs:
            raster = inputs[self.mod_key]
            if not isinstance(raster, RasterImage):
                raise TypeError("ERA5 input must be a RasterImage")
            expected = (config.in_channels, config.max_sequence_length, 1, 1)
            if tuple(raster.image.shape) != expected:
                raise ValueError(f"Expected native ERA5 shape {expected}")
            if raster.timestamps is None or len(raster.timestamps) != expected[1]:
                raise ValueError("ERA5 requires one timestamp per daily timestep")
            starts = [start for start, _ in raster.timestamps]
            if any(b - a != timedelta(days=1) for a, b in zip(starts, starts[1:])):
                raise ValueError("ERA5 timestamps must be consecutive daily samples")
            values.append(raster.image[:, :, 0, 0].T)
            # Match the source encoder's [day-of-year (1-based), month0, year].
            dates.append([[d.timetuple().tm_yday, d.month - 1, d.year] for d in starts])
        raw = torch.stack(values).float()
        # Match Era5TaskDataset: only the exact sentinel is missing, in every
        # band. Tolerance fields in the statistics JSON describe how statistics
        # were estimated; they are not the pretraining dataloader's mask rule.
        valid = raw != self.nodata_value
        normalized = (
            torch.where(valid, raw, self.raw_mean) - self.raw_mean
        ) / self.raw_std
        timestamps = torch.tensor(dates, dtype=torch.long, device=raw.device)
        return normalized, timestamps, valid

    def forward(self, context: ModelContext) -> FeatureVector | FeatureMaps:
        """Return predictions in the task's normalized target space."""
        values, timestamps, valid = self._prepare_inputs(context)
        pooled = self.encoder(values, timestamps, valid_mask=valid)["pooled"]
        predictions = self.head(pooled)
        if self.output_spatial_size is None:
            return FeatureVector(predictions)
        return FeatureMaps(
            [
                predictions[:, :, None, None].expand(
                    -1, -1, self.output_spatial_size, self.output_spatial_size
                )
            ]
        )
