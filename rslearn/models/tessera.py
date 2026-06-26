"""Tessera v1.1 model wrapper.

The architecture and normalization constants are adapted from
https://github.com/ucam-eo/tessera/tree/master/tessera_infer_QAT, which is
released under the MIT license.
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from rslearn.models.component import FeatureExtractor, FeatureMaps
from rslearn.train.model_context import ModelContext, RasterImage
from rslearn.train.transforms.transform import Transform

TESSERA_S2_BANDS = [
    "B04",
    "B02",
    "B03",
    "B08",
    "B8A",
    "B05",
    "B06",
    "B07",
    "B11",
    "B12",
]
TESSERA_S1_BANDS = ["vv", "vh"]

DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "fusion_method": "concat",
    "latent_dim": 192,
    "representation_dim": 192,
    "s2_num_heads": 4,
    "s2_num_layers": 4,
    "s2_dim_feedforward": 2048,
    "s1_num_heads": 4,
    "s1_num_layers": 4,
    "s1_dim_feedforward": 2048,
    "split_s1_modalities": False,
}

NORM_STATS: dict[str, dict[str, list[float]]] = {
    "mpc": {
        "s2_mean": [
            2683.4553,
            2223.3630,
            2432.0950,
            3633.1970,
            3602.1755,
            3006.4324,
            3400.2710,
            3515.6392,
            2456.9163,
            1983.8783,
        ],
        "s2_std": [
            2739.5217,
            2846.2993,
            2690.8250,
            2290.0439,
            2088.8970,
            2673.1106,
            2381.4521,
            2229.5225,
            1601.0942,
            1495.3545,
        ],
        "s1a_mean": [5588.3291, 3025.6270],
        "s1a_std": [1713.4646, 1693.0471],
        "s1d_mean": [5552.9683, 2955.0520],
        "s1d_std": [1685.5857, 1677.6414],
    },
    "aws": {
        "s2_mean": [
            2793.6589,
            2356.7776,
            2551.0496,
            3741.9229,
            3713.7844,
            3120.1997,
            3516.3342,
            3637.0342,
            2501.0283,
            2038.1504,
        ],
        "s2_std": [
            2810.0093,
            2933.8835,
            2755.6360,
            2344.5027,
            2145.7986,
            2743.9019,
            2438.8601,
            2286.5977,
            1680.7367,
            1585.5529,
        ],
        "s1a_mean": [5697.0859, 2838.6687],
        "s1a_std": [1671.3737, 1789.4116],
        "s1d_mean": [5759.1367, 2873.2854],
        "s1d_std": [1583.2858, 1747.8390],
    },
}


class TesseraDataSource(StrEnum):
    """Tessera v1.1 preprocessing/checkpoint source."""

    MPC = "mpc"
    AWS = "aws"


class TesseraNormalize(Transform):
    """Normalize Tessera inputs into the values expected by v1.1 checkpoints.

    Sentinel-2 inputs are expected to be raw DN values in Tessera band order.
    Sentinel-1 inputs are expected to be standard power dB values
    (``10 * log10(linear)``); use ``Sentinel1ToDecibels`` first for linear RTC
    sources.

    All inputs are assumed to be complete (no nodata or empty observations);
    every timestep is standardized and passed through to the model.
    """

    def __init__(
        self,
        data_source: TesseraDataSource | str = TesseraDataSource.MPC,
        s2_selector: str = "s2",
        s1_ascending_selector: str = "s1_ascending",
        s1_descending_selector: str = "s1_descending",
        skip_missing: bool = False,
    ) -> None:
        """Initialize the transform.

        Args:
            data_source: Normalization statistics family, either "mpc" or "aws".
            s2_selector: Selector for the Sentinel-2 ``RasterImage``.
            s1_ascending_selector: Selector for ascending Sentinel-1 ``RasterImage``.
            s1_descending_selector: Selector for descending Sentinel-1 ``RasterImage``.
            skip_missing: If True, skip selectors absent from the input/target dicts.
        """
        super().__init__(skip_missing=skip_missing)
        self.data_source = TesseraDataSource(data_source)
        self.s2_selector = s2_selector
        self.s1_ascending_selector = s1_ascending_selector
        self.s1_descending_selector = s1_descending_selector

        stats = NORM_STATS[self.data_source.value]
        self.register_buffer(
            "s2_mean", torch.tensor(stats["s2_mean"], dtype=torch.float32)
        )
        self.register_buffer(
            "s2_std", torch.tensor(stats["s2_std"], dtype=torch.float32)
        )
        self.register_buffer(
            "s1a_mean", torch.tensor(stats["s1a_mean"], dtype=torch.float32)
        )
        self.register_buffer(
            "s1a_std", torch.tensor(stats["s1a_std"], dtype=torch.float32)
        )
        self.register_buffer(
            "s1d_mean", torch.tensor(stats["s1d_mean"], dtype=torch.float32)
        )
        self.register_buffer(
            "s1d_std", torch.tensor(stats["s1d_std"], dtype=torch.float32)
        )

    def _normalize_s2(self, image: RasterImage) -> RasterImage:
        """Standardize a Sentinel-2 CTHW image with the checkpoint's stats."""
        if image.image.shape[0] != len(TESSERA_S2_BANDS):
            raise ValueError(
                "Tessera s2 input must have "
                f"{len(TESSERA_S2_BANDS)} bands, got {image.image.shape[0]}"
            )
        values = image.image.float()
        mean = self.s2_mean.to(device=values.device, dtype=values.dtype)[
            :, None, None, None
        ]
        std = self.s2_std.to(device=values.device, dtype=values.dtype)[
            :, None, None, None
        ]
        image.image = (values - mean) / (std + 1e-9)
        return image

    @staticmethod
    def _db_to_tessera_scaled(values: torch.Tensor) -> torch.Tensor:
        """Convert standard power dB to Tessera's stored Sentinel-1 int16 scale.

        Tessera's preprocessing stores Sentinel-1 as int16 via
        ``clip((20 * log10(x) + 50) * 200, 0, 32767)``, where ``x`` is the linear
        RTC value, ``+50`` offsets away from negatives and ``* 200`` preserves
        precision in int16. See ``amplitude_to_db`` in
        ``tessera_preprocessing/s1_fast_processor.py`` of the upstream
        https://github.com/ucam-eo/tessera repo. The Tessera v1.1 NORM_STATS are
        computed in this stored space, so inputs must match it.

        Our input is standard power dB, ``10 * log10(x)``, so we multiply by 2 to
        recover Tessera's ``20 * log10(x)`` before applying the offset and scale.
        """
        return torch.clamp((2 * values + 50) * 200, min=0, max=32767)

    def _normalize_s1(
        self, image: RasterImage, mean: torch.Tensor, std: torch.Tensor
    ) -> RasterImage:
        """Scale a Sentinel-1 dB CTHW image and standardize with the stats."""
        if image.image.shape[0] != len(TESSERA_S1_BANDS):
            raise ValueError(
                "Tessera Sentinel-1 inputs must have "
                f"{len(TESSERA_S1_BANDS)} bands, got {image.image.shape[0]}"
            )
        values = image.image.float()
        scaled = self._db_to_tessera_scaled(values)
        mean = mean.to(device=scaled.device, dtype=scaled.dtype)[:, None, None, None]
        std = std.to(device=scaled.device, dtype=scaled.dtype)[:, None, None, None]
        image.image = (scaled - mean) / (std + 1e-9)
        return image

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply Tessera normalization to configured inputs."""
        self.apply_fn(
            self._normalize_s2,
            input_dict,
            target_dict,
            [self.s2_selector],
        )
        self.apply_fn(
            self._normalize_s1,
            input_dict,
            target_dict,
            [self.s1_ascending_selector],
            mean=self.s1a_mean,
            std=self.s1a_std,
        )
        self.apply_fn(
            self._normalize_s1,
            input_dict,
            target_dict,
            [self.s1_descending_selector],
            mean=self.s1d_mean,
            std=self.s1d_std,
        )
        return input_dict, target_dict


class CustomGRUCell(nn.Module):
    """GRU cell used by the Tessera v1.1 temporal pooling layer."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        """Initialize the cell."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ir = nn.Linear(input_size, hidden_size, bias=False)
        self.W_iz = nn.Linear(input_size, hidden_size, bias=False)
        self.W_ih = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_r = nn.Parameter(torch.zeros(hidden_size))
        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize GRU weights (biases are left at zero)."""
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Apply one GRU step."""
        r_t = torch.sigmoid(self.W_ir(x_t) + self.W_hr(h_prev) + self.b_r)
        z_t = torch.sigmoid(self.W_iz(x_t) + self.W_hz(h_prev) + self.b_z)
        h_tilde = torch.tanh(self.W_ih(x_t) + self.W_hh(r_t * h_prev) + self.b_h)
        return (1 - z_t) * h_prev + z_t * h_tilde


class CustomGRU(nn.Module):
    """GRU layer matching the Tessera v1.1 inference module."""

    def __init__(
        self, input_size: int, hidden_size: int, batch_first: bool = True
    ) -> None:
        """Initialize the layer."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.gru_cell = CustomGRUCell(input_size, hidden_size)

    def forward(
        self, x: torch.Tensor, h_0: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the GRU over a sequence."""
        if self.batch_first:
            batch_size, seq_len, _ = x.shape
        else:
            seq_len, batch_size, _ = x.shape
            x = x.transpose(0, 1)

        if h_0 is None:
            h_t = torch.zeros(
                batch_size, self.hidden_size, device=x.device, dtype=x.dtype
            )
        else:
            h_t = h_0

        outputs = []
        for t in range(seq_len):
            h_t = self.gru_cell(x[:, t, :], h_t)
            outputs.append(h_t)
        stacked = torch.stack(outputs, dim=1)

        if not self.batch_first:
            stacked = stacked.transpose(0, 1)
        return stacked, h_t


class CustomTemporalAwarePooling(nn.Module):
    """Temporal-aware pooling used by Tessera v1.1."""

    def __init__(self, input_dim: int) -> None:
        """Initialize the pooling module."""
        super().__init__()
        self.input_dim = input_dim
        self.temporal_context = CustomGRU(input_dim, input_dim, batch_first=True)
        self.query = nn.Linear(input_dim, 1)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool a sequence to a single vector."""
        batch_size, seq_len, dim = x.shape
        if seq_len == 0:
            return torch.zeros(batch_size, dim, device=x.device, dtype=x.dtype)
        if seq_len == 1:
            return x.squeeze(1)
        x_context, _ = self.temporal_context(x)
        x_context = self.layer_norm(x_context)
        attn_weights = torch.softmax(self.query(x_context), dim=1)
        return (attn_weights * x).sum(dim=1)


class TemporalPositionalEncoder(nn.Module):
    """Sinusoidal day-of-year encoder."""

    def __init__(self, d_model: int) -> None:
        """Initialize the encoder."""
        super().__init__()
        self.d_model = d_model

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        """Encode day-of-year values."""
        position = doy.unsqueeze(-1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / self.d_model)
        ).to(doy.device)
        pe = torch.zeros(doy.shape[0], doy.shape[1], self.d_model, device=doy.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


class TransformerEncoder(nn.Module):
    """Tessera temporal transformer encoder."""

    def __init__(
        self,
        band_num: int,
        latent_dim: int,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the encoder."""
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(band_num, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, latent_dim * 4),
        )
        self.temporal_encoder = TemporalPositionalEncoder(d_model=latent_dim * 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim * 4,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.attn_pool = CustomTemporalAwarePooling(latent_dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a BxTx(C+DOY) tensor."""
        bands = x[:, :, :-1]
        doy = x[:, :, -1]
        x = self.embedding(bands) + self.temporal_encoder(doy)
        x = self.transformer_encoder(x)
        return self.attn_pool(x)


class MultimodalV1_1InferenceModel(nn.Module):
    """Tessera v1.1 S2 plus merged-S1 inference model."""

    def __init__(
        self,
        s2_backbone: TransformerEncoder,
        s1_backbone: TransformerEncoder,
        dim_reducer: nn.Module,
        fusion_method: str = "concat",
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.s2_backbone = s2_backbone
        self.s1_backbone = s1_backbone
        self.dim_reducer = dim_reducer
        self.fusion_method = fusion_method

    def forward(self, s2_x: torch.Tensor, s1_x: torch.Tensor) -> torch.Tensor:
        """Compute fused Tessera embeddings."""
        reprs = [self.s2_backbone(s2_x), self.s1_backbone(s1_x)]
        if self.fusion_method == "concat":
            fused = torch.cat(reprs, dim=-1)
        elif self.fusion_method == "sum":
            fused = sum(reprs)
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")
        return self.dim_reducer(fused)


def build_v1_1_inference_model(config: dict[str, Any]) -> MultimodalV1_1InferenceModel:
    """Build a Tessera v1.1 inference model."""
    if bool(config.get("split_s1_modalities", False)):
        raise ValueError(
            "Tessera v1.1 rslearn wrapper requires split_s1_modalities=False"
        )

    latent_dim = int(config.get("latent_dim", 192))
    repr_dim = int(config.get("representation_dim", latent_dim))
    fusion_method = str(config.get("fusion_method", "concat"))

    s2_enc = TransformerEncoder(
        band_num=10,
        latent_dim=latent_dim,
        nhead=int(config["s2_num_heads"]),
        num_encoder_layers=int(config["s2_num_layers"]),
        dim_feedforward=int(config["s2_dim_feedforward"]),
        dropout=0.1,
    )
    s1_enc = TransformerEncoder(
        band_num=2,
        latent_dim=latent_dim,
        nhead=int(config["s1_num_heads"]),
        num_encoder_layers=int(config["s1_num_layers"]),
        dim_feedforward=int(config["s1_dim_feedforward"]),
        dropout=0.1,
    )

    active = 2 if fusion_method == "concat" else 1
    reducer_in = latent_dim * 4 * active
    dim_reducer = nn.Sequential(
        nn.Linear(reducer_in, reducer_in * 2),
        nn.LayerNorm(reducer_in * 2),
        nn.ReLU(inplace=False),
        nn.Dropout(0.2),
        nn.Linear(reducer_in * 2, repr_dim),
    )
    return MultimodalV1_1InferenceModel(
        s2_backbone=s2_enc,
        s1_backbone=s1_enc,
        dim_reducer=dim_reducer,
        fusion_method=fusion_method,
    )


class Tessera(FeatureExtractor):
    """rslearn wrapper for Tessera v1.1.

    The model expects normalized input keys ``s2``, ``s1_ascending``, and
    ``s1_descending``. Use ``TesseraNormalize`` in the data transforms to
    prepare raw source imagery.

    All inputs are assumed to be complete: every timestep of every pixel is
    treated as a valid observation. Nodata or empty observations are not
    handled and would be fed to the model as-is.
    """

    input_keys = ["s2", "s1_ascending", "s1_descending"]

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        pixel_batch_size: int = 1024,
        apply_amp: bool = True,
        autocast_dtype: str | None = "bfloat16",
        random_initialization: bool = False,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize Tessera.

        Args:
            checkpoint_path: Local path to a Tessera v1.1 checkpoint.
            pixel_batch_size: Number of pixels to evaluate per forward chunk.
            apply_amp: Whether to use autocast on CUDA.
            autocast_dtype: Autocast dtype name, or None to disable autocast.
            random_initialization: skip checkpoint loading and initialize the model
                randomly instead.
            model_config: Optional architecture overrides.
        """
        super().__init__()
        self.pixel_batch_size = int(pixel_batch_size)
        self.apply_amp = apply_amp
        self.autocast_dtype = _get_autocast_dtype(autocast_dtype)

        config = dict(DEFAULT_MODEL_CONFIG)
        if model_config is not None:
            config.update(model_config)
        self.model_config = config
        self.embedding_dim = int(config["representation_dim"])

        self.model = build_v1_1_inference_model(config)
        if not random_initialization:
            if checkpoint_path is None:
                raise ValueError(
                    "checkpoint_path is required unless random_initialization=True"
                )
            ckpt = torch.load(
                Path(checkpoint_path), map_location="cpu", weights_only=True
            )
            self.model.load_state_dict(ckpt["model_state"])

    @staticmethod
    def _time_ranges_to_doy(
        time_ranges: list[tuple[datetime, datetime]] | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Convert raster timestamps to midpoint day-of-year values."""
        if time_ranges is None:
            raise ValueError("Tessera requires timestamps on all input RasterImages")
        doys = []
        for start, end in time_ranges:
            midpoint = start + ((end - start) / 2)
            doys.append(midpoint.timetuple().tm_yday)
        return torch.tensor(doys, dtype=torch.float32, device=device)

    @staticmethod
    def _build_sequence(
        image: torch.Tensor, doys: torch.Tensor, num_pixels: int
    ) -> torch.Tensor:
        """Build per-pixel observation sequences with a day-of-year channel.

        Args:
            image: a CTHW tensor for one modality.
            doys: day-of-year values for each timestep, shaped (T,).
            num_pixels: number of pixels (H * W) in the image.

        Returns:
            A (num_pixels, T, C + 1) tensor where the trailing channel is the
            day-of-year feature.
        """
        num_bands, num_timesteps, _, _ = image.shape
        # CTHW -> HWTC -> (num_pixels, T, C).
        bands = image.permute(2, 3, 1, 0).reshape(num_pixels, num_timesteps, num_bands)
        doy = doys.view(1, num_timesteps, 1).expand(num_pixels, num_timesteps, 1)
        return torch.cat([bands.float(), doy], dim=2)

    def _forward_one_sample(
        self,
        s2: RasterImage,
        s1_ascending: RasterImage,
        s1_descending: RasterImage,
    ) -> torch.Tensor:
        """Compute Tessera features for one sample.

        All timesteps are assumed valid, so every pixel's full observation
        sequence is passed to the model.
        """
        s2_image = s2.image
        s1a_image = s1_ascending.image
        s1d_image = s1_descending.image
        if s2_image.shape[0] != 10:
            raise ValueError(
                f"Tessera s2 input must have 10 bands, got {s2_image.shape[0]}"
            )
        if s1a_image.shape[0] != 2 or s1d_image.shape[0] != 2:
            raise ValueError("Tessera Sentinel-1 inputs must each have 2 bands")
        if (
            s2_image.shape[2:] != s1a_image.shape[2:]
            or s2_image.shape[2:] != s1d_image.shape[2:]
        ):
            raise ValueError("Tessera input spatial shapes must match")

        device = s2_image.device
        _, _, height, width = s2_image.shape
        num_pixels = height * width
        s2_doys = self._time_ranges_to_doy(s2.timestamps, device)
        s1a_doys = self._time_ranges_to_doy(s1_ascending.timestamps, device)
        s1d_doys = self._time_ranges_to_doy(s1_descending.timestamps, device)

        # Sentinel-1 ascending and descending observations are merged into a
        # single time series along the temporal axis, ascending first and
        # descending second. This order matches what Tessera expects (position matters
        # for the GRU component).
        s2_seq = self._build_sequence(s2_image, s2_doys, num_pixels)
        s1_seq = torch.cat(
            [
                self._build_sequence(s1a_image, s1a_doys, num_pixels),
                self._build_sequence(s1d_image, s1d_doys, num_pixels),
            ],
            dim=1,
        )

        outputs = []
        for start in range(0, num_pixels, self.pixel_batch_size):
            end = start + self.pixel_batch_size
            output = self._run_model(s2_seq[start:end], s1_seq[start:end])
            outputs.append(output.float())

        features = torch.cat(outputs, dim=0)
        return features.transpose(0, 1).reshape(self.embedding_dim, height, width)

    def _run_model(
        self, s2_batch: torch.Tensor, s1_batch: torch.Tensor
    ) -> torch.Tensor:
        """Run the Tessera model with optional CUDA autocast."""
        if (
            self.apply_amp
            and self.autocast_dtype is not None
            and s2_batch.device.type == "cuda"
        ):
            with torch.amp.autocast("cuda", dtype=self.autocast_dtype):
                return self.model(s2_batch, s1_batch)
        return self.model(s2_batch, s1_batch)

    def forward(self, context: ModelContext) -> FeatureMaps:
        """Compute Tessera feature maps."""
        feature_maps = []
        for inputs in context.inputs:
            for key in self.input_keys:
                if key not in inputs:
                    raise ValueError(f"Tessera requires input key {key!r}")
                if not isinstance(inputs[key], RasterImage):
                    raise ValueError(f"Tessera input key {key!r} must be a RasterImage")
            feature_maps.append(
                self._forward_one_sample(
                    inputs["s2"],
                    inputs["s1_ascending"],
                    inputs["s1_descending"],
                )
            )
        return FeatureMaps([torch.stack(feature_maps, dim=0)])

    def get_backbone_channels(self) -> list[tuple[int, int]]:
        """Return Tessera output channel metadata."""
        return [(1, self.embedding_dim)]


def _get_autocast_dtype(dtype_name: str | None) -> torch.dtype | None:
    """Convert an autocast dtype name to torch dtype."""
    if dtype_name is None:
        return None
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported autocast_dtype {dtype_name!r}")
    return dtype_map[dtype_name]
