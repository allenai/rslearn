"""Tessera v1.1 model wrapper.

The architecture and normalization constants are adapted from
https://github.com/ucam-eo/tessera/tree/master/tessera_infer_QAT, which is
released under the MIT license.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from rslearn.models.component import FeatureExtractor, FeatureMaps
from rslearn.train.model_context import ModelContext, RasterImage
from rslearn.train.transforms.transform import Transform

logger = logging.getLogger(__name__)

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

DEFAULT_NUM_OBS_CHECKPOINTS = list(range(8, 257, 8))

DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "fusion_method": "concat",
    "latent_dim": 192,
    "representation_dim": 192,
    "save_embedding_dim": 128,
    "s2_num_heads": 4,
    "s2_num_layers": 4,
    "s2_dim_feedforward": 2048,
    "s1_num_heads": 4,
    "s1_num_layers": 4,
    "s1_dim_feedforward": 2048,
    "split_s1_modalities": False,
    "num_obs_checkpoints": DEFAULT_NUM_OBS_CHECKPOINTS,
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
        """Normalize a Sentinel-2 CTHW image while preserving empty observations."""
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
        valid = torch.any(values != 0, dim=0, keepdim=True)
        image.image = torch.where(
            valid,
            (values - mean) / (std + 1e-9),
            torch.zeros_like(values),
        )
        return image

    @staticmethod
    def _db_to_tessera_scaled(values: torch.Tensor) -> torch.Tensor:
        """Convert standard power dB to Tessera's upstream Sentinel-1 scale."""
        return torch.clamp((2 * values + 50) * 200, min=0, max=32767)

    def _normalize_s1(
        self, image: RasterImage, mean: torch.Tensor, std: torch.Tensor
    ) -> RasterImage:
        """Normalize a Sentinel-1 CTHW image while preserving clipped-empty pixels."""
        if image.image.shape[0] != len(TESSERA_S1_BANDS):
            raise ValueError(
                "Tessera Sentinel-1 inputs must have "
                f"{len(TESSERA_S1_BANDS)} bands, got {image.image.shape[0]}"
            )
        values = image.image.float()
        scaled = self._db_to_tessera_scaled(values)
        mean = mean.to(device=scaled.device, dtype=scaled.dtype)[:, None, None, None]
        std = std.to(device=scaled.device, dtype=scaled.dtype)[:, None, None, None]
        valid = torch.any(scaled != 0, dim=0, keepdim=True)
        image.image = torch.where(
            valid,
            (scaled - mean) / (std + 1e-9),
            torch.zeros_like(scaled),
        )
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


def build_resample_indices(valid_len: int, target_size: int) -> list[int]:
    """Build deterministic indexes to resize a sequence to target_size.

    Args:
        valid_len: Number of valid observations in the original sequence.
        target_size: Requested output sequence length.

    Returns:
        Indexes into the valid observation list.
    """
    valid_len = int(valid_len)
    target_size = int(target_size)
    if valid_len <= 0:
        return []
    if target_size == valid_len:
        return list(range(valid_len))
    if target_size < valid_len:
        return [
            int((chunk_start + chunk_end) // 2)
            for chunk_start, chunk_end in _array_split_ranges(valid_len, target_size)
            if chunk_end > chunk_start
        ]

    extra = target_size - valid_len
    anchors = torch.linspace(0, valid_len - 1, steps=extra + 2, dtype=torch.float64)[
        1:-1
    ]
    extras = anchors.round().long().clamp(0, valid_len - 1).tolist()
    return list(range(valid_len)) + [int(idx) for idx in extras]


def _array_split_ranges(length: int, sections: int) -> list[tuple[int, int]]:
    """Return ranges equivalent to numpy.array_split for a 1D array."""
    base = length // sections
    remainder = length % sections
    ranges = []
    start = 0
    for idx in range(sections):
        size = base + (1 if idx < remainder else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


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
        """Initialize GRU weights."""
        for name, param in self.named_parameters():
            if "weight" in name or name.startswith("W_"):
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
    """

    input_keys = ["s2", "s1_ascending", "s1_descending"]

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        data_source: TesseraDataSource | str = TesseraDataSource.MPC,
        pixel_batch_size: int = 1024,
        num_obs_checkpoints: list[int] | None = None,
        save_embedding_dim: int = 128,
        apply_amp: bool = True,
        autocast_dtype: str | None = "bfloat16",
        random_initialization: bool = False,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize Tessera.

        Args:
            checkpoint_path: Local path to a Tessera v1.1 checkpoint.
            data_source: Deprecated compatibility argument. Normalization
                statistics are selected in ``TesseraNormalize``.
            pixel_batch_size: Number of pixels to evaluate per forward chunk.
            num_obs_checkpoints: Sequence-length buckets for all-observation inference.
            save_embedding_dim: Number of representation dimensions to return.
            apply_amp: Whether to use autocast on CUDA.
            autocast_dtype: Autocast dtype name, or None to disable autocast.
            random_initialization: Skip checkpoint loading, intended for tests only.
            model_config: Optional architecture overrides.
        """
        super().__init__()
        self.data_source = TesseraDataSource(data_source)
        self.pixel_batch_size = int(pixel_batch_size)
        self.apply_amp = apply_amp
        self.autocast_dtype = _get_autocast_dtype(autocast_dtype)

        config = dict(DEFAULT_MODEL_CONFIG)
        if model_config is not None:
            config.update(model_config)
        if num_obs_checkpoints is not None:
            config["num_obs_checkpoints"] = num_obs_checkpoints
        config["save_embedding_dim"] = save_embedding_dim
        self.model_config = config
        self.num_obs_checkpoints = sorted(
            {int(v) for v in config["num_obs_checkpoints"] if int(v) > 0}
        )
        if not self.num_obs_checkpoints:
            raise ValueError(
                "num_obs_checkpoints must include at least one positive integer"
            )
        self.save_embedding_dim = int(config["save_embedding_dim"])
        if self.save_embedding_dim <= 0:
            raise ValueError("save_embedding_dim must be positive")
        representation_dim = int(config["representation_dim"])
        if self.save_embedding_dim > representation_dim:
            raise ValueError(
                "save_embedding_dim must be less than or equal to representation_dim"
            )

        self.model = build_v1_1_inference_model(config)
        if not random_initialization:
            if checkpoint_path is None:
                raise ValueError(
                    "checkpoint_path is required unless random_initialization=True"
                )
            self._load_checkpoint(Path(checkpoint_path))

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load a Tessera checkpoint into the inference graph."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model_state" in ckpt:
            raw_state = ckpt["model_state"]
        elif "model_state_dict" in ckpt:
            raw_state = ckpt["model_state_dict"]
        else:
            raw_state = ckpt

        cleaned = {}
        for key, value in raw_state.items():
            if key.startswith("_orig_mod."):
                key = key[len("_orig_mod.") :]
            if key.startswith("projector.") or key.startswith(
                "segmented_matryoshka_projector."
            ):
                continue
            cleaned[key] = value

        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if missing:
            logger.info("Missing Tessera checkpoint keys: %s", missing[:10])
        if unexpected:
            logger.info("Unexpected Tessera checkpoint keys: %s", unexpected[:10])

    def _to_bin(self, n: int) -> int:
        """Map an observation count to a Tessera sequence-length bucket."""
        for checkpoint in self.num_obs_checkpoints:
            if n <= checkpoint:
                return checkpoint
        return self.num_obs_checkpoints[-1]

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

    def _prepare_s2_sequence(
        self,
        image: torch.Tensor,
        doys: torch.Tensor,
        row: int,
        col: int,
        target: int,
    ) -> torch.Tensor:
        """Prepare one pixel's S2 sequence."""
        series = image[:, :, row, col].transpose(0, 1)
        valid_idx = torch.nonzero(
            torch.any(series != 0, dim=1), as_tuple=False
        ).flatten()
        if len(valid_idx) == 0:
            return torch.zeros(target, 11, dtype=image.dtype, device=image.device)
        local_idx = torch.tensor(
            build_resample_indices(len(valid_idx), target),
            dtype=torch.long,
            device=image.device,
        )
        real_idx = valid_idx[local_idx]
        bands = series[real_idx].float()
        return torch.cat([bands, doys[real_idx, None]], dim=1)

    def _prepare_s1_sequence(
        self,
        asc_image: torch.Tensor,
        asc_doys: torch.Tensor,
        desc_image: torch.Tensor,
        desc_doys: torch.Tensor,
        row: int,
        col: int,
        target: int,
    ) -> torch.Tensor:
        """Prepare one pixel's merged S1 sequence."""
        parts = []
        for image, doys in [
            (asc_image, asc_doys),
            (desc_image, desc_doys),
        ]:
            series = image[:, :, row, col].transpose(0, 1)
            valid_idx = torch.nonzero(
                torch.any(series != 0, dim=1), as_tuple=False
            ).flatten()
            if len(valid_idx) == 0:
                continue
            bands = series[valid_idx].float()
            parts.append(torch.cat([bands, doys[valid_idx, None]], dim=1))

        if not parts:
            return torch.zeros(
                target, 3, dtype=asc_image.dtype, device=asc_image.device
            )

        merged = torch.cat(parts, dim=0)
        local_idx = torch.tensor(
            build_resample_indices(merged.shape[0], target),
            dtype=torch.long,
            device=asc_image.device,
        )
        return merged[local_idx]

    def _forward_one_sample(
        self,
        s2: RasterImage,
        s1_ascending: RasterImage,
        s1_descending: RasterImage,
    ) -> torch.Tensor:
        """Compute Tessera features for one sample."""
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
        s2_doys = self._time_ranges_to_doy(s2.timestamps, device)
        s1a_doys = self._time_ranges_to_doy(s1_ascending.timestamps, device)
        s1d_doys = self._time_ranges_to_doy(s1_descending.timestamps, device)

        features = torch.zeros(
            self.save_embedding_dim, height, width, dtype=torch.float32, device=device
        )
        buckets: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        s2_valid = torch.any(s2_image != 0, dim=0)
        s1a_valid = torch.any(s1a_image != 0, dim=0)
        s1d_valid = torch.any(s1d_image != 0, dim=0)
        for row in range(height):
            for col in range(width):
                s2_target = self._to_bin(int(s2_valid[:, row, col].sum().item()))
                s1_target = self._to_bin(
                    int(s1a_valid[:, row, col].sum().item())
                    + int(s1d_valid[:, row, col].sum().item())
                )
                buckets[(s2_target, s1_target)].append((row, col))

        for (s2_target, s1_target), coords in buckets.items():
            for start in range(0, len(coords), self.pixel_batch_size):
                chunk = coords[start : start + self.pixel_batch_size]
                s2_batch = torch.stack(
                    [
                        self._prepare_s2_sequence(
                            s2_image, s2_doys, row, col, s2_target
                        )
                        for row, col in chunk
                    ],
                    dim=0,
                )
                s1_batch = torch.stack(
                    [
                        self._prepare_s1_sequence(
                            s1a_image,
                            s1a_doys,
                            s1d_image,
                            s1d_doys,
                            row,
                            col,
                            s1_target,
                        )
                        for row, col in chunk
                    ],
                    dim=0,
                )
                output = self._run_model(s2_batch, s1_batch)[
                    :, : self.save_embedding_dim
                ]
                output = output.float()
                for idx, (row, col) in enumerate(chunk):
                    features[:, row, col] = output[idx]
        return features

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
        return [(1, self.save_embedding_dim)]


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
