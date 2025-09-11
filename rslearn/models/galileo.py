"""Galileo models."""

import collections.abc
import itertools
import json
import math
from collections import OrderedDict
from collections import OrderedDict as OrderedDictType
from collections.abc import Sequence
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from huggingface_hub import hf_hub_download
from torch import Tensor, vmap
from torch.jit import Final

from rslearn.log_utils import get_logger

logger = get_logger(__name__)

# constants
CONFIG_FILENAME = "config.json"
ENCODER_FILENAME = "encoder.pt"
BASE_GSD = 10
DEFAULT_MONTH = 5
HF_HUB_ID = "nasaharvest/galileo"


# Galileo provides three sizes: nano, tiny, base
class GalileoSize(str, Enum):
    """Size of the Galileo model."""

    NANO = "nano"
    TINY = "tiny"
    BASE = "base"


pretrained_weights: dict[GalileoSize, str] = {
    GalileoSize.NANO: "models/nano",
    GalileoSize.TINY: "models/tiny",
    GalileoSize.BASE: "models/base",
}

# band information
S1_BANDS = ["VV", "VH"]
S1_SHIFT_VALUES = [25.0, 25.0]
S1_DIV_VALUES = [25.0, 25.0]
S2_BANDS = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
]
S2_SHIFT_VALUES = [0.0] * len(S2_BANDS)
S2_DIV_VALUES = [1e4] * len(S2_BANDS)
ERA5_BANDS = ["temperature_2m", "total_precipitation_sum"]
# for temperature, shift to celcius and then divide by 35 based on notebook (ranges from)
# 37 to -22 degrees celcius
# For rainfall, based on
# https://github.com/nasaharvest/lem/blob/main/notebooks/exploratory_data_analysis.ipynb
ERA5_SHIFT_VALUES = [-272.15, 0.0]
ERA5_DIV_VALUES = [35.0, 0.03]
TC_BANDS = ["def", "soil", "aet"]
TC_SHIFT_VALUES = [0.0, 0.0, 0.0]
TC_DIV_VALUES = [4548, 8882, 2000]
VIIRS_BANDS = ["avg_rad"]
VIIRS_SHIFT_VALUES = [0.0]
# visually checked - this seems much more reasonable than
# the GEE estimate
VIIRS_DIV_VALUES = [100]
SRTM_BANDS = ["elevation", "slope"]
# visually gauged 90th percentile from
# https://github.com/nasaharvest/lem/blob/main/notebooks/exploratory_data_analysis.ipynb
SRTM_SHIFT_VALUES = [0.0, 0.0]
SRTM_DIV_VALUES = [2000.0, 50.0]
DW_BANDS = [
    "DW_water",
    "DW_trees",
    "DW_grass",
    "DW_flooded_vegetation",
    "DW_crops",
    "DW_shrub_and_scrub",
    "DW_built",
    "DW_bare",
    "DW_snow_and_ice",
]
DW_SHIFT_VALUES = [0] * len(DW_BANDS)
DW_DIV_VALUES = [1] * len(DW_BANDS)

WC_BANDS = [
    "WC_temporarycrops",
    "WC_maize",
    "WC_wintercereals",
    "WC_springcereals",
    "WC_irrigation",
]
WC_SHIFT_VALUES = [0] * len(WC_BANDS)
WC_DIV_VALUES = [100] * len(WC_BANDS)
STATIC_DW_BANDS = [f"{x}_static" for x in DW_BANDS]
STATIC_WC_BANDS = [f"{x}_static" for x in WC_BANDS]

LANDSCAN_BANDS = ["b1"]
# LANDSCAN values range from approximately 0 to 185000 in 2022: https://code.earthengine.google.com/?scriptPath=users/sat-io/awesome-gee-catalog-examples:population-socioeconomics/LANDSCAN-GLOBAL
LANDSCAN_SHIFT_VALUES = [92500]
LANDSCAN_DIV_VALUES = [92500]
LOCATION_BANDS = ["x", "y", "z"]

SPACE_TIME_BANDS = S1_BANDS + S2_BANDS + ["NDVI"]
TIME_BANDS = ERA5_BANDS + TC_BANDS + VIIRS_BANDS
SPACE_BANDS = SRTM_BANDS + DW_BANDS + WC_BANDS
STATIC_BANDS = LANDSCAN_BANDS + LOCATION_BANDS + STATIC_DW_BANDS + STATIC_WC_BANDS

# 0 for NDVI
SPACE_TIME_SHIFT_VALUES = np.array(S1_SHIFT_VALUES + S2_SHIFT_VALUES + [0])
SPACE_TIME_DIV_VALUES = np.array(S1_DIV_VALUES + S2_DIV_VALUES + [1])
TIME_SHIFT_VALUES = np.array(ERA5_SHIFT_VALUES + TC_SHIFT_VALUES + VIIRS_SHIFT_VALUES)
TIME_DIV_VALUES = np.array(ERA5_DIV_VALUES + TC_DIV_VALUES + VIIRS_DIV_VALUES)
SPACE_SHIFT_VALUES = np.array(SRTM_SHIFT_VALUES + DW_SHIFT_VALUES + WC_SHIFT_VALUES)
SPACE_DIV_VALUES = np.array(SRTM_DIV_VALUES + DW_DIV_VALUES + WC_DIV_VALUES)
# [0s, 1s] for the locations
STATIC_SHIFT_VALUES = np.array(
    LANDSCAN_SHIFT_VALUES + [0, 0, 0] + DW_SHIFT_VALUES + WC_SHIFT_VALUES
)
STATIC_DIV_VALUES = np.array(
    LANDSCAN_DIV_VALUES + [1, 1, 1] + DW_DIV_VALUES + WC_DIV_VALUES
)

SPACE_TIME_BANDS_GROUPS_IDX: OrderedDictType[str, list[int]] = OrderedDict(
    {
        "S1": [SPACE_TIME_BANDS.index(b) for b in S1_BANDS],
        "S2_RGB": [SPACE_TIME_BANDS.index(b) for b in ["B2", "B3", "B4"]],
        "S2_Red_Edge": [SPACE_TIME_BANDS.index(b) for b in ["B5", "B6", "B7"]],
        "S2_NIR_10m": [SPACE_TIME_BANDS.index(b) for b in ["B8"]],
        "S2_NIR_20m": [SPACE_TIME_BANDS.index(b) for b in ["B8A"]],
        "S2_SWIR": [SPACE_TIME_BANDS.index(b) for b in ["B11", "B12"]],
        "NDVI": [SPACE_TIME_BANDS.index("NDVI")],
    }
)

TIME_BAND_GROUPS_IDX: OrderedDictType[str, list[int]] = OrderedDict(
    {
        "ERA5": [TIME_BANDS.index(b) for b in ERA5_BANDS],
        "TC": [TIME_BANDS.index(b) for b in TC_BANDS],
        "VIIRS": [TIME_BANDS.index(b) for b in VIIRS_BANDS],
    }
)

SPACE_BAND_GROUPS_IDX: OrderedDictType[str, list[int]] = OrderedDict(
    {
        "SRTM": [SPACE_BANDS.index(b) for b in SRTM_BANDS],
        "DW": [SPACE_BANDS.index(b) for b in DW_BANDS],
        "WC": [SPACE_BANDS.index(b) for b in WC_BANDS],
    }
)

STATIC_BAND_GROUPS_IDX: OrderedDictType[str, list[int]] = OrderedDict(
    {
        "LS": [STATIC_BANDS.index(b) for b in LANDSCAN_BANDS],
        "location": [STATIC_BANDS.index(b) for b in LOCATION_BANDS],
        "DW_static": [STATIC_BANDS.index(b) for b in STATIC_DW_BANDS],
        "WC_static": [STATIC_BANDS.index(b) for b in STATIC_WC_BANDS],
    }
)


NORMALIZING_DICT = {
    "total_n": 127155,
    "sampled_n": 10000,
    "13": {
        "mean": [
            -11.728724389184965,
            -18.85558188024017,
            1395.3408730676722,
            1338.4026921784578,
            1343.09883810357,
            1543.8607982512297,
            2186.2022069512263,
            2525.0932853316694,
            2410.3377187373408,
            2750.2854646886753,
            2234.911100061487,
            1474.5311266077113,
            0.2892116502999044,
        ],
        "std": [
            4.887145774840316,
            5.730270320384293,
            917.7041440370853,
            913.2988423581528,
            1092.678723527555,
            1047.2206083460424,
            1048.0101611156767,
            1143.6903026819996,
            1098.979177731649,
            1204.472755085893,
            1145.9774063078878,
            980.2429840007796,
            0.2720939024500081,
        ],
    },
    "16": {
        "mean": [
            673.0152819503361,
            5.930092668915115,
            0.10470439140978786,
            0.23965913270066183,
            0.08158044385860364,
            0.04246976254259546,
            0.11304392863520317,
            0.17329647890362473,
            0.0698981691616277,
            0.12130267132802142,
            0.04671318615236216,
            10.973119802517362,
            1.0927069179958768,
            1.6991394232855903,
            0.03720594618055555,
            1.3671352688259548,
        ],
        "std": [
            983.0697298296237,
            8.167406789813247,
            0.18771647977504985,
            0.2368313455675914,
            0.08024268534756586,
            0.04045374496146404,
            0.11350342472061795,
            0.1279898111718168,
            0.12042341550438586,
            0.13602408145504347,
            0.043971116096060345,
            31.255340146970997,
            10.395974878206689,
            12.92380617159917,
            1.9285254295940466,
            11.612179775408928,
        ],
    },
    "6": {
        "mean": [
            271.5674963541667,
            0.08554303677156568,
            657.3181260091111,
            692.1291795806885,
            562.781331880633,
            1.5647115934036673,
        ],
        "std": [
            79.80828940314429,
            0.11669547098151486,
            704.0008695557707,
            925.0116126406431,
            453.2434022278578,
            7.513020170832818,
        ],
    },
    "18": {
        "mean": [
            188.20315880851746,
            0.2804946561574936,
            0.11371652073860168,
            0.058778801321983334,
            0.10474256777763366,
            0.2396918488264084,
            0.08152248692512512,
            0.04248040814399719,
            0.11303179881572724,
            0.17326324067115784,
            0.06998309404850006,
            0.12122812910079957,
            0.04671641788482666,
            10.98456594619751,
            1.0968475807189941,
            1.6947754135131836,
            0.03320046615600586,
            1.3602827312469483,
        ],
        "std": [
            1154.5919128300602,
            0.5276998078079327,
            0.7021637331734328,
            0.36528892213195063,
            0.17470213191865785,
            0.20411195416718833,
            0.0660782470089761,
            0.03380702424871257,
            0.09809195568521663,
            0.11292471052124119,
            0.09720748930233268,
            0.12912217763726777,
            0.0399973913151906,
            23.725471823867462,
            5.715238079725388,
            9.030481416228302,
            0.9950220242487364,
            7.754429123862099,
        ],
    },
}


class Normalizer:
    """Normalize Galileo inputs."""

    std_bands: dict[int, list] = {
        len(SPACE_TIME_BANDS): [b for b in SPACE_TIME_BANDS if b != "NDVI"],
        len(SPACE_BANDS): SRTM_BANDS,
        len(TIME_BANDS): TIME_BANDS,
        len(STATIC_BANDS): LANDSCAN_BANDS,
    }

    def __init__(self, std_multiplier: float = 2):
        """Normalize Galileo inputs.

        Args:
            std_multiplier: std_multiplier to apply
        """
        name_to_bands = {
            len(SPACE_TIME_BANDS): SPACE_TIME_BANDS,
            len(SPACE_BANDS): SPACE_BANDS,
            len(TIME_BANDS): TIME_BANDS,
            len(STATIC_BANDS): STATIC_BANDS,
        }
        self.shift_div_dict = {
            len(SPACE_TIME_BANDS): {
                "shift": deepcopy(SPACE_TIME_SHIFT_VALUES),
                "div": deepcopy(SPACE_TIME_DIV_VALUES),
            },
            len(SPACE_BANDS): {
                "shift": deepcopy(SPACE_SHIFT_VALUES),
                "div": deepcopy(SPACE_DIV_VALUES),
            },
            len(TIME_BANDS): {
                "shift": deepcopy(TIME_SHIFT_VALUES),
                "div": deepcopy(TIME_DIV_VALUES),
            },
            len(STATIC_BANDS): {
                "shift": deepcopy(STATIC_SHIFT_VALUES),
                "div": deepcopy(STATIC_DIV_VALUES),
            },
        }
        for key_as_str, val in NORMALIZING_DICT.items():
            if "n" in key_as_str:
                continue
            key = int(key_as_str)
            bands_to_replace = self.std_bands[key]
            for band in bands_to_replace:
                band_idx = name_to_bands[key].index(band)
                mean = cast(dict[str, list], val)["mean"][band_idx]
                std = cast(dict[str, list], val)["std"][band_idx]
                min_value = mean - (std_multiplier * std)
                max_value = mean + (std_multiplier * std)
                div = max_value - min_value
                if div == 0:
                    raise ValueError(f"{band} has div value of 0")
                self.shift_div_dict[key]["shift"][band_idx] = min_value
                self.shift_div_dict[key]["div"][band_idx] = div

    @staticmethod
    def _normalize(
        x: torch.Tensor, shift_values: torch.Tensor, div_values: torch.Tensor
    ) -> torch.Tensor:
        x = (x - shift_values) / div_values
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the normalizer."""
        div_values = self.shift_div_dict[x.shape[-1]]["div"]
        shift_values = self.shift_div_dict[x.shape[-1]]["shift"]
        return self._normalize(x, shift_values, div_values)


DEFAULT_NORMALIZER = Normalizer()


class MaskedOutput(NamedTuple):
    """A masked output (i.e. an input to Galileo).

    A mask can take 3 values:
    0: seen by the encoder (i.e. makes the key and value tokens in the decoder)
    1: not seen by the encoder, and ignored by the decoder
    2: not seen by the encoder, and processed by the decoder (the decoder's query values)
    """

    s_t_x: torch.Tensor  # [B, H, W, T, len(SPACE_TIME_BANDS)]
    sp_x: torch.Tensor  # [B, H, W, len(SPACE_BANDS)]
    t_x: torch.Tensor  # [B, T, len(TIME_BANDS)]
    st_x: torch.Tensor  # [B, len(STATIC_BANDS)]
    s_t_m: torch.Tensor  # [B, H, W, T, len(SPACE_TIME_BANDS_GROUPS_IDX)]
    sp_m: torch.Tensor  # [B, H, W, len(SPACE_BAND_GROUPS_IDX)]
    t_m: torch.Tensor  # [B, T, len(TIME_BAND_GROUPS_IDX)]
    st_m: torch.Tensor  # [B, len(STATIC_BAND_GROUPS_IDX)]
    months: torch.Tensor  # [B, T]


def get_2d_sincos_pos_embed_with_resolution(
    embed_dim: int,
    grid_size: int,
    res: torch.Tensor,
    cls_token: bool = False,
    device: str = "cpu",
) -> torch.Tensor:
    """Create 2d sincos embeddings with resolution.

    grid_size: int of the grid height and width
    res: array of size n, representing the resolution of a pixel (say, in meters),

    Return:
    pos_embed: [n,grid_size*grid_size, embed_dim] or [n,1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    res = res.to(device)
    grid_h = torch.arange(grid_size, device=device)
    grid_w = torch.arange(grid_size, device=device)
    grid = torch.meshgrid(
        grid_w, grid_h, indexing="xy"
    )  # here h goes first,direction reversed for numpy
    grid = torch.stack(grid, dim=0)  # 2 x h x w

    # grid = grid.reshape([2, 1, grid_size, grid_size])
    grid = torch.einsum("chw,n->cnhw", grid, res)  # 2 x n x h x w
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_embed_from_grid_torch(
        embed_dim, grid
    )  #  # (nxH*W, D/2)
    pos_embed = pos_embed.reshape(n, h * w, embed_dim)
    if cls_token:
        pos_embed = torch.cat(
            [
                torch.zeros([n, 1, embed_dim], device=pos_embed.device),
                pos_embed,
            ],
            dim=1,
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid_torch(
    embed_dim: int, grid: torch.Tensor
) -> torch.Tensor:
    """get_2d_sincos_pos_embed_from_grid_torch."""
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(
    embed_dim: int, pos: torch.Tensor
) -> torch.Tensor:
    """get_1d_sincos_pos_embed_from_grid_torch.

    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, device=pos.device) / embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def get_month_encoding_table(embed_dim: int) -> torch.Tensor:
    """Sinusoid month encoding table, for 12 months indexed from 0-11."""
    assert embed_dim % 2 == 0
    angles = torch.arange(0, 13) / (12 / (2 * np.pi))

    sin_table = torch.sin(torch.stack([angles for _ in range(embed_dim // 2)], axis=-1))
    cos_table = torch.cos(torch.stack([angles for _ in range(embed_dim // 2)], axis=-1))
    month_table = torch.concatenate([sin_table[:-1], cos_table[:-1]], axis=-1)

    return month_table  # (M, D)


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    warmup_epochs: int,
    total_epochs: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Decay the learning rate with half-cycle cosine after warmup."""
    if epoch < warmup_epochs:
        lr = max_lr * epoch / warmup_epochs
    else:
        lr = min_lr + (max_lr - min_lr) * 0.5 * (
            1.0
            + math.cos(
                math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            )
        )
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


# thanks to https://github.com/bwconrad/flexivit/ for this nice implementation
# of the FlexiPatchEmbed module
def to_2tuple(x: int | tuple[int, int]) -> tuple[int, int]:
    """to_2tuple."""
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)  # type: ignore
    return tuple(itertools.repeat(x, 2))  # type: ignore


class FlexiPatchEmbed(nn.Module):
    """FlexiPatchEmbed."""

    def __init__(
        self,
        patch_size: int | tuple[int, int],
        in_chans: int = 3,
        embed_dim: int = 128,
        norm_layer: nn.Module | None = None,
        bias: bool = True,
        patch_size_seq: Sequence[int] = (1, 2, 3, 4, 5, 6),
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        """2D image to patch embedding w/ flexible patch sizes.

        Extended from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L24
        by https://github.com/bwconrad/flexivit/

        Args:
            patch_size: Base patch size. i.e the size of the parameter buffer
            in_chans: Number of input image channels
            embed_dim: Network embedding dimension size
            norm_layer: Optional normalization layer
            bias: Whether to use bias in convolution
            patch_size_seq: List of patch sizes to randomly sample from
            interpolation: Resize interpolation type
            antialias: Whether to apply antialiasing resizing
        """
        super().__init__()

        self.patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # Flexi specific attributes
        self.interpolation = interpolation
        self.antialias = antialias

        self.patch_size_seq = patch_size_seq

        # Pre-calculate pinvs
        self.pinvs = self._cache_pinvs()

    def _cache_pinvs(self) -> dict:
        """Pre-calculate all pinv matrices."""
        pinvs = {}
        for ps in self.patch_size_seq:
            tuple_ps = to_2tuple(ps)
            pinvs[tuple_ps] = self._calculate_pinv(self.patch_size, tuple_ps)
        return pinvs

    def _resize(self, x: Tensor, shape: tuple[int, int]) -> Tensor:
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=self.interpolation,
            antialias=self.antialias,
        )
        return x_resized[0, 0, ...]

    def _calculate_pinv(
        self, old_shape: tuple[int, int], new_shape: tuple[int, int]
    ) -> Tensor:
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(self._resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    def resize_patch_embed(
        self, patch_embed: Tensor, new_patch_size: tuple[int, int]
    ) -> torch.Tensor:
        """Resize patch_embed to target resolution via pseudo-inverse resizing."""
        # Return original kernel if no resize is necessary
        if self.patch_size == new_patch_size:
            return patch_embed

        # Calculate pseudo-inverse of resize matrix
        if new_patch_size not in self.pinvs:
            self.pinvs[new_patch_size] = self._calculate_pinv(
                self.patch_size, new_patch_size
            )
        pinv = self.pinvs[new_patch_size]
        pinv = pinv.to(patch_embed.device)

        def resample_patch_embed(patch_embed: Tensor) -> torch.Tensor:
            h, w = new_patch_size
            resampled_kernel = pinv @ patch_embed.reshape(-1)
            return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

        v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)

        return v_resample_patch_embed(patch_embed)

    def forward(
        self,
        x: Tensor,
        patch_size: int | tuple[int, int] | None = None,
    ) -> Tensor | tuple[Tensor, tuple[int, int]]:
        """Forward pass."""
        # x has input shape [b, h, w, (t), c]
        batch_size = x.shape[0]
        has_time_dimension = False
        num_timesteps = 0  # ignored if has_time_dimension is False
        if len(x.shape) == 5:
            has_time_dimension = True
            num_timesteps = x.shape[3]
            x = rearrange(x, "b h w t c -> (b t) c h w")
        else:
            x = rearrange(x, "b h w c -> b c h w")

        if not patch_size:
            # During evaluation use base patch size if not specified
            patch_size = self.patch_size

        patch_size = to_2tuple(patch_size)

        # Resize conv weights
        if patch_size == self.patch_size:
            weight = self.proj.weight
        else:
            weight = self.resize_patch_embed(self.proj.weight, patch_size)
        # Apply conv with resized weights
        x = F.conv2d(x, weight, bias=self.proj.bias, stride=patch_size)

        if has_time_dimension:
            x = rearrange(x, "(b t) c h w -> b h w t c", b=batch_size, t=num_timesteps)
        else:
            x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)

        return x


class Attention(nn.Module):
    """Attention."""

    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    fast_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        cross_attn: bool = False,
    ) -> None:
        """Initialize attention."""
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fast_attn = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )  # FIXME

        self.cross_attn = cross_attn

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        B, N, C = x.shape

        q = self.q(x)

        if y is None:
            assert not self.cross_attn
            k = self.k(x)
            v = self.v(x)
        else:
            assert self.cross_attn
            k = self.k(y)
            v = self.v(y)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        q, k = self.q_norm(q), self.k_norm(k)
        if self.fast_attn:
            if attn_mask is not None:
                attn_mask = attn_mask[:, None, None].repeat((1, self.num_heads, N, 1))
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                # a value of True indicates that the element should take part in attention
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p,
            )
        else:
            if attn_mask is not None:
                raise NotImplementedError
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: nn.Module = nn.GELU,
        bias: bool = True,
        drop: float = 0.0,
    ) -> None:
        """Initialize the MLP."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerScale(nn.Module):
    """LayerScale."""

    def __init__(
        self, dim: int, init_values: float = 1e-5, inplace: bool = False
    ) -> None:
        """Init layerscale."""
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    """Drop path."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float) -> None:
        """Init."""
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    """An Attention block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float | None = None,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        cross_attn: bool = False,
    ) -> None:
        """Init."""
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
            cross_attn=cross_attn,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, attn_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Forward."""
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), y, attn_mask)))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


class ModuleListWithInit(nn.ModuleList):
    """module list with an init function."""

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)


class GalileoBase(nn.Module):
    """Galileo Base."""

    cross_attn: bool

    def __init__(
        self,
        embedding_size: int = 128,
        depth: int = 2,
        mlp_ratio: int = 2,
        num_heads: int = 8,
        max_sequence_length: int = 24,
        base_patch_size: int = 4,
        use_channel_embs: bool = True,
        drop_path: float = 0.0,
    ) -> None:
        """Init."""
        super().__init__()

        self.space_time_groups = SPACE_TIME_BANDS_GROUPS_IDX
        self.space_groups = SPACE_BAND_GROUPS_IDX
        self.time_groups = TIME_BAND_GROUPS_IDX
        self.static_groups = STATIC_BAND_GROUPS_IDX
        self.embedding_size = embedding_size
        self.base_patch_size = base_patch_size

        self.blocks = ModuleListWithInit(
            [
                Block(
                    embedding_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                    cross_attn=self.cross_attn,
                    drop_path=drop_path,
                )
                for _ in range(depth)
            ]
        )

        self.max_sequence_length = max_sequence_length
        # we have 4 embeddings (pos_in_time, pos_in_space, month, channel) so each get
        # 0.25 of the dimension. This will change soon anyway
        self.pos_embed = nn.Parameter(
            get_1d_sincos_pos_embed_from_grid_torch(
                int(embedding_size * 0.25), torch.arange(max_sequence_length)
            ),
            requires_grad=False,
        )
        month_tab = get_month_encoding_table(int(embedding_size * 0.25))
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)
        if use_channel_embs:
            args = {"requires_grad": True}
        else:
            args = {"requires_grad": False}
        self.s_t_channel_embed = nn.Parameter(
            torch.zeros(len(SPACE_TIME_BANDS_GROUPS_IDX), int(embedding_size * 0.25)),
            **args,
        )
        self.sp_channel_embed = nn.Parameter(
            torch.zeros(len(SPACE_BAND_GROUPS_IDX), int(embedding_size * 0.25)), **args
        )
        self.t_channel_embed = nn.Parameter(
            torch.zeros(len(TIME_BAND_GROUPS_IDX), int(embedding_size * 0.25)), **args
        )
        self.st_channel_embed = nn.Parameter(
            torch.zeros(len(STATIC_BAND_GROUPS_IDX), int(embedding_size * 0.25)), **args
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @classmethod
    def collapse_and_combine_hwtc(
        cls,
        s_t_x: torch.Tensor,
        sp_x: torch.Tensor,
        t_x: torch.Tensor,
        st_x: torch.Tensor,
        s_t_m: torch.Tensor,
        sp_m: torch.Tensor,
        t_m: torch.Tensor,
        st_m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """collapse_and_combine_hwtc."""
        s_t_x = rearrange(s_t_x, "b h w t c_g d -> b (h w t c_g) d")
        sp_x = rearrange(sp_x, "b h w c_g d -> b (h w c_g) d")
        t_x = rearrange(t_x, "b t c_g d -> b (t c_g) d")

        s_t_m = rearrange(s_t_m, "b h w t c_g-> b (h w t c_g)")
        sp_m = rearrange(sp_m, "b h w c_g-> b (h w c_g)")
        t_m = rearrange(t_m, "b t c_g -> b (t c_g)")

        x = torch.cat(
            [
                s_t_x,
                sp_x,
                t_x,
                st_x,
            ],
            dim=1,
        )
        m = torch.cat([s_t_m, sp_m, t_m, st_m], dim=1)
        return x, m

    @classmethod
    def split_and_expand_hwtc(
        cls,
        x: torch.Tensor,
        h: int,
        w: int,
        t: int,
        s_t_c_g: int,
        sp_c_g: int,
        t_c_g: int,
        st_c_g: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """split_and_expand_hwtc."""
        n_s_t_t = h * w * t * s_t_c_g
        n_t_t = t * t_c_g

        s_t_x = rearrange(
            x[:, :n_s_t_t], "b (h w t c) d -> b h w t c d", h=h, w=w, t=t, c=s_t_c_g
        )
        sp_x = rearrange(
            x[:, n_s_t_t : -(n_t_t + st_c_g)],
            "b (h w c) d -> b h w c d",
            h=h,
            w=w,
            c=sp_c_g,
        )
        t_x = rearrange(
            x[:, -(n_t_t + st_c_g) : -st_c_g], "b (t c) d -> b t c d", t=t, c=t_c_g
        )
        st_x = x[:, -st_c_g:]

        return s_t_x, sp_x, t_x, st_x

    def apply_encodings(
        self,
        s_t_x: torch.Tensor,
        sp_x: torch.Tensor,
        t_x: torch.Tensor,
        st_x: torch.Tensor,
        months: torch.Tensor,
        patch_size: int,
        input_res: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """apply_encodings."""
        b, h, w, t, s_t_c_g, _ = s_t_x.shape
        sp_c_g, t_c_g = sp_x.shape[-2], t_x.shape[-2]
        st_c_g = st_x.shape[-2]

        s_t_channel = repeat(
            self.s_t_channel_embed, "c_g d -> b h w t c_g d", b=b, h=h, w=w, t=t
        )
        t_channel = repeat(self.t_channel_embed, "c_g d -> b t c_g d", b=b, t=t)
        st_channel = repeat(self.st_channel_embed, "c_g d -> b c_g d", b=b)
        sp_channel = repeat(
            self.sp_channel_embed, "c_g d -> b h w c_g d", b=b, h=h, w=w
        )

        pos_embed_s_t = repeat(
            self.pos_embed[:t], "t d -> b h w t c_g d", b=b, h=h, w=w, c_g=s_t_c_g
        )
        m_embed_s_t = repeat(
            self.month_embed(months), "b t d -> b h w t c_g d", h=h, w=w, c_g=s_t_c_g
        )

        pos_embed_t = repeat(self.pos_embed[:t], "t d -> b t c_g d", b=b, c_g=t_c_g)
        m_embed_t = repeat(self.month_embed(months), "b t d -> b t c_g d", c_g=t_c_g)
        t_zeros = torch.zeros(
            b, t, t_c_g, int(self.embedding_size * 0.25), device=t_x.device
        )

        sp_zeros = torch.zeros(
            b,
            h,
            w,
            sp_c_g,
            sp_channel.shape[-1] * 2,
            device=sp_channel.device,
        )

        st_zeros = torch.zeros(
            b, st_c_g, st_channel.shape[-1] * 3, device=st_channel.device
        )

        # find the resolution that each token represents, which will be
        # the number of pixels in a patch * the resolution of each pixel
        if patch_size is None:
            patch_size = self.base_patch_size
        token_res = input_res * patch_size
        gsd_ratio = token_res / BASE_GSD

        assert h == w, (
            "get_2d_sincos_pos_embed_with_resolution currently requires that h==w"
        )
        spatial_embed = get_2d_sincos_pos_embed_with_resolution(
            int(self.embedding_size * 0.25),
            h,
            torch.ones(b).to(s_t_x.device) * gsd_ratio,
            device=s_t_x.device,
        )
        spatial_embed = rearrange(spatial_embed, "b (h w) d -> b h w d", h=h, w=w)
        spatial_embed_s_t = repeat(
            spatial_embed, "b h w d -> b h w t c_g d", h=h, w=w, t=t, c_g=s_t_c_g
        )
        spatial_embed_s = repeat(
            spatial_embed, "b h w d -> b h w c_g d", h=h, w=w, c_g=sp_c_g
        )

        s_t_embed = torch.cat(
            [s_t_channel, pos_embed_s_t, m_embed_s_t, spatial_embed_s_t], dim=-1
        )
        sp_embed = torch.cat([sp_channel, sp_zeros, spatial_embed_s], dim=-1)
        t_embed = torch.cat([t_channel, pos_embed_t, m_embed_t, t_zeros], dim=-1)
        st_embed = torch.cat([st_channel, st_zeros], dim=-1)
        return s_t_x + s_t_embed, sp_x + sp_embed, t_x + t_embed, st_x + st_embed


class Encoder(GalileoBase):
    """Galileo Encoder."""

    cross_attn = False

    def __init__(
        self,
        max_patch_size: int = 8,
        embedding_size: int = 128,
        depth: int = 2,
        mlp_ratio: int = 2,
        num_heads: int = 8,
        max_sequence_length: int = 24,
        freeze_projections: bool = False,
        drop_path: float = 0.0,
    ) -> None:
        """Init."""
        super().__init__(
            embedding_size,
            depth,
            mlp_ratio,
            num_heads,
            max_sequence_length,
            max_patch_size,
            use_channel_embs=True,
            drop_path=drop_path,
        )

        self.space_time_embed = nn.ModuleDict(
            {
                group_name: FlexiPatchEmbed(
                    in_chans=len(group),
                    embed_dim=embedding_size,
                    patch_size=max_patch_size,
                )
                for group_name, group in self.space_time_groups.items()
            }
        )
        self.space_embed = nn.ModuleDict(
            {
                group_name: FlexiPatchEmbed(
                    in_chans=len(group),
                    embed_dim=embedding_size,
                    patch_size=max_patch_size,
                )
                for group_name, group in self.space_groups.items()
            }
        )
        self.time_embed = nn.ModuleDict(
            {
                group_name: nn.Linear(
                    in_features=len(group), out_features=embedding_size
                )
                for group_name, group in self.time_groups.items()
            }
        )
        self.static_embed = nn.ModuleDict(
            {
                group_name: nn.Linear(
                    in_features=len(group), out_features=embedding_size
                )
                for group_name, group in self.static_groups.items()
            }
        )
        if freeze_projections:
            self.space_time_embed.requires_grad_(False)
            self.space_embed.requires_grad_(False)
            self.time_embed.requires_grad_(False)
            self.static_embed.requires_grad_(False)
        self.norm = nn.LayerNorm(embedding_size)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def apply_linear_projection(
        self,
        s_t_x: torch.Tensor,
        sp_x: torch.Tensor,
        t_x: torch.Tensor,
        st_x: torch.Tensor,
        s_t_m: torch.Tensor,
        sp_m: torch.Tensor,
        t_m: torch.Tensor,
        st_m: torch.Tensor,
        patch_size: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """apply_linear_projection.

        Given a [B, H, W, (T), C] inputs, returns a [B, H, W, (T), C_G, D] output.
        We assume that the spatial masks are consistent for the given patch size,
        so that if patch_size == 2 then one possible mask would be
        [0, 0, 1, 1]
        [0, 0, 1, 1]
        [1, 1, 0, 0]
        [1, 1, 0, 0]
        for the H, W dimensions
        """
        b, h, w, t, _ = s_t_x.shape
        new_h, new_w = h // patch_size, w // patch_size

        s_t_l, sp_l, t_l, st_l, s_t_m_l, sp_m_l, t_m_l, st_m_l = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for idx, (channel_group, channel_idxs) in enumerate(
            self.space_time_groups.items()
        ):
            s_t_m_l.append(s_t_m[:, 0::patch_size, 0::patch_size, :, idx])
            if s_t_m_l[-1].min() == 0:
                s_t_l.append(
                    self.space_time_embed[channel_group](
                        s_t_x[:, :, :, :, channel_idxs], patch_size=patch_size
                    )
                )
            else:
                s_t_l.append(
                    torch.empty(
                        b,
                        new_h,
                        new_w,
                        t,
                        self.embedding_size,
                        dtype=s_t_x.dtype,
                        device=s_t_x.device,
                    )
                )
        for idx, (channel_group, channel_idxs) in enumerate(self.space_groups.items()):
            sp_m_l.append(sp_m[:, 0::patch_size, 0::patch_size, idx])
            if sp_m_l[-1].min() == 0:
                sp_l.append(
                    self.space_embed[channel_group](
                        sp_x[:, :, :, channel_idxs], patch_size=patch_size
                    )
                )
            else:
                sp_l.append(
                    torch.empty(
                        b,
                        new_h,
                        new_w,
                        self.embedding_size,
                        dtype=sp_x.dtype,
                        device=sp_x.device,
                    )
                )

        for idx, (channel_group, channel_idxs) in enumerate(self.time_groups.items()):
            t_m_l.append(t_m[:, :, idx])
            if t_m_l[-1].min() == 0:
                t_l.append(self.time_embed[channel_group](t_x[:, :, channel_idxs]))
            else:
                t_l.append(
                    torch.empty(
                        b, t, self.embedding_size, dtype=t_x.dtype, device=t_x.device
                    )
                )

        for idx, (channel_group, channel_idxs) in enumerate(self.static_groups.items()):
            st_m_l.append(st_m[:, idx])
            if st_m_l[-1].min() == 0:
                st_l.append(self.static_embed[channel_group](st_x[:, channel_idxs]))
            else:
                st_l.append(
                    torch.empty(
                        b, self.embedding_size, dtype=st_x.dtype, device=st_x.device
                    )
                )

        return (
            torch.stack(s_t_l, dim=-2),
            torch.stack(sp_l, dim=-2),
            torch.stack(t_l, dim=-2),
            torch.stack(st_l, dim=-2),
            torch.stack(s_t_m_l, dim=-1),
            torch.stack(sp_m_l, dim=-1),
            torch.stack(t_m_l, dim=-1),
            torch.stack(st_m_l, dim=-1),
        )

    @staticmethod
    def remove_masked_tokens(
        x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Remove masked tokens."""
        org_mask_dtype = mask.dtype
        mask = mask.bool()
        # https://stackoverflow.com/a/68621610/2332296
        # move all non-masked values to the front of their rows
        sorted_mask, indices = torch.sort(
            (~mask).int(), dim=1, descending=True, stable=True
        )
        x = x.gather(1, indices[:, :, None].expand_as(x))
        # set masked values to 0 (not really necessary since we'll ignore them anyway)
        x = x * sorted_mask.unsqueeze(-1)

        # cut off to the length of the longest sequence
        max_length = sorted_mask.sum(-1).max()
        x = x[:, :max_length]
        updated_mask = 1 - sorted_mask[:, :max_length]

        return x, indices, updated_mask.to(dtype=org_mask_dtype)

    @staticmethod
    def add_removed_tokens(
        x: torch.Tensor, indices: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """add_removed_tokens."""
        masked_tokens = repeat(
            torch.zeros_like(x[0, 0, :]), "d -> b t d", b=x.shape[0], t=indices.shape[1]
        )
        full_mask = torch.cat(
            (
                mask,
                torch.ones(
                    (x.shape[0], indices.shape[1] - x.shape[1]),
                    device=x.device,
                    dtype=mask.dtype,
                ),
            ),
            dim=-1,
        )
        # can't set value on leaf variable
        out = masked_tokens.clone()
        # put tokens in full masked tensor (at the first N positions in every row)
        out[~full_mask.bool()] = x[~mask.bool()]
        # then move them to their original positions
        out = out.scatter(1, indices[:, :, None].expand_as(out), out)
        full_mask = full_mask.scatter(1, indices.expand_as(full_mask), full_mask)
        return out, full_mask

    def apply_attn(
        self,
        s_t_x: torch.Tensor,
        sp_x: torch.Tensor,
        t_x: torch.Tensor,
        st_x: torch.Tensor,
        s_t_m: torch.Tensor,
        sp_m: torch.Tensor,
        t_m: torch.Tensor,
        st_m: torch.Tensor,
        months: torch.Tensor,
        patch_size: int,
        input_res: int,
        exit_after: int | None,
        token_exit_cfg: dict | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """apply_attn."""
        if token_exit_cfg:
            exit_s_t, exit_sp, exit_t, exit_st = self.create_token_exit_ids(
                s_t_x, sp_x, t_x, st_x, token_exit_cfg
            )
            exit_ids_seq, _ = self.collapse_and_combine_hwtc(
                exit_s_t, exit_sp, exit_t, exit_st, s_t_m, sp_m, t_m, st_m
            )
            # exited_tokens starts as linear projections!
            exited_tokens, _ = self.collapse_and_combine_hwtc(
                s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m
            )
        else:
            exit_ids_seq = None
            exited_tokens = None

        _, h, w, t, s_t_c_g, _ = s_t_x.shape
        sp_c_g, t_c_g, st_c_g = sp_x.shape[3], t_x.shape[-2], st_x.shape[-2]
        s_t_x, sp_x, t_x, st_x = self.apply_encodings(
            s_t_x, sp_x, t_x, st_x, months, patch_size, input_res
        )
        x, m = self.collapse_and_combine_hwtc(
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m
        )

        # we only care about the values >= 1 for this mask, since 2 just tells the decoder
        # to decode those tokens. From the perspective of the encoder, 1 and 2 are equivalent
        # since they both represent masked values
        new_m = m >= 1
        x, indices, new_m = self.remove_masked_tokens(
            x, new_m
        )  # new_m is shape (bsz, seq_len)

        if exit_ids_seq is not None:
            exit_ids_seq, _, _ = self.remove_masked_tokens(exit_ids_seq, m >= 1)
            # still linear projections
            exited_tokens, _, _ = self.remove_masked_tokens(exited_tokens, m >= 1)

        for i_blk, blk in enumerate(self.blocks):
            if (exit_after is not None) and ((i_blk + 1) > exit_after):
                # if exit_after is N, then we exit after the Nth layer
                # if exit_after is 0, then all layers are skipped
                break

            # skip the 0th block since this is just the linear
            # projection
            if (exit_ids_seq is not None) and (i_blk > 0):
                assert exited_tokens is not None
                # half depth
                exited_tokens = torch.where(
                    condition=(exit_ids_seq == i_blk),
                    input=x.detach(),
                    other=exited_tokens.detach(),
                )

            # we take the inverse of the mask because a value
            # of True indicates the value *should* take part in
            # attention
            x = blk(x=x, y=None, attn_mask=~new_m.bool())

        if exit_ids_seq is not None:
            assert exited_tokens is not None
            # full depth
            # IMPORTANT: write this to x
            x = torch.where(
                condition=(exit_ids_seq == (i_blk + 1)),  # 2 for full depth
                input=x.detach(),
                other=exited_tokens.detach(),
            )

        # we don't care about the mask returned by add_removed_tokens, since we will
        # just use the original, unclipped mask here
        x, _ = self.add_removed_tokens(x, indices, new_m)
        return (
            *self.split_and_expand_hwtc(x, h, w, t, s_t_c_g, sp_c_g, t_c_g, st_c_g),
            s_t_m,
            sp_m,
            t_m,
            st_m,
        )

    @classmethod
    def average_tokens(
        cls,
        s_t_x: torch.Tensor,
        sp_x: torch.Tensor,
        t_x: torch.Tensor,
        st_x: torch.Tensor,
        s_t_m: torch.Tensor,
        sp_m: torch.Tensor,
        t_m: torch.Tensor,
        st_m: torch.Tensor,
    ) -> torch.Tensor:
        """average_tokens."""
        x, m = cls.collapse_and_combine_hwtc(
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m
        )
        x, _, m = cls.remove_masked_tokens(x, m)
        x_for_mean = x * (1 - m.unsqueeze(-1))
        return x_for_mean.sum(dim=1) / torch.sum(1 - m, -1, keepdim=True)

    @classmethod
    def apply_mask_and_average_tokens_per_patch(
        cls,
        s_t_x: torch.Tensor,
        sp_x: torch.Tensor,
        t_x: torch.Tensor,
        st_x: torch.Tensor,
        s_t_m: torch.Tensor,
        sp_m: torch.Tensor,
        t_m: torch.Tensor,
        st_m: torch.Tensor,
    ) -> torch.Tensor:
        """apply_mask_and_average_tokens_per_patch."""
        s_t_x = rearrange(s_t_x, "b t_h t_w t c_g d -> b (t_h t_w) (t c_g) d")
        sp_x = rearrange(sp_x, "b t_h t_w c_g d -> b (t_h t_w) c_g d")
        # repeat time tokens over space
        t_x = repeat(
            rearrange(t_x, "b t c_g d -> b (t c_g) d"),
            "b n d -> b s n d",
            s=sp_x.shape[1],
        )
        st_x = repeat(st_x, "b c_g d -> b s c_g d", s=sp_x.shape[1])
        s_t_m = rearrange(s_t_m, "b t_h t_w t c_g-> b (t_h t_w) (t c_g)")
        sp_m = rearrange(sp_m, "b t_h t_w c_g-> b (t_h t_w) c_g")
        t_m = repeat(
            rearrange(t_m, "b t c_g -> b (t c_g)"), "b n -> b s n", s=sp_x.shape[1]
        )
        st_m = repeat(st_m, "b c_g -> b s c_g", s=sp_x.shape[1])

        x = torch.cat([s_t_x, sp_x, t_x, st_x], dim=2)  # B, S, N, D
        m = torch.cat([s_t_m, sp_m, t_m, st_m], dim=2)  # B, S, N

        x_for_mean = x * (1 - m.unsqueeze(-1))

        return x_for_mean.sum(dim=2) / torch.sum(1 - m, -1, keepdim=True)

    def create_token_exit_ids(
        self,
        s_t_x: torch.Tensor,
        sp_x: torch.Tensor,
        t_x: torch.Tensor,
        st_x: torch.Tensor,
        token_exit_cfg: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """create_token_exit_ids."""
        exit_s_t = torch.zeros_like(s_t_x)
        exit_sp = torch.zeros_like(sp_x)
        exit_t = torch.zeros_like(t_x)
        exit_st = torch.zeros_like(st_x)

        for idx, (key, _) in enumerate(self.space_time_groups.items()):
            exit_s_t[:, :, :, :, idx, :] = token_exit_cfg[key]

        for idx, (key, _) in enumerate(self.space_groups.items()):
            exit_sp[:, :, :, idx, :] = token_exit_cfg[key]

        for idx, (key, _) in enumerate(self.time_groups.items()):
            exit_t[:, :, idx, :] = token_exit_cfg[key]

        for idx, (key, _) in enumerate(self.static_groups.items()):
            exit_st[:, idx, :] = token_exit_cfg[key]
        return exit_s_t, exit_sp, exit_t, exit_st

    def forward(
        self,
        s_t_x: torch.Tensor,
        sp_x: torch.Tensor,
        t_x: torch.Tensor,
        st_x: torch.Tensor,
        s_t_m: torch.Tensor,
        sp_m: torch.Tensor,
        t_m: torch.Tensor,
        st_m: torch.Tensor,
        months: torch.Tensor,
        patch_size: int,
        input_resolution_m: int = BASE_GSD,
        exit_after: int | None = None,
        token_exit_cfg: dict | None = None,
        add_layernorm_on_exit: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Forward."""
        (
            s_t_x,
            sp_x,
            t_x,
            st_x,
            s_t_m,
            sp_m,
            t_m,
            st_m,
        ) = self.apply_linear_projection(
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, patch_size
        )

        if (exit_after is None) or (exit_after > 0):
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m = self.apply_attn(
                s_t_x,
                sp_x,
                t_x,
                st_x,
                s_t_m,
                sp_m,
                t_m,
                st_m,
                months,
                patch_size,
                input_resolution_m,
                exit_after=exit_after,
                token_exit_cfg=token_exit_cfg,
            )

        if add_layernorm_on_exit:
            s_t_x = self.norm(s_t_x)
            sp_x = self.norm(sp_x)
            t_x = self.norm(t_x)
            st_x = self.norm(st_x)

        return (s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months)

    @classmethod
    def load_from_folder(cls, folder: Path, device: torch.device) -> "Encoder":
        """Load a model from a folder containing an encoder.pt and config.json."""
        if not (folder / CONFIG_FILENAME).exists():
            all_files_in_folder = [f.name for f in folder.glob("*")]
            raise ValueError(
                f"Expected {CONFIG_FILENAME} in {folder}, found {all_files_in_folder}"
            )
        if not (folder / ENCODER_FILENAME).exists():
            all_files_in_folder = [f.name for f in folder.glob("*")]
            raise ValueError(
                f"Expected {ENCODER_FILENAME} in {folder}, found {all_files_in_folder}"
            )

        with (folder / CONFIG_FILENAME).open("r") as f:
            config = json.load(f)
            model_config = config["model"]
            encoder_config = model_config["encoder"]
        encoder = cls(**encoder_config)

        state_dict = torch.load(
            folder / ENCODER_FILENAME, map_location=device, weights_only=True
        )
        for key in list(state_dict.keys()):
            # this cleans the state dict, which occasionally had an extra
            # ".backbone" included in the key names
            state_dict[key.replace(".backbone", "")] = state_dict.pop(key)
        encoder.load_state_dict(state_dict)
        return encoder


class GalileoModel(nn.Module):
    """Galileo backbones."""

    input_keys = [
        "s1",
        "s2",
        "era5",
        "tc",
        "viirs",
        "srtm",
        "dw",
        "wc",
        "landscan",
        "latlon",
    ]

    def __init__(self, size: GalileoSize, patch_size: int = 4) -> None:
        """Initialize the Galileo model.

        Args:
            size: The size of the Galileo model.
            patch_size: The patch size to use.
        """
        super().__init__()
        _ = hf_hub_download(
            repo_id=HF_HUB_ID,
            filename=f"{pretrained_weights[size]}/{ENCODER_FILENAME}",
            revision="f039dd5dde966a931baeda47eb680fa89b253e4e",
        )
        config_file = hf_hub_download(
            repo_id=HF_HUB_ID,
            filename=f"{pretrained_weights[size]}/{CONFIG_FILENAME}",
            revision="f039dd5dde966a931baeda47eb680fa89b253e4e",
        )

        model_folder = Path(config_file).parent
        assert (model_folder / ENCODER_FILENAME).exists()
        assert (model_folder / CONFIG_FILENAME).exists()

        self.model = Encoder.load_from_folder(model_folder, device=torch.device("cpu"))

        self.s_t_channels_s2 = [
            idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S2" in key
        ]
        self.s_t_channels_s1 = [
            idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S1" in key
        ]

        self.patch_size = patch_size

    @staticmethod
    def to_cartesian(
        lat: float | np.ndarray | torch.Tensor, lon: float | np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Transform latitudes and longitudes to cartesian coordinates."""
        if isinstance(lat, float):
            assert -90 <= lat <= 90, (
                f"lat out of range ({lat}). Make sure you are in EPSG:4326"
            )
            assert -180 <= lon <= 180, (
                f"lon out of range ({lon}). Make sure you are in EPSG:4326"
            )
            assert isinstance(lon, float), f"Expected float got {type(lon)}"
            # transform to radians
            lat = lat * math.pi / 180
            lon = lon * math.pi / 180
            x = math.cos(lat) * math.cos(lon)
            y = math.cos(lat) * math.sin(lon)
            z = math.sin(lat)
            return np.array([x, y, z])
        elif isinstance(lon, np.ndarray):
            assert -90 <= lat.min(), (
                f"lat out of range ({lat.min()}). Make sure you are in EPSG:4326"
            )
            assert 90 >= lat.max(), (
                f"lat out of range ({lat.max()}). Make sure you are in EPSG:4326"
            )
            assert -180 <= lon.min(), (
                f"lon out of range ({lon.min()}). Make sure you are in EPSG:4326"
            )
            assert 180 >= lon.max(), (
                f"lon out of range ({lon.max()}). Make sure you are in EPSG:4326"
            )
            assert isinstance(lat, np.ndarray), f"Expected np.ndarray got {type(lat)}"
            # transform to radians
            lat = lat * math.pi / 180
            lon = lon * math.pi / 180
            x_np = np.cos(lat) * np.cos(lon)
            y_np = np.cos(lat) * np.sin(lon)
            z_np = np.sin(lat)
            return np.stack([x_np, y_np, z_np], axis=-1)
        elif isinstance(lon, torch.Tensor):
            assert -90 <= lat.min(), (
                f"lat out of range ({lat.min()}). Make sure you are in EPSG:4326"
            )
            assert 90 >= lat.max(), (
                f"lat out of range ({lat.max()}). Make sure you are in EPSG:4326"
            )
            assert -180 <= lon.min(), (
                f"lon out of range ({lon.min()}). Make sure you are in EPSG:4326"
            )
            assert 180 >= lon.max(), (
                f"lon out of range ({lon.max()}). Make sure you are in EPSG:4326"
            )
            assert isinstance(lat, torch.Tensor), (
                f"Expected torch.Tensor got {type(lat)}"
            )
            # transform to radians
            lat = lat * math.pi / 180
            lon = lon * math.pi / 180
            x_t = torch.cos(lat) * torch.cos(lon)
            y_t = torch.cos(lat) * torch.sin(lon)
            z_t = torch.sin(lat)
            return torch.stack([x_t, y_t, z_t], dim=-1)
        else:
            raise AssertionError(f"Unexpected input type {type(lon)}")

    @classmethod
    def construct_galileo_input(
        cls,
        s1: torch.Tensor | None = None,  # [H, W, T, D]
        s2: torch.Tensor | None = None,  # [H, W, T, D]
        era5: torch.Tensor | None = None,  # [T, D]
        tc: torch.Tensor | None = None,  # [T, D]
        viirs: torch.Tensor | None = None,  # [T, D]
        srtm: torch.Tensor | None = None,  # [H, W, D]
        dw: torch.Tensor | None = None,  # [H, W, D]
        wc: torch.Tensor | None = None,  # [H, W, D]
        landscan: torch.Tensor | None = None,  # [D]
        latlon: torch.Tensor | None = None,  # [D]
        months: torch.Tensor | None = None,  # [T]
        normalize: bool = False,
    ) -> MaskedOutput:
        """Construct a Galileo input."""
        space_time_inputs = [s1, s2]
        time_inputs = [era5, tc, viirs]
        space_inputs = [srtm, dw, wc]
        static_inputs = [landscan, latlon]
        devices = [
            x.device
            for x in space_time_inputs + time_inputs + space_inputs + static_inputs
            if x is not None
        ]

        if len(devices) == 0:
            raise ValueError("At least one input must be not None")
        if not all(devices[0] == device for device in devices):
            raise ValueError("Received tensors on multiple devices")
        device = devices[0]

        # first, check all the input shapes are consistent
        batch_list = (
            [x.shape[0] for x in space_time_inputs if x is not None]
            + [x.shape[0] for x in time_inputs if x is not None]
            + [x.shape[0] for x in space_inputs if x is not None]
            + [x.shape[0] for x in static_inputs if x is not None]
        )
        timesteps_list = [x.shape[3] for x in space_time_inputs if x is not None] + [
            x.shape[1] for x in time_inputs if x is not None
        ]
        height_list = [x.shape[1] for x in space_time_inputs if x is not None] + [
            x.shape[0] for x in space_inputs if x is not None
        ]
        width_list = [x.shape[2] for x in space_time_inputs if x is not None] + [
            x.shape[1] for x in space_inputs if x is not None
        ]
        if len(batch_list) > 0:
            if len(set(batch_list)) > 1:
                raise ValueError("Inconsistent number of batch sizes per input")
            b = batch_list[0]

        if len(timesteps_list) > 0:
            if not all(timesteps_list[0] == timestep for timestep in timesteps_list):
                raise ValueError("Inconsistent number of timesteps per input")
            t = timesteps_list[0]
        else:
            t = 1

        if len(height_list) > 0:
            if not all(height_list[0] == height for height in height_list):
                raise ValueError("Inconsistent heights per input")
            if not all(width_list[0] == width for width in width_list):
                raise ValueError("Inconsistent widths per input")
            h = height_list[0]
            w = width_list[0]
        else:
            h, w = 1, 1

        # now, we can construct our empty input tensors. By default, everything is masked
        s_t_x = torch.zeros(
            (b, h, w, t, len(SPACE_TIME_BANDS)), dtype=torch.float, device=device
        )
        s_t_m = torch.ones(
            (b, h, w, t, len(SPACE_TIME_BANDS_GROUPS_IDX)),
            dtype=torch.float,
            device=device,
        )
        sp_x = torch.zeros(
            (b, h, w, len(SPACE_BANDS)), dtype=torch.float, device=device
        )
        sp_m = torch.ones(
            (b, h, w, len(SPACE_BAND_GROUPS_IDX)), dtype=torch.float, device=device
        )
        t_x = torch.zeros((b, t, len(TIME_BANDS)), dtype=torch.float, device=device)
        t_m = torch.ones(
            (b, t, len(TIME_BAND_GROUPS_IDX)), dtype=torch.float, device=device
        )
        st_x = torch.zeros((b, len(STATIC_BANDS)), dtype=torch.float, device=device)
        st_m = torch.ones(
            (b, len(STATIC_BAND_GROUPS_IDX)), dtype=torch.float, device=device
        )

        for x, bands_list, group_key in zip(
            [s1, s2], [S1_BANDS, S2_BANDS], ["S1", "S2"]
        ):
            if x is not None:
                indices = [
                    idx for idx, val in enumerate(SPACE_TIME_BANDS) if val in bands_list
                ]
                groups_idx = [
                    idx
                    for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX)
                    if group_key in key
                ]
                s_t_x[:, :, :, :, indices] = x
                s_t_m[:, :, :, :, groups_idx] = 0

        for x, bands_list, group_key in zip(
            [srtm, dw, wc], [SRTM_BANDS, DW_BANDS, WC_BANDS], ["SRTM", "DW", "WC"]
        ):
            if x is not None:
                indices = [
                    idx for idx, val in enumerate(SPACE_BANDS) if val in bands_list
                ]
                groups_idx = [
                    idx
                    for idx, key in enumerate(SPACE_BAND_GROUPS_IDX)
                    if group_key in key
                ]
                sp_x[:, :, :, indices] = x
                sp_m[:, :, :, groups_idx] = 0

        for x, bands_list, group_key in zip(
            [era5, tc, viirs],
            [ERA5_BANDS, TC_BANDS, VIIRS_BANDS],
            ["ERA5", "TC", "VIIRS"],
        ):
            if x is not None:
                indices = [
                    idx for idx, val in enumerate(TIME_BANDS) if val in bands_list
                ]
                groups_idx = [
                    idx
                    for idx, key in enumerate(TIME_BAND_GROUPS_IDX)
                    if group_key in key
                ]
                t_x[:, :, indices] = x
                t_m[:, :, groups_idx] = 0

        for x, bands_list, group_key in zip(
            [landscan, latlon], [LANDSCAN_BANDS, LOCATION_BANDS], ["LS", "location"]
        ):
            if x is not None:
                if group_key == "location":
                    # transform latlon to cartesian
                    x = cast(torch.Tensor, cls.to_cartesian(x[0], x[1]))
                indices = [
                    idx for idx, val in enumerate(STATIC_BANDS) if val in bands_list
                ]
                groups_idx = [
                    idx
                    for idx, key in enumerate(STATIC_BAND_GROUPS_IDX)
                    if group_key in key
                ]
                st_x[indices] = x
                st_m[groups_idx] = 0

        if months is None:
            months = torch.ones((b, t), dtype=torch.long, device=device) * DEFAULT_MONTH
        else:
            if months.shape[1] != t:
                raise ValueError("Incorrect number of input months")

        if normalize:
            s_t_x = (
                torch.from_numpy(DEFAULT_NORMALIZER(s_t_x.cpu().numpy()))
                .to(device)
                .float()
            )
            sp_x = (
                torch.from_numpy(DEFAULT_NORMALIZER(sp_x.cpu().numpy()))
                .to(device)
                .float()
            )
            t_x = (
                torch.from_numpy(DEFAULT_NORMALIZER(t_x.cpu().numpy()))
                .to(device)
                .float()
            )
            st_x = (
                torch.from_numpy(DEFAULT_NORMALIZER(st_x.cpu().numpy()))
                .to(device)
                .float()
            )

        return MaskedOutput(
            s_t_x=s_t_x,
            s_t_m=s_t_m,
            sp_x=sp_x,
            sp_m=sp_m,
            t_x=t_x,
            t_m=t_m,
            st_x=st_x,
            st_m=st_m,
            months=months,
        )

    def forward(self, inputs: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Compute feature maps from the Galileo backbone.

        Inputs:
            inputs
        """
        stacked_inputs = {}
        for key in inputs[0].keys():
            # assume all the keys in an input are consistent
            if key in self.input_keys:
                stacked_inputs[key] = torch.stack([inp[key] for inp in inputs], dim=0)

        s_t_channels = []
        for space_time_modality in ["s1", "s2"]:
            if space_time_modality not in stacked_inputs:
                continue
            if space_time_modality == "s1":
                s_t_channels += self.s_t_channels_s1
            else:
                s_t_channels += self.s_t_channels_s2
            cur = stacked_inputs[space_time_modality]
            # Check if it's single or multitemporal, and reshape accordingly
            num_bands = len(S2_BANDS) if space_time_modality == "s2" else len(S1_BANDS)
            num_timesteps = cur.shape[1] // num_bands
            cur = rearrange(cur, "b (t c) h w -> b h w t c", t=num_timesteps)
            stacked_inputs[space_time_modality] = cur

        for time_modality in ["era5", "tc", "viirs"]:
            if time_modality not in stacked_inputs:
                continue
            cur = stacked_inputs[time_modality]
            # Check if it's single or multitemporal, and reshape accordingly
            num_bands = {
                "era5": len(ERA5_BANDS),
                "tc": len(TC_BANDS),
                "viirs": len(VIIRS_BANDS),
            }[time_modality]
            num_timesteps = cur.shape[1] // num_bands
            cur = rearrange(cur, "b (t c) -> b t c", t=num_timesteps)
            stacked_inputs[space_time_modality] = cur

        galileo_input = self.construct_galileo_input(**stacked_inputs, normalize=True)
        h = galileo_input.s_t_x.shape[1]
        if h < self.patch_size:
            logger.warning(
                f"Given patch size {self.patch_size} < h {h}. Reducing patch size to {h}"
            )
            patch_size = h
        else:
            patch_size = self.patch_size

        outputs = self.model(
            s_t_x=galileo_input.s_t_x,
            s_t_m=galileo_input.s_t_m,
            sp_x=galileo_input.sp_x,
            sp_m=galileo_input.sp_m,
            t_x=galileo_input.t_x,
            t_m=galileo_input.t_m,
            st_x=galileo_input.st_x,
            st_m=galileo_input.st_m,
            months=galileo_input.months,
            patch_size=patch_size,
        )
        if h == patch_size:
            # only one spatial patch, so we can just take an average
            # of all the tokens to output b c_g 1 1
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, _ = outputs
            averaged = self.model.average_tokens(
                s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m
            )
            return [repeat(averaged, "b d -> b d 1 1")]
        else:
            s_t_x = outputs[0]
            # we will be assuming we only want s_t_x, and (for now) that we want s1 or s2 bands
            # s_t_x has shape [b, h, w, t, c_g, d]
            # and we want [b, d, h, w]
            return [
                rearrange(
                    s_t_x[:, :, :, :, s_t_channels, :].mean(dim=3),
                    "b h w c_g d -> b c_g d h w",
                ).mean(dim=1)
            ]
