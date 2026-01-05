"""Galileo models."""

import math
import tempfile
from contextlib import nullcontext
from datetime import datetime
from enum import StrEnum
from typing import cast

import numpy as np
import torch
from einops import rearrange, repeat
from huggingface_hub import hf_hub_download
from upath import UPath

from rslearn.log_utils import get_logger
from rslearn.models.component import FeatureExtractor, FeatureMaps
from rslearn.models.galileo.single_file_galileo import (
    CONFIG_FILENAME,
    DW_BANDS,
    ENCODER_FILENAME,
    ERA5_BANDS,
    LANDSCAN_BANDS,
    LOCATION_BANDS,
    S1_BANDS,
    S2_BANDS,
    SPACE_BAND_GROUPS_IDX,
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
    SRTM_BANDS,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
    TC_BANDS,
    TIME_BAND_GROUPS_IDX,
    TIME_BANDS,
    VIIRS_BANDS,
    WC_BANDS,
    Encoder,
    MaskedOutput,
    Normalizer,
)
from rslearn.train.model_context import ModelContext

logger = get_logger(__name__)


HF_HUB_ID = "nasaharvest/galileo"
DEFAULT_MONTH = 5


# Galileo provides three sizes: nano, tiny, base
class GalileoSize(StrEnum):
    """Size of the Galileo model."""

    NANO = "nano"
    TINY = "tiny"
    BASE = "base"


pretrained_weights: dict[GalileoSize, str] = {
    GalileoSize.NANO: "models/nano",
    GalileoSize.TINY: "models/tiny",
    GalileoSize.BASE: "models/base",
}

DEFAULT_NORMALIZER = Normalizer()

AUTOCAST_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class GalileoModel(FeatureExtractor):
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

    def __init__(
        self,
        size: GalileoSize,
        patch_size: int = 4,
        pretrained_path: str | UPath | None = None,
        autocast_dtype: str | None = "bfloat16",
    ) -> None:
        """Initialize the Galileo model.

        Args:
            size: The size of the Galileo model.
            patch_size: The patch size to use.
            pretrained_path: the local path to the pretrained weights. Otherwise it is
                downloaded and cached in temp directory.
            autocast_dtype: which dtype to use for autocasting, or set None to disable.
        """
        super().__init__()
        if pretrained_path is None:
            pretrained_path = UPath(tempfile.gettempdir(), "rslearn_cache", "galileo")

        pretrained_path_for_size = UPath(pretrained_path) / pretrained_weights[size]
        if not (pretrained_path_for_size / CONFIG_FILENAME).exists():
            _ = hf_hub_download(
                local_dir=pretrained_path,
                repo_id=HF_HUB_ID,
                filename=f"{pretrained_weights[size]}/{CONFIG_FILENAME}",
                revision="f039dd5dde966a931baeda47eb680fa89b253e4e",
            )
        if not (pretrained_path_for_size / ENCODER_FILENAME).exists():
            _ = hf_hub_download(
                local_dir=pretrained_path,
                repo_id=HF_HUB_ID,
                filename=f"{pretrained_weights[size]}/{ENCODER_FILENAME}",
                revision="f039dd5dde966a931baeda47eb680fa89b253e4e",
            )

        assert (pretrained_path_for_size / ENCODER_FILENAME).exists()
        assert (pretrained_path_for_size / CONFIG_FILENAME).exists()

        self.model = Encoder.load_from_folder(
            pretrained_path_for_size, device=torch.device("cpu")
        )

        self.s_t_channels_s2 = [
            idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S2" in key
        ]
        self.s_t_channels_s1 = [
            idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S1" in key
        ]

        self.size = size
        self.patch_size = patch_size

        if autocast_dtype is not None:
            self.autocast_dtype = AUTOCAST_DTYPE_MAP[autocast_dtype]
        else:
            self.autocast_dtype = None

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
            x.shape[1] for x in space_inputs if x is not None
        ]
        width_list = [x.shape[2] for x in space_time_inputs if x is not None] + [
            x.shape[2] for x in space_inputs if x is not None
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
                    x = cast(torch.Tensor, cls.to_cartesian(x[:, 0], x[:, 1]))
                indices = [
                    idx for idx, val in enumerate(STATIC_BANDS) if val in bands_list
                ]
                groups_idx = [
                    idx
                    for idx, key in enumerate(STATIC_BAND_GROUPS_IDX)
                    if group_key in key
                ]
                st_x[:, indices] = x
                st_m[:, groups_idx] = 0

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

    @staticmethod
    def time_ranges_to_timestamps(
        time_ranges: list[tuple[datetime, datetime]],
        device: torch.device,
    ) -> torch.Tensor:
        """Turn the time ranges stored in a RasterImage to timestamps accepted by Galileo.

        Galileo only uses the month associated with each timestamp, so we take the midpoint
        the time range. For some inputs (e.g. Sentinel 2) we take an image from a specific
        time so that start_time == end_time == mid_time.
        """
        mid_ranges = [t[0] + ((t[1] - t[0]) / 2) for t in time_ranges]
        # months are indexed 0-11
        return torch.tensor(
            [d.month - 1 for d in mid_ranges], dtype=torch.int32, device=device
        )

    def forward(self, context: ModelContext) -> FeatureMaps:
        """Compute feature maps from the Galileo backbone.

        Args:
            context: the model context. Input dicts should contain keys corresponding to Galileo.input_keys
                (also documented below) and values are tensors of the following shapes,
                per input key:
                    "s1": B C T H W
                    "s2": B C T H W
                    "era5": B C T H W  (we will average over the H, W dimensions)
                    "tc": B C T H W  (we will average over the H, W dimensions)
                    "viirs": B C T H W  (we will average over the H, W dimensions)
                    "srtm": B C 1 H W (SRTM has no temporal dimension)
                    "dw": : B C 1 H W (Dynamic World should be averaged over time)
                    "wc": B C 1 H W (WorldCereal has no temporal dimension)
                    "landscan":  B C 1 H W  (we will average over the H, W dimensions)
                    "latlon":  B C 1 H W  (we will average over the H, W dimensions)

        The output will be an embedding representing the pooled tokens. If there is
        only a single token per h/w dimension (i.e. patch_size == h,w), then we will take
        a pool of all the unmasked tokens.

        If there are many spatial tokens per h/w dimension (patch_size > h,w), then we will
        take a pool of the space_time unmasked tokens (i.e. of the s1 and s2 tokens).
        """
        space_time_modalities = ["s1", "s2"]
        time_modalities = ["era5", "tc", "viirs"]
        stacked_inputs = {}
        months: torch.Tensor | None = None
        for key in context.inputs[0].keys():
            # assume all the keys in an input are consistent
            if key in self.input_keys:
                stacked_inputs[key] = torch.stack(
                    [inp[key].image for inp in context.inputs], dim=0
                )
                if key in space_time_modalities + time_modalities:
                    if months is None:
                        if context.inputs[0][key].timestamps is not None:
                            months = torch.stack(
                                [
                                    self.time_ranges_to_timestamps(
                                        inp[key].timestamps,  # type: ignore
                                        device=stacked_inputs[key].device,
                                    )
                                    for inp in context.inputs
                                ],
                                dim=0,
                            )

        if months is not None:
            stacked_inputs["months"] = months

        s_t_channels = []
        for space_time_modality in space_time_modalities:
            if space_time_modality not in stacked_inputs:
                continue
            if space_time_modality == "s1":
                s_t_channels += self.s_t_channels_s1
            else:
                s_t_channels += self.s_t_channels_s2
            cur = stacked_inputs[space_time_modality]
            cur = rearrange(cur, "b c t h w -> b h w t c")
            stacked_inputs[space_time_modality] = cur

        for space_modality in ["srtm", "dw", "wc"]:
            if space_modality not in stacked_inputs:
                continue
            # take the first (and assumed only) timestep
            stacked_inputs[space_modality] = stacked_inputs[space_modality][:, :, 0]
            stacked_inputs[space_modality] = rearrange(
                stacked_inputs[space_modality], "b c h w -> b h w c"
            )

        for time_modality in time_modalities:
            if time_modality not in stacked_inputs:
                continue
            cur = stacked_inputs[time_modality]
            # take the average over the h, w bands since Galileo
            # treats it as a pixel-timeseries
            cur = rearrange(
                torch.nanmean(cur, dim=(-1, -2)),
                "b c t -> b t c",
            )
            stacked_inputs[time_modality] = cur

        for static_modality in ["landscan", "latlon"]:
            if static_modality not in stacked_inputs:
                continue
            cur = stacked_inputs[static_modality]
            stacked_inputs[static_modality] = torch.nanmean(cur, dim=(2, 3, 4))

        galileo_input = self.construct_galileo_input(**stacked_inputs, normalize=True)
        h = galileo_input.s_t_x.shape[1]
        if h < self.patch_size:
            logger.warning(
                f"Given patch size {self.patch_size} < h {h}. Reducing patch size to {h}"
            )
            patch_size = h
        else:
            patch_size = self.patch_size

        # Decide context based on self.autocast_dtype.
        device = galileo_input.s_t_x.device
        if self.autocast_dtype is None:
            torch_context = nullcontext()
        else:
            assert device is not None
            torch_context = torch.amp.autocast(
                device_type=device.type, dtype=self.autocast_dtype
            )
        with torch_context:
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
            return FeatureMaps([repeat(averaged, "b d -> b d 1 1")])
        else:
            s_t_x = outputs[0]
            # we will be assuming we only want s_t_x, and (for now) that we want s1 or s2 bands
            # s_t_x has shape [b, h, w, t, c_g, d]
            # and we want [b, d, h, w]
            return FeatureMaps(
                [
                    rearrange(
                        s_t_x[:, :, :, :, s_t_channels, :].mean(dim=3),
                        "b h w c_g d -> b c_g d h w",
                    ).mean(dim=1)
                ]
            )

    def get_backbone_channels(self) -> list:
        """Returns the output channels of this model when used as a backbone.

        The output channels is a list of (patch_size, depth) that corresponds
        to the feature maps that the backbone returns.

        Returns:
            the output channels of the backbone as a list of (patch_size, depth) tuples.
        """
        if self.size == GalileoSize.BASE:
            depth = 768
        elif self.model_size == GalileoSize.TINY:
            depth = 192
        elif self.model_size == GalileoSize.NANO:
            depth = 128
        else:
            raise ValueError(f"Invalid model size: {self.size}")
        return [(self.patch_size, depth)]
