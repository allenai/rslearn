"""Presto wrapper to ingest Masked Helios Samples."""

import logging
import tempfile
from typing import Any

import torch
from einops import rearrange, repeat
from huggingface_hub import hf_hub_download
from torch import nn
from upath import UPath

from rslearn.models.presto.single_file_presto import (
    ERA5_BANDS,
    NUM_DYNAMIC_WORLD_CLASSES,
    PRESTO_ADD_BY,
    PRESTO_BANDS,
    PRESTO_DIV_BY,
    PRESTO_S1_BANDS,
    PRESTO_S2_BANDS,
    SRTM_BANDS,
)
from rslearn.models.presto.single_file_presto import Presto as SFPresto

logger = logging.getLogger(__name__)

INPUT_PRESTO_BANDS = [b for b in PRESTO_BANDS if b != "B09"]
INPUT_PRESTO_S2_BANDS = [b for b in PRESTO_S2_BANDS if b != "B09"]

PRESTO_S1_SUBTRACT_VALUE = -25.0
PRESTO_S1_DIV_VALUE = 25.0
PRESTO_S2_SUBTRACT_VALUE = 0.0
PRESTO_S2_DIV_VALUE = 1e4

HF_HUB_ID = "nasaharvest/presto"
MODEL_FILENAME = "default_model.pt"


class Presto(nn.Module):
    """Presto."""

    input_keys = [
        "s1",
        "s2",
        "era5",
        "srtm",
        "dynamic_world",
        "latlon",
    ]

    def __init__(
        self,
        pretrained_path: str | UPath | None = None,
        pixel_batch_size: int = 128,
    ):
        """Initialize the Presto wrapper.

        Args:
            pretrained_path: The directory to load from
            pixel_batch_size: If the input has a h,w dimension >1, this is
                flattened into a batch dimension (b h w) before being passed
                to the model (since Presto is designed for pixel timeseries).
        """
        super().__init__()

        if pretrained_path is None:
            pretrained_path = UPath(tempfile.gettempdir(), "rslearn_cache", "presto")
        if not (UPath(pretrained_path) / MODEL_FILENAME).exists():
            _ = hf_hub_download(
                local_dir=UPath(pretrained_path),
                repo_id=HF_HUB_ID,
                filename=MODEL_FILENAME,
                # pin the model to a specific hugging face commit
                revision="1b97f885969da4e2d5834ca8c92707c737911464",
            )

        model = SFPresto.construct()
        model.load_state_dict(
            torch.load(
                UPath(pretrained_path) / MODEL_FILENAME,
                map_location="cpu",
                weights_only=True,
            )
        )
        self.pixel_batch_size = pixel_batch_size
        self.model = model.encoder
        self.month = 6  # default month

    def construct_presto_input(
        self,
        s1: torch.Tensor | None = None,
        s1_bands: torch.Tensor | None = None,
        s2: torch.Tensor | None = None,
        s2_bands: torch.Tensor | None = None,
        era5: torch.Tensor | None = None,
        era5_bands: torch.Tensor | None = None,
        srtm: torch.Tensor | None = None,
        srtm_bands: torch.Tensor | None = None,
        dynamic_world: torch.Tensor | None = None,
        months: torch.Tensor | None = None,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inputs are paired into a tensor input <X> and a list <X>_bands, which describes <X>.

        <X> should have shape (b, num_timesteps, h, w len(<X>_bands)), with the following bands for
        each input:

        s1: ["VV", "VH"]
        s2: ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
        era5: ["temperature_2m", "total_precipitation"]
            "temperature_2m": Temperature of air at 2m above the surface of land,
                sea or in-land waters in Kelvin (K)
            "total_precipitation": Accumulated liquid and frozen water, including rain and snow,
                that falls to the Earth's surface. Measured in metres (m)
        srtm: ["elevation", "slope"]

        dynamic_world is a 1d input of shape (num_timesteps,) representing the dynamic world classes
            of each timestep for that pixel
        """
        bs = [x.shape[0] for x in [s1, s2, era5, srtm] if x is not None]
        hs = [x.shape[2] for x in [s1, s2, era5, srtm] if x is not None]
        ws = [x.shape[3] for x in [s1, s2, era5, srtm] if x is not None]
        devices = [x.device for x in [s1, s2, era5, srtm] if x is not None]

        assert len(set(bs)) == 1
        assert len(set(hs)) == 1
        assert len(set(ws)) == 1
        assert len(set(devices)) == 1
        b, h, w, device = bs[0], hs[0], ws[0], devices[0]

        # these values will be initialized as
        # we iterate through the data
        x: torch.Tensor | None = None
        mask: torch.Tensor | None = None
        t: int | None = None

        for band_group in [
            (s1, s1_bands),
            (s2, s2_bands),
            (era5, era5_bands),
            (srtm, srtm_bands),
        ]:
            data, input_bands = band_group
            if data is not None:
                assert input_bands is not None
            else:
                continue

            m_t = data.shape[1] // len(input_bands)
            if t is None:
                t = m_t
            else:
                if t != m_t:
                    raise ValueError("inconsistent values for t")

            data = rearrange(data, "b (t c) h w -> b t h w c", t=m_t)
            if x is None:
                x = torch.zeros(b, t, h, w, len(INPUT_PRESTO_BANDS), device=device)
            if mask is None:
                mask = torch.ones(b, t, h, w, len(INPUT_PRESTO_BANDS), device=device)

            # construct a mapping from the input bands to the presto input bands
            input_to_output_mapping = [
                INPUT_PRESTO_BANDS.index(val) for val in input_bands
            ]
            x[:, :, :, :, input_to_output_mapping] = data
            mask[:, :, :, :, input_to_output_mapping] = 0

        assert x is not None
        assert mask is not None
        assert t is not None

        if dynamic_world is None:
            dynamic_world = (
                torch.ones(b, t, h, w, device=device) * NUM_DYNAMIC_WORLD_CLASSES
            )

        if months is None:
            months = torch.ones((b, t), device=device) * self.month
        else:
            assert months.shape[-1] == t

        if normalize:
            x = (x + PRESTO_ADD_BY.to(device=device)) / PRESTO_DIV_BY.to(device=device)
        return x, mask, dynamic_world.long(), months.long()

    def forward(self, inputs: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Compute feature maps from the Presto backbone.

        Inputs:
            inputs
        """
        stacked_inputs = {}
        latlons: torch.Tensor | None = None
        for key in inputs[0].keys():
            # assume all the keys in an input are consistent
            if key in self.input_keys:
                if key == "latlon":
                    latlons = torch.stack([inp[key] for inp in inputs], dim=0)
                else:
                    stacked_inputs[key] = torch.stack(
                        [inp[key] for inp in inputs], dim=0
                    )

        (
            x,
            mask,
            dynamic_world,
            months,
        ) = self.construct_presto_input(
            **stacked_inputs,
            s1_bands=PRESTO_S1_BANDS,
            s2_bands=INPUT_PRESTO_S2_BANDS,
            era5_bands=ERA5_BANDS,
            srtm_bands=SRTM_BANDS,
            normalize=True,
        )
        b, _, h, w, _ = x.shape

        output_features = torch.zeros(
            b * h * w, self.model.embedding_size, device=x.device
        )

        x = rearrange(x, "b t h w d -> (b h w) t d")
        mask = rearrange(mask, "b t h w d -> (b h w) t d")
        dynamic_world = rearrange(dynamic_world, "b t h w -> (b h w) t")
        months = repeat(months, "b t -> (b h w) t", h=h, w=w)
        if latlons is not None:
            latlons = rearrange(latlons, "b c h w -> (b h w) c")

        for batch_idx in range(0, b * h * w, self.pixel_batch_size):
            x_b = x[batch_idx : batch_idx + self.pixel_batch_size]
            mask_b = mask[batch_idx : batch_idx + self.pixel_batch_size]
            dw = dynamic_world[batch_idx : batch_idx + self.pixel_batch_size]
            months_b = months[batch_idx : batch_idx + self.pixel_batch_size]
            if latlons is not None:
                l_b = latlons[batch_idx : batch_idx + self.pixel_batch_size]
            else:
                l_b = None
            output_b = self.model(
                x=x_b,
                dynamic_world=dw,
                mask=mask_b,
                month=months_b,
                latlons=l_b,
                eval_task=True,
            )
            output_features[batch_idx : batch_idx + self.pixel_batch_size] = output_b

        return [rearrange(output_features, "(b h w) d -> b d h w", h=h, w=w, b=b)]
