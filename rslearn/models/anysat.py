"""AnySat model.

This code loads the AnySat model from torch hub. See
https://github.com/gastruc/AnySat for applicable license and copyright information.
"""

from datetime import datetime

import torch
from einops import rearrange

from rslearn.train.model_context import ModelContext

from .component import FeatureExtractor, FeatureMaps

# AnySat github: https://github.com/gastruc/AnySat
# Modalities and expected resolutions (meters)
MODALITY_RESOLUTIONS: dict[str, float] = {
    "aerial": 0.2,
    "aerial-flair": 0.2,
    "spot": 1,
    "naip": 1.25,
    "s2": 10,
    "s1-asc": 10,
    "s1": 10,
    "alos": 30,
    "l7": 30,
    "l8": 10,  # L8 must be upsampled to 10 m in AnySat
    "modis": 250,
}

# Modalities and expected band names
MODALITY_BANDS: dict[str, list[str]] = {
    "aerial": ["R", "G", "B", "NiR"],
    "aerial-flair": ["R", "G", "B", "NiR", "Elevation"],
    "spot": ["R", "G", "B"],
    "naip": ["R", "G", "B", "NiR"],
    "s2": ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8a", "B11", "B12"],
    "s1-asc": ["VV", "VH"],
    "s1": ["VV", "VH", "Ratio"],
    "alos": ["HH", "HV", "Ratio"],
    "l7": ["B1", "B2", "B3", "B4", "B5", "B7"],
    "l8": ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"],
    "modis": ["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
}

# Modalities that require *_dates* input
TIME_SERIES_MODALITIES = {"s2", "s1-asc", "s1", "alos", "l7", "l8", "modis"}


class AnySat(FeatureExtractor):
    """AnySat backbone (outputs one feature map)."""

    def __init__(
        self,
        modalities: list[str],
        patch_size_meters: int,
        output: str = "patch",
        output_modality: str | None = None,
        hub_repo: str = "gastruc/anysat",
        pretrained: bool = True,
        force_reload: bool = False,
        flash_attn: bool = False,
    ) -> None:
        """Initialize an AnySat model.

        Args:
            modalities: list of modalities to use as input (1 or more).
            patch_size_meters: patch size in meters (must be multiple of 10). Avoid having more than 1024 patches per tile
                ie, the height/width in meters should be <= 32 * patch_size_meters.
            dates: dict mapping time-series modalities to list of dates (day number in a year, 0-255).
            output: 'patch' (default) or 'dense'. Use 'patch' for classification tasks,
                'dense' for segmentation tasks.
            output_modality: required if output='dense', specifies which modality to use
                for the dense output (one of the input modalities).
            hub_repo: torch.hub repository to load AnySat from.
            pretrained: whether to load pretrained weights.
            force_reload: whether to force re-download of the model.
            flash_attn: whether to use flash attention (if available).
        """
        super().__init__()

        if not modalities:
            raise ValueError("At least one modality must be specified.")
        for m in modalities:
            if m not in MODALITY_RESOLUTIONS:
                raise ValueError(f"Invalid modality: {m}")

        if patch_size_meters % 10 != 0:
            raise ValueError(
                "In AnySat, `patch_size` is in meters and must be a multiple of 10."
            )

        output = output.lower()
        if output not in {"patch", "dense"}:
            raise ValueError("`output` must be 'patch' or 'dense'.")
        if output == "dense" and output_modality is None:
            raise ValueError("`output_modality` is required when output='dense'.")

        self.modalities = modalities
        self.patch_size_meters = int(patch_size_meters)
        self.output = output
        self.output_modality = output_modality

        self.model = torch.hub.load(  # nosec B614
            hub_repo,
            "anysat",
            pretrained=pretrained,
            force_reload=force_reload,
            flash_attn=flash_attn,
        )
        self._embed_dim = 768  # base width, 'dense' returns 2x

    @staticmethod
    def time_ranges_to_doy(
        time_ranges: list[tuple[datetime, datetime]],
        device: torch.device,
    ) -> torch.Tensor:
        """Turn the time ranges stored in a RasterImage to timestamps accepted by AnySat.

        AnySat uses the doy with each timestamp, so we take the midpoint
        the time range. For some inputs (e.g. Sentinel 2) we take an image from a specific
        time so that start_time == end_time == mid_time.
        """
        doys = [(t[0] + ((t[1] - t[0]) / 2)).timetuple().tm_yday for t in time_ranges]
        return torch.tensor(doys, dtype=torch.int32, device=device)

    def forward(self, context: ModelContext) -> FeatureMaps:
        """Forward pass for the AnySat model.

        Args:
            context: the model context. Input dicts must include modalities as keys
                which are defined in the self.modalities list

        Returns:
            a FeatureMaps with one feature map at the configured patch size.
        """
        inputs = context.inputs

        batch: dict[str, torch.Tensor] = {}
        spatial_extent: tuple[float, float] | None = None

        for modality in self.modalities:
            if modality not in inputs[0]:
                raise ValueError(f"Modality '{modality}' not present in inputs.")

            cur = torch.stack(
                [inp[modality].image for inp in inputs], dim=0
            )  # (B, C, T, H, W)

            if modality in TIME_SERIES_MODALITIES:
                num_bands = cur.shape[1]
                cur = rearrange(cur, "b c t h w -> b t c h w")
                H, W = cur.shape[-2], cur.shape[-1]

                if inputs[0][modality].timestamps is None:
                    raise ValueError(
                        f"Require timestamps for time series modality {modality}"
                    )
                timestamps = torch.stack(
                    [
                        self.time_ranges_to_doy(inp[modality].timestamps, cur.device)  # type: ignore
                        for inp in inputs
                    ],
                    dim=0,
                )
                batch[f"{modality}_dates"] = timestamps
            else:
                # take the first (assumed only) timestep
                cur = cur[:, :, 0]
                num_bands = cur.shape[1]
                H, W = cur.shape[-2], cur.shape[-1]

            if num_bands != len(MODALITY_BANDS[modality]):
                raise ValueError(
                    f"Modality '{modality}' expected {len(MODALITY_BANDS[modality])} bands, "
                    f"got {num_bands} (shape {tuple(cur.shape)})"
                )

            batch[modality] = cur

            # Ensure same spatial extent across all modalities (H*res, W*res)
            extent = (
                H * MODALITY_RESOLUTIONS[modality],
                W * MODALITY_RESOLUTIONS[modality],
            )
            if spatial_extent is None:
                spatial_extent = extent
            elif spatial_extent != extent:
                raise ValueError(
                    "All modalities must share the same spatial extent (H*res, W*res)."
                )

        kwargs = {"patch_size": self.patch_size_meters, "output": self.output}
        if self.output == "dense":
            kwargs["output_modality"] = self.output_modality

        features = self.model(batch, **kwargs)
        return FeatureMaps([rearrange(features, "b h w d -> b d h w")])

    def get_backbone_channels(self) -> list:
        """Returns the output channels of this model when used as a backbone.

        The output channels is a list of (patch_size, depth) that corresponds
        to the feature maps that the backbone returns.

        Returns:
            the output channels of the backbone as a list of (patch_size, depth) tuples.
        """
        if self.output == "patch":
            return [(self.patch_size_meters // 10, 768)]
        elif self.output == "dense":
            return [(1, 1536)]
        else:
            raise ValueError(f"invalid output type: {self.output}")
