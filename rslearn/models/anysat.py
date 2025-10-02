"""AnySat model."""

import math
from typing import Any

import torch
from einops import rearrange

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


class AnySat(torch.nn.Module):
    """AnySat backbone (outputs one feature map)."""

    def __init__(
        self,
        modalities: list[str],
        patch_size_meters: int,
        dates: dict[str, list[int]],
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
            patch_size_meters: patch size in meters (must be multiple of 10).
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

        if not all(m in TIME_SERIES_MODALITIES for m in dates.keys()):
            raise ValueError("`dates` keys must be time-series modalities only.")
        for m in modalities:
            if m in TIME_SERIES_MODALITIES and m not in dates:
                raise ValueError(
                    f"Missing required dates for time-series modality '{m}'."
                )

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
        self.dates = dates
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
        # Assuming all batches have the same spatial shapes, adjust patch size only once
        self.adjusted_patch_size_meters = 0

    @staticmethod
    def _ceil_to_multiple(x: int, base: int) -> int:
        """Round x up to nearest multiple of base."""
        return int(((x + base - 1) // base) * base)

    def _update_effective_patch_size_meters(
        self, spatial_shapes: dict[str, tuple[int, int]]
    ) -> None:
        """Update self.patch_size_meters to ensure ≤ 32 * 32 patches for all modalities.

        As noted in the AnySat repo: in general, avoid having more than 1024 patches per tile.
        Equivalent to ensure:
          ps_m >= res_m * max(H_m, W_m) / 32
        Take max across modalities, and round up to nearest 10 m.

        Args:
            spatial_shapes: dict mapping modality to (H, W) of that modality in the batch.
        """
        required_ps_m = self.patch_size_meters
        for m, (H, W) in spatial_shapes.items():
            res_m = MODALITY_RESOLUTIONS[m]
            need_m = math.ceil((res_m * max(H, W)) / 32.0)
            if need_m > required_ps_m:
                required_ps_m = need_m
        self.adjusted_patch_size_meters = self._ceil_to_multiple(required_ps_m, 10)

    def forward(self, inputs: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Forward pass for the AnySat model.

        Args:
            inputs: input dicts that must include modalities as keys which are defined in the self.modalities list

        Returns:
            List[torch.Tensor]: Single-scale feature tensors from the encoder.
        """
        if not inputs:
            raise ValueError("empty inputs")

        batch: dict[str, torch.Tensor] = {}
        spatial_shapes: dict[str, tuple[int, int]] = {}
        spatial_extent: tuple[float, float] | None = None

        for modality in self.modalities:
            if modality not in inputs[0]:
                raise ValueError(f"Modality '{modality}' not present in inputs.")

            cur = torch.stack(
                [inp[modality] for inp in inputs], dim=0
            )  # (B, C, H, W) or (B, T*C, H, W)

            if modality in TIME_SERIES_MODALITIES:
                num_dates = len(self.dates[modality])
                num_bands = cur.shape[1] // num_dates
                cur = rearrange(
                    cur, "b (t c) h w -> b t c h w", t=num_dates, c=num_bands
                )
                H, W = cur.shape[-2], cur.shape[-1]
            else:
                num_bands = cur.shape[1]
                H, W = cur.shape[-2], cur.shape[-1]

            if num_bands != len(MODALITY_BANDS[modality]):
                raise ValueError(
                    f"Modality '{modality}' expected {len(MODALITY_BANDS[modality])} bands, "
                    f"got {num_bands} (shape {tuple(cur.shape)})"
                )

            batch[modality] = cur
            spatial_shapes[modality] = (H, W)

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

        if self.adjusted_patch_size_meters == 0:
            self._update_effective_patch_size_meters(spatial_shapes)

        # Add *_dates
        to_add = {}
        for modality, x in list(batch.items()):
            if modality in TIME_SERIES_MODALITIES:
                B, T = x.shape[0], x.shape[1]
                d = torch.as_tensor(
                    self.dates[modality], dtype=torch.long, device=x.device
                )
                if d.ndim != 1 or d.numel() != T:
                    raise ValueError(
                        f"dates for '{modality}' must be 1D length {T}, got {tuple(d.shape)}"
                    )
                to_add[f"{modality}_dates"] = d.unsqueeze(0).repeat(B, 1)

        batch.update(to_add)

        kwargs = {"patch_size": self.adjusted_patch_size_meters, "output": self.output}
        if self.output == "dense":
            kwargs["output_modality"] = self.output_modality

        features = self.model(batch, **kwargs)
        return [rearrange(features, "b h w d -> b d h w")]

    def get_backbone_channels(self) -> list:
        """Returns the output channels of this model when used as a backbone.

        The output channels is a list of (patch_size, depth) that corresponds
        to the feature maps that the backbone returns.

        Returns:
            the output channels of the backbone as a list of (patch_size, depth) tuples.
        """
        if self.output == "patch":
            return [(self.adjusted_patch_size_meters // 10, 768)]
        elif self.output == "dense":
            return [(1, 1536)]
        else:
            raise ValueError(f"invalid output type: {self.output}")
