"""AnySat mode."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rslearn.train.transforms.transform import Transform

ANYSAT_SUPPORTED = [
    # time series
    "s2",
    "s1",
    "s1-asc",
    "l7",
    "l8",
    "modis",
    "alos",
    # single-date / VHR
    "aerial",
    "aerial-flair",
    "spot",
    "naip",
]
PATCH_SIZE = 1  # dense per-pixel features


class AnySat(nn.Module):
    """AnySat backbone."""

    def __init__(self, modalities: list[str] = ["s2"]) -> None:
        """Initialize the AnySat model."""
        super().__init__()
        for m in modalities:
            if m not in ANYSAT_SUPPORTED:
                raise ValueError(f"Invalid modality: {m}")
        self.modalities = list(modalities)

        self.model = torch.hub.load(  # nosec B614
            "gastruc/anysat",
            "anysat",
            pretrained=True,
            flash_attn=False,
            force_reload=False,
        )
        self._last_depth: int | None = None

    @staticmethod
    def _calc_patch_size(h_pixels: int) -> int:
        """Calculate patch size based on image height in pixels."""
        h_in_m = h_pixels * 10
        return min(40, h_in_m)

    def _stack_imgs(self, items: list[torch.Tensor]) -> torch.Tensor:
        """Stack image tensors into a batch."""
        x0 = items[0]
        if x0.dim() == 3:
            # (C, H, W) -> (B, C, H, W)
            if not all(x.dim() == 3 and x.shape == x0.shape for x in items):
                raise ValueError("All single-date tensors must share (C, H, W)")
            return torch.stack(items, dim=0)
        elif x0.dim() == 4:
            # (T, C, H, W) -> (B, T, C, H, W)
            T, C, H, W = x0.shape
            if not all(x.dim() == 4 and x.shape == (T, C, H, W) for x in items):
                raise ValueError("All time-series tensors must share (T, C, H, W)")
            return torch.stack(items, dim=0)
        else:
            raise ValueError("Expected (C, H, W) or (T, C, H, W)")

    def _stack_dates(self, items: list[torch.Tensor], B: int, T: int) -> torch.Tensor:
        """Stack date tensors into a batch."""
        fixed = []
        for d in items:
            if d.dim() == 1:  # (T,)
                d = d.unsqueeze(0)  # (1, T)
            if d.size(-1) != T:
                raise ValueError("All *_dates must share T")
            fixed.append(d)
        d0 = fixed[0]
        out = (
            torch.cat(fixed, dim=0)
            if d0.size(0) == 1
            else torch.stack(fixed, dim=0).squeeze(1)
        )
        if out.shape != (B, T):
            raise ValueError("Batched *_dates must be (B,T)")
        return out.long()

    def forward(self, inputs: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Forward pass for the AnySat model."""
        if not inputs:
            raise ValueError("inputs must be non-empty")

        batch: dict[str, torch.Tensor] = {}
        date_keys: dict[str, torch.Tensor] = {}
        present: list[str] = []

        for m in self.modalities:
            if m not in inputs[0]:
                continue
            imgs = [s[m] for s in inputs]
            x = self._stack_imgs(imgs)
            batch[m] = x
            present.append(m)

            # TODO: compute dates
            # time-series modalities require *_dates (B, T)
            if x.dim() == 5:
                T = x.shape[1]
                dkey = f"{m}_dates"
                if dkey not in inputs[0]:
                    raise ValueError(f"Missing '{dkey}' for time-series modality '{m}'")
                dates = [s[dkey] for s in inputs]
                date_keys[dkey] = self._stack_dates(dates, B=x.shape[0], T=T)

        if not present:
            raise RuntimeError("No supported modalities found in inputs")

        out_modality = present[0]
        if batch[out_modality].dim() == 5:
            _, T, C, H, W = batch[out_modality].shape
        else:
            _, C, H, W = batch[out_modality].shape
        patch_size = self._calc_patch_size(H)

        xdict: dict[str, torch.Tensor] = {}
        xdict.update(batch)
        xdict.update(date_keys)

        # TODO: do we want tile too?
        out = self.model(
            x=xdict,
            patch_size=patch_size,
            output="dense",
            output_modality=out_modality,
        )
        if out.dim() != 4:
            raise RuntimeError(f"Unexpected AnySat output shape {tuple(out.shape)}")
        self._last_depth = int(out.shape[1])
        return [out]

    def get_backbone_channels(self) -> list[tuple[int, int]]:
        """Returns the output channels of this model when used as a backbone."""
        depth = self._last_depth
        return [(PATCH_SIZE, depth)]  # type: ignore


class AnySatNormalize(Transform):
    """Normalize inputs using AnySat normalization."""

    def __init__(
        self, stats: dict[str, tuple[list[float], list[float]]] | None = None
    ) -> None:
        """Initialize a new AnySatNormalize."""
        super().__init__()
        self.stats = stats or {}

    @staticmethod
    def _norm(x: torch.Tensor, means: list[float], stds: list[float]) -> torch.Tensor:
        """Normalize a tensor with given means and stds."""
        if x.dim() == 3:
            C, H, W = x.shape
            m = torch.tensor(means, dtype=x.dtype, device=x.device).view(C, 1, 1)
            s = (
                torch.tensor(stds, dtype=x.dtype, device=x.device)
                .clamp_min(1e-12)
                .view(C, 1, 1)
            )
            return (x - m) / s
        elif x.dim() == 4:
            T, C, H, W = x.shape
            m = torch.tensor(means, dtype=x.dtype, device=x.device).view(1, C, 1, 1)
            s = (
                torch.tensor(stds, dtype=x.dtype, device=x.device)
                .clamp_min(1e-12)
                .view(1, C, 1, 1)
            )
            return (x - m) / s
        else:
            raise ValueError("Expected (C, H, W) or (T, C, H, W)")

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply normalization to the input dict."""
        for m, tensor in list(input_dict.items()):
            if not isinstance(tensor, torch.Tensor):
                continue
            if m not in self.stats:
                # allow per-sample override: input_dict['s2_stats']={'means':..., 'stds':...}
                sstats = input_dict.get(f"{m}_stats")
                if sstats is None:
                    continue
                means, stds = sstats["means"], sstats["stds"]
            else:
                means, stds = self.stats[m]
            input_dict[m] = self._norm(tensor, means, stds)
        return input_dict, target_dict
