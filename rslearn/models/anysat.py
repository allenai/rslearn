"""AnySat model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

ANYSAT_MODALITIES = ["s2", "s1", "l8"]
# We return per-pixel (dense) features, so the effective patch size is 1.
PATCH_SIZE = 1


class AnySat(nn.Module):
    """AnySat backbone."""

    def __init__(self, modalities: list[str] = ["s2"]) -> None:
        """Initialize the AnySat backbone.

        Args:
            modalities: Modalities to use. Must be a subset of {'s2','s1','l8'}.
                        Order matters for output_modality preference (first present wins).
        """
        super().__init__()
        for m in modalities:
            if m not in ANYSAT_MODALITIES:
                raise ValueError(
                    f"Invalid modality: {m}. Supported: {ANYSAT_MODALITIES}"
                )
        self.modalities = list(modalities)

        # Load AnySat from torch.hub.
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
        """Heuristic from paper/README: cap subpatch size at 40 m."""
        # If your inputs are 10 m/pixel (e.g., S2 tiles), this matches the common setting.
        h_in_m = h_pixels * 10
        return min(40, h_in_m)

    def forward(self, inputs: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            inputs: list of per-sample dicts. Each dict should already contain:
                - For any chosen modality m ∈ {'s2','s1','l8'}:
                    m:            (T, C, H, W)   # time-first
                    f"{m}_dates": (T,) or (1, T) or (B, T)  # zero-based day-of-year
              No band reordering/normalization is done here; provide tensors as AnySat expects.

        Returns:
            A single-element list with the spatial feature map: (B, D, H, W).
        """
        if len(inputs) == 0:
            raise ValueError("inputs must be a non-empty list of sample dicts.")

        # Collate batch for present modalities.
        batch_inputs: dict[str, torch.Tensor] = {}
        batch_dates: dict[str, torch.Tensor] = {}

        # Determine device from the first found tensor
        device = None
        for s in inputs:
            for k, v in s.items():
                if isinstance(v, torch.Tensor):
                    device = v.device
                    break
            if device is not None:
                break
        if device is None:
            device = torch.device("cpu")

        # Stack (T,C,H,W) -> (B,T,C,H,W) and dates -> (B,T)
        present_modalities: list[str] = []
        for m in self.modalities:
            if (m in inputs[0]) and (f"{m}_dates" in inputs[0]):
                present_modalities.append(m)
                imgs = [s[m] for s in inputs]  # each (T,C,H,W)
                dates = [s[f"{m}_dates"] for s in inputs]  # each (T,) or (B?,T)

                # Images
                if not all(isinstance(t, torch.Tensor) and t.dim() == 4 for t in imgs):
                    raise ValueError(f"All '{m}' tensors must be (T,C,H,W).")
                B = len(imgs)
                T, C, H, W = imgs[0].shape
                if not all(t.shape == (T, C, H, W) for t in imgs):
                    raise ValueError(
                        f"All '{m}' tensors must share the same (T,C,H,W)."
                    )
                batch_inputs[m] = torch.stack(imgs, dim=0)  # (B,T,C,H,W)

                # Dates
                fixed_dates = []
                for d in dates:
                    if d.dim() == 1:  # (T,) -> (1,T)
                        d = d.unsqueeze(0)
                    if d.dim() == 2 and d.size(0) == 1:
                        # (1,T) -> (1,T)
                        pass
                    elif d.dim() == 2 and d.size(0) == B:
                        # Already (B,T) for some pipelines
                        pass
                    else:
                        raise ValueError(
                            f"'{m}_dates' must be (T,), (1,T) or (B,T); got {tuple(d.shape)}"
                        )
                    fixed_dates.append(d)
                # Now stack along batch, resulting (B,T)
                d0 = fixed_dates[0]
                if d0.size(0) == 1:
                    dstack = torch.cat(fixed_dates, dim=0)
                else:
                    dstack = torch.stack(fixed_dates, dim=0).squeeze(1)
                if dstack.dim() != 2 or dstack.size(0) != B or dstack.size(1) != T:
                    raise ValueError(
                        f"Stacked '{m}_dates' must be (B,T); got {tuple(dstack.shape)}"
                    )
                batch_dates[f"{m}_dates"] = dstack.to(device=device, dtype=torch.long)

        if not present_modalities:
            raise RuntimeError(
                f"No supported modalities found in inputs for {self.modalities}. "
                f"Make sure each sample dict has modality and '{m}_dates' tensors."
            )

        # Choose output modality by the order the user configured (first present wins).
        output_modality = next(m for m in self.modalities if m in present_modalities)

        # Merge image & date dicts for AnySat call.
        anysat_inputs: dict[str, torch.Tensor] = {}
        anysat_inputs.update(batch_inputs)
        anysat_inputs.update(batch_dates)

        # Patch size heuristic from H; assumes consistent H across batch.
        _, T, C, H, W = batch_inputs[output_modality].shape
        patch_size = self._calc_patch_size(H)

        # AnySat forward with dense (per-pixel) output.
        out = self.model(
            x=anysat_inputs,  # {'s2','s1','l8', 's2_dates','s1_dates','l8_dates'}
            patch_size=patch_size,
            output="dense",  # returns (B, D, H, W)
            output_modality=output_modality,
        )
        if out.dim() != 4:
            raise RuntimeError(
                f"Unexpected AnySat output shape {tuple(out.shape)}; expected (B,D,H,W)."
            )

        self._last_depth = int(out.shape[1])
        return [out]

    def get_backbone_channels(self) -> list[tuple[int, int]]:
        """Return [(patch_size, depth)] like TerraMind."""
        depth = self._last_depth if self._last_depth is not None else 1024
        return [(PATCH_SIZE, depth)]
