"""Undecimated (stationary) wavelet transform and multiscale loss for ERA5.

Implements a dependency-free differentiable SWT along the time axis using
dilated depthwise ``F.conv1d``.  The transform is non-decimated (à trous):
each level ``j`` dilates the filters by ``2^j`` so every coefficient band
has the same length ``T`` as the input — no downsampling, no alignment
headaches.

Ported from olmoearth_pretrain for supervised ERA5 encoding; no external
wavelet package is required.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wavelet filter banks (hardcoded, no external dependency)
# ---------------------------------------------------------------------------

# Daubechies-2 (db2) decomposition filters
_DB2_LO: list[float] = [
    -0.12940952255092145,
    0.22414386804185735,
    0.836516303737469,
    0.48296291314469025,
]
_DB2_HI: list[float] = [
    -0.48296291314469025,
    0.836516303737469,
    -0.22414386804185735,
    -0.12940952255092145,
]

# Haar (db1)
_HAAR_LO: list[float] = [0.7071067811865476, 0.7071067811865476]
_HAAR_HI: list[float] = [-0.7071067811865476, 0.7071067811865476]

_FILTER_BANKS: dict[str, tuple[list[float], list[float]]] = {
    "db2": (_DB2_LO, _DB2_HI),
    "haar": (_HAAR_LO, _HAAR_HI),
    "db1": (_HAAR_LO, _HAAR_HI),
}


def _get_filters(name: str) -> tuple[list[float], list[float]]:
    key = name.lower()
    if key not in _FILTER_BANKS:
        raise ValueError(
            f"Unknown wavelet {name!r}; available: {sorted(_FILTER_BANKS)}"
        )
    return _FILTER_BANKS[key]


# ---------------------------------------------------------------------------
# SWT module
# ---------------------------------------------------------------------------


class StationaryWaveletTransform1d(nn.Module):
    """Causal undecimated (stationary) wavelet transform along the time axis.

    Input shape ``[B, V, T]`` (channels-first, matching conv1d).
    Output: list of ``(approx, detail)`` tuples per level, each ``[B, V, T']``
    where ``T' = T - target_start`` when cropping is active.

    Uses causal (left-only) zero-padding. When a ``target_start`` buffer is provided
    to :meth:`forward`, the first ``target_start`` coefficients are discarded
    so that every returned coefficient is free of boundary artifacts.
    """

    def __init__(
        self,
        num_channels: int,
        max_levels: int = 6,
        wavelet: str = "haar",
    ) -> None:
        """Initialize the stationary wavelet transform filters."""
        super().__init__()
        lo, hi = _get_filters(wavelet)
        self.filter_len = len(lo)
        self.max_levels = max_levels

        # Depthwise filters: [V, 1, K] repeated for groups=V conv
        lo_t = torch.tensor(lo, dtype=torch.float32).flip(0)
        hi_t = torch.tensor(hi, dtype=torch.float32).flip(0)
        # Shape [num_channels, 1, K]
        lo_w = lo_t.unsqueeze(0).unsqueeze(0).expand(num_channels, -1, -1).clone()
        hi_w = hi_t.unsqueeze(0).unsqueeze(0).expand(num_channels, -1, -1).clone()
        self.register_buffer("lo_filter", lo_w)
        self.register_buffer("hi_filter", hi_w)
        self.num_channels = num_channels

    def forward(
        self,
        x: Tensor,
        levels: list[int] | None = None,
        target_start: int = 83,
    ) -> list[tuple[Tensor, Tensor]]:
        """Compute the causal undecimated SWT.

        Args:
            x: ``[B, V, T]`` input signal.
            levels: Which decomposition levels to return (0-indexed).
                ``None`` returns all ``max_levels`` levels.
            target_start: If > 0, crop the first ``target_start`` timesteps
                from every returned band so that only the target window
                (free of boundary effects) is returned.

        Returns:
            List of ``(approx, detail)`` pairs, one per requested level.
            Each tensor has shape ``[B, V, T']`` where
            ``T' = T - target_start``.
        """
        if levels is None:
            levels = list(range(self.max_levels))
        results: list[tuple[Tensor, Tensor]] = []
        current = x
        for j in range(max(levels) + 1):
            dilation = 2**j
            pad = dilation * (self.filter_len - 1)
            padded = F.pad(current, (pad, 0), mode="constant", value=0.0)
            approx = F.conv1d(
                padded, self.lo_filter, groups=self.num_channels, dilation=dilation
            )
            detail = F.conv1d(
                padded, self.hi_filter, groups=self.num_channels, dilation=dilation
            )
            if j in levels:
                if target_start > 0:
                    results.append(
                        (approx[:, :, target_start:], detail[:, :, target_start:])
                    )
                else:
                    results.append((approx, detail))
            current = approx
        return results


# ---------------------------------------------------------------------------
# Band stacking helper (shared by the encoder input adapter)
# ---------------------------------------------------------------------------


def swt_bands_to_channels(
    bands: list[tuple[Tensor, Tensor]],
    include_approx: bool = True,
) -> Tensor:
    """Stack SWT bands into a var-major channel tensor.

    Turns the per-level ``(approx, detail)`` list returned by
    :meth:`StationaryWaveletTransform1d.forward` into a single dense tensor
    suitable as encoder input.

    Args:
        bands: List of ``(approx, detail)`` per level, each ``[B, V, T]``.
        include_approx: If True, append the *deepest* level's approximation
            band after the detail bands (making the representation complete).

    Returns:
        ``[B, T, V * n_bands]`` where ``n_bands = len(bands) + include_approx``.
        Channels are **var-major**: ``c = v * n_bands + s`` with scale order
        ``[detail_0, ..., detail_{L-1}, (approx_deepest)]``, so a
        ``view(B, T, V, n_bands)`` recovers per-variable scale groups.
    """
    if not bands:
        raise ValueError("swt_bands_to_channels requires at least one SWT level")
    band_list = [detail for _, detail in bands]  # detail_0 .. detail_{L-1}
    if include_approx:
        band_list.append(bands[-1][0])  # deepest-level approximation
    # Each band is [B, V, T]; stack along a new scale axis -> [B, n_bands, V, T].
    stacked = torch.stack(band_list, dim=1)
    b, n_bands, v, t = stacked.shape
    # var-major flatten: [B, V, n_bands, T] -> [B, V*n_bands, T] -> [B, T, C].
    stacked = stacked.permute(0, 2, 1, 3).reshape(b, v * n_bands, t)
    return stacked.transpose(1, 2).contiguous()
