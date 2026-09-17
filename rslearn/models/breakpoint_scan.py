"""Learned changepoint scan over the token (time) dimension of a TokenFeatureMaps."""

from enum import StrEnum
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from rslearn.train.model_context import ModelContext

from .component import FeatureMaps, IntermediateComponent, TokenFeatureMaps


class BreakpointOutput(StrEnum):
    """Which feature of the changepoint scan a BreakpointScan returns."""

    # Per-split scorer features max-pooled over the splits (B x hidden x H x W).
    EVIDENCE = "evidence"
    # Split-attention-weighted mean of the tokens before the breakpoint (B x C x H x W).
    BEFORE = "before"
    # Split-attention-weighted mean of the tokens after the breakpoint (B x C x H x W).
    AFTER = "after"
    # Channel concatenation of BEFORE and AFTER (B x 2C x H x W).
    BEFORE_AFTER = "before_after"


class BreakpointScan(IntermediateComponent):
    """Changepoint scan over the T chronological tokens at each spatial location.

    The input is a BCHWT token feature map (one token per timestep). For every
    candidate split t in [0, T-2], the mean of the tokens up to and including t (the
    "before" aggregate A_t) and the mean of the tokens after t (the "after" aggregate
    B_t) are compared via |B_t - A_t| by a shared 1x1 conv scorer. Depending on the
    configured output, the component returns one of:

    - EVIDENCE: the per-split hidden scorer features max-pooled over the splits
      (B x hidden x H x W). A decoder on this feature can only express "change" as a
      before-vs-after dissimilarity at some breakpoint.
    - BEFORE / AFTER: the before and after aggregates weighted by a softmax over a
      scalar per-split score (B x C x H x W). These represent the state before and
      after the most likely breakpoint and are suitable for decoding the source and
      destination classes of a transition.
    - BEFORE_AFTER: the channel concatenation of before and after (B x 2C x H x W).
    """

    def __init__(
        self, in_dim: int, output: BreakpointOutput, hidden: int = 256
    ) -> None:
        """Create a new BreakpointScan.

        Args:
            in_dim: the token embedding dimension C.
            output: which feature of the scan to return.
            hidden: the hidden width of the split scorer, which is also the channel
                count of the evidence feature.
        """
        super().__init__()
        self.in_dim = in_dim
        # Accept the string value too (e.g. from tests or programmatic use).
        self.output = BreakpointOutput(output)
        self.hidden = hidden
        # Scorer applied to |after - before| at every split.
        self.split_proj = nn.Sequential(
            nn.Conv2d(in_dim, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        # Scalar per-split score for the split-attention over breakpoints.
        self.split_score = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, intermediates: Any, context: ModelContext) -> FeatureMaps:
        """Apply the changepoint scan.

        Args:
            intermediates: the output from the previous component, which must be a
                TokenFeatureMaps with a single BCHWT feature map.
            context: the model context.

        Returns:
            a FeatureMaps with one BCHW map, the configured output of the scan.
        """
        if not isinstance(intermediates, TokenFeatureMaps):
            raise ValueError("input to BreakpointScan must be a TokenFeatureMaps")
        if len(intermediates.feature_maps) != 1:
            raise ValueError(
                "input to BreakpointScan must have one feature map, but got "
                f"{len(intermediates.feature_maps)}"
            )
        feature = intermediates.feature_maps[0]
        B, C, H, W, T = feature.shape
        if C != self.in_dim:
            raise ValueError(f"BreakpointScan expected {self.in_dim} channels, got {C}")
        if T < 2:
            raise ValueError(f"BreakpointScan needs at least 2 tokens, got {T}")

        S = T - 1
        cums = feature.cumsum(dim=-1)  # (B, C, H, W, T)
        total = cums[..., -1:]
        counts = torch.arange(1, T, device=feature.device, dtype=feature.dtype)
        before = cums[..., :-1] / counts  # (B, C, H, W, S) mean of tokens [0, t]
        after = (total - cums[..., :-1]) / (T - counts)  # mean of tokens (t, T)
        con = (after - before).abs()

        # Scorer over all splits: fold S into the batch dimension.
        con = con.permute(0, 4, 1, 2, 3).reshape(B * S, C, H, W)
        hidden = self.split_proj(con)  # (B*S, hidden, H, W)
        scores = self.split_score(hidden)  # (B*S, 1, H, W)
        hidden = hidden.reshape(B, S, self.hidden, H, W)
        scores = scores.reshape(B, S, H, W)

        if self.output == BreakpointOutput.EVIDENCE:
            return FeatureMaps([hidden.max(dim=1).values])  # (B, hidden, H, W)

        w = F.softmax(scores, dim=1)  # (B, S, H, W)
        w = w.permute(0, 2, 3, 1).unsqueeze(1)  # (B, 1, H, W, S)
        if self.output == BreakpointOutput.BEFORE:
            return FeatureMaps([(before * w).sum(dim=-1)])  # (B, C, H, W)
        if self.output == BreakpointOutput.AFTER:
            return FeatureMaps([(after * w).sum(dim=-1)])
        before_agg = (before * w).sum(dim=-1)
        after_agg = (after * w).sum(dim=-1)
        return FeatureMaps([torch.cat([before_agg, after_agg], dim=1)])  # (B, 2C, H, W)
