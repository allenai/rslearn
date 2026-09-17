"""Tests for rslearn.models.breakpoint_scan."""

import pytest
import torch

from rslearn.models.breakpoint_scan import BreakpointOutput, BreakpointScan
from rslearn.models.component import FeatureMaps, TokenFeatureMaps
from rslearn.train.model_context import ModelContext

DIM = 8
HIDDEN = 6
CONTEXT = ModelContext(inputs=[], metadatas=[])
TOL = dict(atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "output,channels",
    [
        (BreakpointOutput.EVIDENCE, HIDDEN),
        (BreakpointOutput.BEFORE, DIM),
        (BreakpointOutput.AFTER, DIM),
        (BreakpointOutput.BEFORE_AFTER, 2 * DIM),
    ],
)
def test_output_shapes(output: BreakpointOutput, channels: int) -> None:
    """Each output option yields a single BCHW map with the expected channels."""
    model = BreakpointScan(in_dim=DIM, output=output, hidden=HIDDEN)
    out = model(TokenFeatureMaps([torch.randn(2, DIM, 3, 4, 5)]), CONTEXT)
    assert isinstance(out, FeatureMaps)
    assert len(out.feature_maps) == 1
    assert out.feature_maps[0].shape == (2, channels, 3, 4)


def test_single_split() -> None:
    """With T=2 there is one split so before/after are the two tokens themselves."""
    x = torch.randn(1, DIM, 2, 2, 2)
    tokens = TokenFeatureMaps([x])
    evidence = BreakpointScan(
        in_dim=DIM, output=BreakpointOutput.EVIDENCE, hidden=HIDDEN
    )
    before = BreakpointScan(in_dim=DIM, output=BreakpointOutput.BEFORE, hidden=HIDDEN)
    after = BreakpointScan(in_dim=DIM, output=BreakpointOutput.AFTER, hidden=HIDDEN)
    both = BreakpointScan(
        in_dim=DIM, output=BreakpointOutput.BEFORE_AFTER, hidden=HIDDEN
    )
    # Share one set of scorer weights across the four instances.
    for model in (before, after, both):
        model.load_state_dict(evidence.state_dict())

    # The after mean is computed via cumsum (total - before), so allow float error.
    assert torch.allclose(before(tokens, CONTEXT).feature_maps[0], x[..., 0], **TOL)
    assert torch.allclose(after(tokens, CONTEXT).feature_maps[0], x[..., 1], **TOL)
    # The evidence is the scorer applied to |t1 - t0|.
    expected = evidence.split_proj((x[..., 1] - x[..., 0]).abs())
    assert torch.allclose(evidence(tokens, CONTEXT).feature_maps[0], expected, **TOL)
    # before_after is the channel concatenation.
    assert torch.allclose(
        both(tokens, CONTEXT).feature_maps[0],
        torch.cat([x[..., 0], x[..., 1]], dim=1),
        **TOL,
    )


def test_constant_series_has_no_evidence() -> None:
    """A constant time series has zero |after - before| at every split."""
    x = torch.ones(1, DIM, 2, 2, 6) * 3.0
    tokens = TokenFeatureMaps([x])
    evidence = BreakpointScan(
        in_dim=DIM, output=BreakpointOutput.EVIDENCE, hidden=HIDDEN
    )
    before = BreakpointScan(in_dim=DIM, output=BreakpointOutput.BEFORE, hidden=HIDDEN)
    before.load_state_dict(evidence.state_dict())
    # All splits see the same zero contrast, so evidence equals the scorer at zero.
    expected = evidence.split_proj(torch.zeros(1, DIM, 2, 2))
    assert torch.allclose(evidence(tokens, CONTEXT).feature_maps[0], expected, **TOL)
    # The before aggregate of a constant series is the constant.
    assert torch.allclose(before(tokens, CONTEXT).feature_maps[0], x[..., 0], **TOL)
