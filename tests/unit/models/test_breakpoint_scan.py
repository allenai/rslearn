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


@pytest.mark.parametrize(
    "output",
    [
        BreakpointOutput.EVIDENCE,
        BreakpointOutput.BEFORE,
        BreakpointOutput.AFTER,
        BreakpointOutput.BEFORE_AFTER,
    ],
)
def test_mask_matches_truncated_series(output: BreakpointOutput) -> None:
    """Masking trailing tokens gives the same result as truncating the series."""
    model = BreakpointScan(in_dim=DIM, output=output, hidden=HIDDEN)
    T = 6
    num_valid = 4
    x = torch.randn(1, DIM, 2, 2, T)
    mask = torch.ones(1, 2, 2, T, dtype=torch.bool)
    mask[..., num_valid:] = False

    masked = model(TokenFeatureMaps([x], masks=[mask]), CONTEXT).feature_maps[0]
    truncated = model(TokenFeatureMaps([x[..., :num_valid]]), CONTEXT).feature_maps[0]
    assert torch.allclose(masked, truncated, **TOL)

    # Perturbing the masked tokens does not change the output.
    x2 = x.clone()
    x2[..., num_valid:] += 100.0
    masked2 = model(TokenFeatureMaps([x2], masks=[mask]), CONTEXT).feature_maps[0]
    assert torch.allclose(masked, masked2, **TOL)


def test_mask_per_sample_lengths() -> None:
    """Samples in a batch may have different numbers of valid tokens."""
    model = BreakpointScan(in_dim=DIM, output=BreakpointOutput.BEFORE, hidden=HIDDEN)
    T = 5
    x = torch.randn(2, DIM, 1, 1, T)
    mask = torch.ones(2, 1, 1, T, dtype=torch.bool)
    mask[1, ..., 2:] = False

    out = model(TokenFeatureMaps([x], masks=[mask]), CONTEXT).feature_maps[0]
    # Sample 0 is fully valid and equals the unmasked result.
    full = model(TokenFeatureMaps([x[:1]]), CONTEXT).feature_maps[0]
    assert torch.allclose(out[:1], full, **TOL)
    # Sample 1 has T=2 valid tokens so there is a single valid split, and the
    # before aggregate is exactly the first token.
    assert torch.allclose(out[1], x[1, ..., 0], **TOL)


@pytest.mark.parametrize("output", list(BreakpointOutput))
def test_all_masked_location_no_nan(output: BreakpointOutput) -> None:
    """Locations with fewer than two valid tokens still give finite outputs."""
    model = BreakpointScan(in_dim=DIM, output=output, hidden=HIDDEN)
    x = torch.randn(1, DIM, 1, 2, 4)
    mask = torch.ones(1, 1, 2, 4, dtype=torch.bool)
    # First location: no valid tokens. Second location: one valid token.
    mask[:, 0, 0, :] = False
    mask[:, 0, 1, 1:] = False
    out = model(TokenFeatureMaps([x], masks=[mask]), CONTEXT).feature_maps[0]
    assert torch.isfinite(out).all()
