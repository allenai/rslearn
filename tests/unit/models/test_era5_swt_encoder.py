import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import torch

from rslearn.models.component import FeatureMaps
from rslearn.models.era5_encoder import Era5DailyEncoderConfig
from rslearn.models.era5_swt import StationaryWaveletTransform1d, swt_bands_to_channels
from rslearn.models.era5_swt_encoder import Era5SWTEncoder
from rslearn.train.model_context import ModelContext, RasterImage


@pytest.fixture
def encoder(tmp_path: Path) -> Era5SWTEncoder:
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "raw_mean": [10.0] * 14,
                "raw_std": [2.0] * 14,
                "band_order": [str(i) for i in range(14)],
                # Statistics-estimation metadata must not alter runtime masks.
                "nodata_tol": 0.5,
                "exact_match_vars": ["7"],
                "mean": [0.0] * 98,
                "std": [1.0] * 98,
            }
        )
    )
    return Era5SWTEncoder(
        encoder_config=Era5DailyEncoderConfig(
            embedding_size=12,
            depth=1,
            num_heads=3,
            dropout=0.0,
            is_swt_input=True,
            swt_input_stats_path=str(stats_path),
        ),
        raw_stats_path=str(stats_path),
        output_spatial_size=32,
    ).eval()


@pytest.fixture
def context() -> ModelContext:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    timestamps = [
        (start + timedelta(days=i), start + timedelta(days=i + 1)) for i in range(448)
    ]
    return ModelContext(
        inputs=[
            {"era5_daily": RasterImage(torch.full((14, 448, 1, 1), 12.0), timestamps)}
        ],
        metadatas=[],
    )


def test_haar_is_causal_and_channels_are_variable_major() -> None:
    swt = StationaryWaveletTransform1d(2, max_levels=1)
    x = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]])
    channels = swt_bands_to_channels(swt(x, target_start=0))
    # Each variable contributes detail then approximation. Causal Haar uses
    # previous-minus-current for detail and current-plus-previous for approx.
    expected = torch.tensor([[[-1, 1, -2, 2], [-1, 3, -2, 6], [-1, 5, -2, 10]]])
    torch.testing.assert_close(channels, expected.float() / 2**0.5)
    changed = x.clone()
    changed[:, :, -1] = 1000
    torch.testing.assert_close(
        swt_bands_to_channels(swt(changed, target_start=0))[:, :-1], channels[:, :-1]
    )


def test_missing_variable_does_not_erase_other_variables(
    encoder: Era5SWTEncoder, context: ModelContext
) -> None:
    context.inputs[0]["era5_daily"].image[0, 10, 0, 0] = -9999
    normalized, timestamps, valid = encoder._prepare_inputs(context)
    assert normalized[0, 10, 0] == 0
    assert normalized[0, 10, 1] == 1
    assert not valid[0, 10, 0] and valid[0, 10, 1]
    assert timestamps[0, 59].tolist() == [60, 1, 2020]  # leap day
    bands = encoder.encoder._apply_swt(normalized, valid)
    assert torch.count_nonzero(bands[:, :, :7]) == 0
    assert torch.count_nonzero(bands[:, :, 7:]) > 0


def test_encoder_output_matches_label_grid(
    encoder: Era5SWTEncoder, context: ModelContext
) -> None:
    with torch.inference_mode():
        output = encoder(context)
    assert isinstance(output, FeatureMaps)
    prediction = output.feature_maps[0]
    assert prediction.shape == (1, 1, 32, 32)
    assert torch.isfinite(prediction).all()
    torch.testing.assert_close(
        prediction, prediction[:, :, :1, :1].expand_as(prediction)
    )


def test_missing_timestamps_fail_explicitly(
    encoder: Era5SWTEncoder, context: ModelContext
) -> None:
    context.inputs[0]["era5_daily"].timestamps = None
    with pytest.raises(ValueError, match="timestamp"):
        encoder(context)


def test_missing_values_use_exact_sentinel_in_every_band(
    encoder: Era5SWTEncoder, context: ModelContext
) -> None:
    image = context.inputs[0]["era5_daily"].image
    image[:, 10, 0, 0] = -9998.75
    image[:, 11, 0, 0] = -9999
    normalized, _, valid = encoder._prepare_inputs(context)
    assert valid[0, 10].all()
    assert not valid[0, 11].any()
    assert torch.count_nonzero(normalized[0, 10]) == 14
    assert torch.count_nonzero(normalized[0, 11]) == 0


def test_entirely_missing_sample_matches_pretraining_zero_fill(
    encoder: Era5SWTEncoder, context: ModelContext
) -> None:
    context.inputs[0]["era5_daily"].image.fill_(-9999)
    normalized, _, valid = encoder._prepare_inputs(context)
    assert not valid.any()
    assert torch.count_nonzero(normalized) == 0
    assert torch.count_nonzero(encoder.encoder._apply_swt(normalized, valid)) == 0
