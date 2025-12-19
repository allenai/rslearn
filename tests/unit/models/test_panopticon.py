"""Tests for the Panopticon model."""

import logging

import pytest
import torch

from rslearn.models.panopticon import Panopticon, PanopticonModalities
from rslearn.utils.raster_format import RasterImage

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_panopticon() -> None:
    """Test the Panopticon model."""
    band_order = {
        PanopticonModalities.SENTINEL2.value: [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
        ]
    }
    with torch.no_grad():
        panopticon = Panopticon(band_order=band_order).to(DEVICE)

    input_hw = 32
    inputs = [
        {
            PanopticonModalities.SENTINEL2.value: RasterImage(
                torch.randn(
                    (
                        len(band_order[PanopticonModalities.SENTINEL2.value]),
                        1,
                        input_hw,
                        input_hw,
                    ),
                    dtype=torch.float32,
                    device=DEVICE,
                )
            ),
        },
        {
            PanopticonModalities.SENTINEL2.value: RasterImage(
                torch.randn(
                    (
                        len(band_order[PanopticonModalities.SENTINEL2.value]),
                        1,
                        input_hw,
                        input_hw,
                    ),
                    dtype=torch.float32,
                    device=DEVICE,
                )
            ),
        },
    ]
    feature_list = panopticon(inputs)
    features = feature_list[0]
    logger.info(f"Features shape: {features.shape}")
    # always resizes to 224 with patch size 14
    assert features.shape == (2, 16, 16, panopticon.output_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_panopticon_multiple_modalities() -> None:
    """Test the Panopticon model with multiple modalities."""
    band_order = {
        PanopticonModalities.SENTINEL2.value: [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
        ],
        PanopticonModalities.SENTINEL1.value: ["vv", "vh"],
        PanopticonModalities.LANDSAT8.value: [
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B9",
            "B10",
            "B11",
        ],
    }
    input_hw = 32
    inputs = [
        {
            PanopticonModalities.SENTINEL2.value: RasterImage(
                torch.randn(
                    (
                        len(band_order[PanopticonModalities.SENTINEL2.value]),
                        1,
                        input_hw,
                        input_hw,
                    ),
                    dtype=torch.float32,
                    device=DEVICE,
                )
            ),
            PanopticonModalities.SENTINEL1.value: RasterImage(
                torch.randn(
                    (
                        len(band_order[PanopticonModalities.SENTINEL1.value]),
                        1,
                        input_hw,
                        input_hw,
                    ),
                    dtype=torch.float32,
                    device=DEVICE,
                )
            ),
            PanopticonModalities.LANDSAT8.value: RasterImage(
                torch.randn(
                    (
                        len(band_order[PanopticonModalities.LANDSAT8.value]),
                        1,
                        input_hw,
                        input_hw,
                    ),
                    dtype=torch.float32,
                    device=DEVICE,
                )
            ),
        },
    ]
    # move to device if possible
    panopticon = Panopticon(band_order=band_order).to(DEVICE)
    feature_list = panopticon(inputs)
    features = feature_list[0]
    logger.info(f"Features shape: {features.shape}")
    assert features.shape == (1, 16, 16, panopticon.output_dim)
