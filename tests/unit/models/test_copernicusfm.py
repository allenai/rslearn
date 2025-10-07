"""Test Copernicus FM model."""

import pytest
import torch

from rslearn.models.copernicusfm import CopernicusFM, CopernicusFMModality

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
def test_copernicusfm() -> None:
    """Verify that the forward pass for CROMA works."""
    input_hw = 32
    # We override the temporary directory so we don't retain the model weights outside
    # of this test.

    band_order = {
        CopernicusFMModality.SENTINEL2_L2A.value: [
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B11",
            "B12",
        ],
        CopernicusFMModality.SENTINEL1.value: ["vv", "vh"],
    }
    inputs = [
        {
            "sentinel2": torch.zeros(
                (
                    len(band_order[CopernicusFMModality.SENTINEL2_L2A.value]),
                    input_hw,
                    input_hw,
                ),
                dtype=torch.float32,
                device=DEVICE,
            ),
            "sentinel1": torch.zeros(
                (
                    len(band_order[CopernicusFMModality.SENTINEL1.value]),
                    input_hw,
                    input_hw,
                ),
                dtype=torch.float32,
                device=DEVICE,
            ),
        }
    ]
    copernicusfm = CopernicusFM(band_order=band_order, cache_dir=None).to(DEVICE)
    with torch.no_grad():
        feature_list = copernicusfm(inputs)
    assert (
        feature_list[0].shape == torch.Size([1, 768, 14, 14]) and len(feature_list) == 1
    )
