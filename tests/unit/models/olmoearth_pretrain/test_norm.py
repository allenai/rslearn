"""Test rslearn.models.olmoearth_pretrain.norm"""

from datetime import datetime

import torch

from rslearn.models.olmoearth_pretrain.norm import OlmoEarthNormalize
from rslearn.utils.raster_format import RasterImage


def test_forward() -> None:
    """Test forward pass with randomly initialized model."""
    norm_layer = OlmoEarthNormalize(
        band_names={
            "sentinel2_l2a": [
                "B02",
                "B03",
                "B04",
                "B08",
                "B05",
                "B06",
                "B07",
                "B8A",
                "B11",
                "B12",
                "B01",
                "B09",
            ]
        }
    )

    T = 2
    H = 4
    W = 4
    inputs = {
        # 12 channels per timestep.
        "sentinel2_l2a": RasterImage(
            image=torch.zeros(
                (12, T, H, W), dtype=torch.float32, device=torch.device("cpu")
            ),
            timestamps=[
                (datetime(2025, x, 1), datetime(2025, x, 1)) for x in range(1, T + 1)
            ],
        )
    }
    normed_inputs, _ = norm_layer(input_dict=inputs, target_dict={})

    assert (
        normed_inputs["sentinel2_l2a"].image.shape
        == inputs["sentinel2_l2a"].image.shape
    )
