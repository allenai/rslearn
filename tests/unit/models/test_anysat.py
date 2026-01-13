"""Test the AnySat model."""

import pathlib
from datetime import datetime
from typing import Any

import huggingface_hub.constants
import pytest
import torch

from rslearn.models.anysat import AnySat
from rslearn.train.model_context import ModelContext, RasterImage


@pytest.mark.slow
def test_anysat_various_modalities(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    # Use monkeypatch to set HF_HUB_CACHE so we can store the weights in a temp dir.
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))

    scenarios: list[dict[str, Any]] = [
        # 1. Single s2 (dense)
        {
            "modalities": ["s2"],
            "inputs": [
                {
                    "s2": RasterImage(
                        torch.zeros((10, 3, 16, 16)),
                        timestamps=[
                            (datetime(2025, x, 1), datetime(2025, x, 1))
                            for x in range(1, 3 + 1)
                        ],
                    )
                }
            ],
            "patch_size": 20,
            "expected_shape": (1, 1536, 16, 16),
            "mode": "dense",
            "output_modality": "s2",
        },
        # 2. Multimodal: s1-asc + s2 (patch)
        {
            "modalities": ["s1-asc", "s2"],
            "inputs": [
                {
                    "s1-asc": RasterImage(
                        torch.zeros((2, 4, 16, 16)),
                        timestamps=[
                            (datetime(2025, x, 1), datetime(2025, x, 1))
                            for x in range(1, 4 + 1)
                        ],
                    ),
                    "s2": RasterImage(
                        torch.zeros((10, 3, 16, 16)),
                        timestamps=[
                            (datetime(2025, x, 1), datetime(2025, x, 1))
                            for x in range(1, 3 + 1)
                        ],
                    ),
                }
            ],
            "patch_size": 20,
            "expected_shape": (1, 768, 8, 8),
            "mode": "patch",
            "output_modality": None,
        },
        # 3. Landsat 8 (patch)
        {
            "modalities": ["l8"],
            "inputs": [
                {
                    "l8": RasterImage(
                        torch.zeros((11, 3, 16, 16)),
                        timestamps=[
                            (datetime(2025, x, 1), datetime(2025, x, 1))
                            for x in range(1, 3 + 1)
                        ],
                    )
                }
            ],
            "patch_size": 20,
            "expected_shape": (1, 768, 8, 8),
            "mode": "patch",
            "output_modality": None,
        },
    ]

    for scenario in scenarios:
        model = AnySat(
            modalities=scenario["modalities"],
            patch_size_meters=scenario["patch_size"],
            output=scenario["mode"],
            output_modality=scenario["output_modality"],
        )
        # Only one feature map returned
        context = ModelContext(
            inputs=scenario["inputs"],
            metadatas=[],
        )
        features = model.forward(context).feature_maps[0]

        assert features.shape == scenario["expected_shape"]  # type: ignore

        if scenario["mode"] == "patch":
            assert model.get_backbone_channels() == [
                (model.patch_size_meters // 10, 768)
            ]
        else:
            assert model.get_backbone_channels() == [(1, 1536)]
