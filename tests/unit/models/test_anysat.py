"""Test the AnySat model."""

import pathlib
from typing import Any

import huggingface_hub.constants
import pytest
import torch

from rslearn.models.anysat import AnySat


@pytest.mark.slow
def test_anysat_various_modalities(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    # Use monkeypatch to set HF_HUB_CACHE so we can store the weights in a temp dir.
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))

    scenarios: list[dict[str, Any]] = [
        # 1. Single s2 (dense)
        {
            "modalities": ["s2"],
            "dates": {"s2": list(range(3))},
            "inputs": [{"s2": torch.zeros((3 * 10, 64, 64))}],
            "patch_size": 20,
            "expected_shape": (1, 1536, 64, 64),
            "mode": "dense",
            "output_modality": "s2",
        },
        # 2. Multimodal: s1-asc + s2 (patch)
        {
            "modalities": ["s1-asc", "s2"],
            "dates": {"s1-asc": list(range(4)), "s2": list(range(3))},
            "inputs": [
                {
                    "s1-asc": torch.zeros((4 * 2, 64, 64)),
                    "s2": torch.zeros((3 * 10, 64, 64)),
                }
            ],
            "patch_size": 20,
            "expected_shape": (1, 768, 32, 32),
            "mode": "patch",
            "output_modality": None,
        },
        # 3. Landsat 8 (patch)
        {
            "modalities": ["l8"],
            "dates": {"l8": list(range(3))},
            "inputs": [{"l8": torch.zeros((3 * 11, 64, 64))}],
            "patch_size": 20,
            "expected_shape": (1, 768, 32, 32),
            "mode": "patch",
            "output_modality": None,
        },
        # 4. Single-date naip (patch)
        {
            "modalities": ["naip"],
            "dates": {},
            "inputs": [{"naip": torch.zeros((4, 512, 512))}],
            "patch_size": 20,
            "expected_shape": (1, 768, 32, 32),
            "mode": "patch",
            "output_modality": None,
        },
        # 5. naip + s2 with equal extent, dense output
        {
            "modalities": ["naip", "s2"],
            "dates": {"s2": list(range(3))},
            "inputs": [
                {
                    "naip": torch.zeros((4, 512, 512)),
                    "s2": torch.zeros((3 * 10, 64, 64)),
                }
            ],
            "patch_size": 10,
            "expected_shape": (1, 1536, 64, 64),
            "mode": "dense",
            "output_modality": "s2",
        },
    ]

    for scenario in scenarios:
        model = AnySat(
            modalities=scenario["modalities"],
            patch_size_meters=scenario["patch_size"],
            dates=scenario["dates"],
            output=scenario["mode"],
            output_modality=scenario["output_modality"],
        )
        # Only one feature map returned
        features = model.forward(scenario["inputs"])[0]

        assert features.shape == scenario["expected_shape"]  # type: ignore

        if scenario["mode"] == "patch":
            assert model.get_backbone_channels() == [
                (model.patch_size_meters // 10, 768)
            ]
        else:
            assert model.get_backbone_channels() == [(1, 1536)]
