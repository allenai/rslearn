import argparse
import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rslearn.dataset import WindowLayerData

MODULE_PATH = Path(__file__).parents[3] / "docs/examples/align_item_groups_by_name.py"
SPEC = importlib.util.spec_from_file_location("align_item_groups_by_name", MODULE_PATH)
assert SPEC is not None
align_items = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = align_items
SPEC.loader.exec_module(align_items)


def make_item(name: str) -> dict:
    return {
        "name": name,
        "geometry": {
            "time_range": [
                datetime(2024, 1, 1, tzinfo=UTC).isoformat(),
                datetime(2024, 1, 2, tzinfo=UTC).isoformat(),
            ]
        },
    }


def test_align_target_layer_matches_default_name_template() -> None:
    reference_data = WindowLayerData(
        "sentinel2",
        [
            [make_item("S2C_31TEJ_20250424_0_L2A")],
            [make_item("S2C_31TEJ_20250425_0_L2A")],
        ],
    )
    first_target_range = (
        datetime(2025, 4, 24, tzinfo=UTC),
        datetime(2025, 4, 25, tzinfo=UTC),
    )
    second_target_range = (
        datetime(2025, 4, 25, tzinfo=UTC),
        datetime(2025, 4, 26, tzinfo=UTC),
    )
    target_data = WindowLayerData(
        "lai",
        [
            [make_item("S2C_31TEJ_20250425_0_L2A_LAI")],
            [make_item("S2C_31TEJ_20250424_0_L2A_LAI")],
        ],
        group_time_ranges=[
            second_target_range,
            first_target_range,
        ],
        materialized=True,
    )
    args = argparse.Namespace(
        target_name_template="{reference_item_name}_{target_layer_upper}",
        group_time_range_source="target",
    )

    aligned_data = align_items.align_target_layer(reference_data, target_data, args)

    assert aligned_data.layer_name == "lai"
    assert aligned_data.materialized is False
    assert [group[0]["name"] for group in aligned_data.serialized_item_groups] == [
        "S2C_31TEJ_20250424_0_L2A_LAI",
        "S2C_31TEJ_20250425_0_L2A_LAI",
    ]
    assert aligned_data.group_time_ranges == [
        first_target_range,
        second_target_range,
    ]


def test_align_target_layer_matches_custom_name_template() -> None:
    reference_data = WindowLayerData(
        "sentinel2",
        [[make_item("S2A_scene_20240101")]],
    )
    target_data = WindowLayerData(
        "lai",
        [
            [make_item("LAI_scene_20240102")],
            [make_item("LAI_scene_20240101")],
        ],
    )
    args = argparse.Namespace(
        target_name_template="{target_layer_upper}_scene_20240101",
        group_time_range_source="none",
    )

    aligned_data = align_items.align_target_layer(reference_data, target_data, args)

    assert aligned_data.serialized_item_groups[0][0]["name"] == "LAI_scene_20240101"
    assert aligned_data.group_time_ranges is None


def test_align_target_layer_raises_when_target_name_is_missing() -> None:
    reference_data = WindowLayerData(
        "sentinel2",
        [[make_item("S2C_31TEJ_20250424_0_L2A")]],
    )
    target_data = WindowLayerData(
        "lai",
        [[make_item("S2C_31TEJ_20250425_0_L2A_LAI")]],
    )
    args = argparse.Namespace(
        target_name_template="{reference_item_name}_{target_layer_upper}",
        group_time_range_source="target",
    )

    with pytest.raises(
        ValueError,
        match="S2C_31TEJ_20250424_0_L2A_LAI",
    ):
        align_items.align_target_layer(reference_data, target_data, args)


def test_format_target_name_rejects_unknown_template_field() -> None:
    with pytest.raises(ValueError, match="unknown target_name_template field"):
        align_items.format_target_name(
            [make_item("S2C_31TEJ_20250424_0_L2A")],
            "lai",
            "{missing}",
        )
