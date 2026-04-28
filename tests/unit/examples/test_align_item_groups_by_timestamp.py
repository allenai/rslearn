import argparse
import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rslearn.dataset import WindowLayerData

MODULE_PATH = (
    Path(__file__).parents[3] / "docs/examples/align_item_groups_by_timestamp.py"
)
SPEC = importlib.util.spec_from_file_location(
    "align_item_groups_by_timestamp", MODULE_PATH
)
assert SPEC is not None
align_items = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = align_items
SPEC.loader.exec_module(align_items)


def make_item(name: str, start: datetime) -> dict:
    return {
        "name": name,
        "geometry": {
            "time_range": [
                start.isoformat(),
                (start + timedelta(minutes=10)).isoformat(),
            ]
        },
    }


def test_align_target_layer_matches_nearest_timestamp() -> None:
    first_reference_time = datetime(2024, 1, 1, 12, tzinfo=UTC)
    second_reference_time = datetime(2024, 1, 2, 12, tzinfo=UTC)
    first_target_time = first_reference_time + timedelta(minutes=5)
    second_target_time = second_reference_time - timedelta(minutes=5)
    unused_target_time = datetime(2024, 1, 10, 12, tzinfo=UTC)

    reference_data = WindowLayerData(
        "sentinel2",
        [
            [make_item("sentinel2_a", first_reference_time)],
            [make_item("sentinel2_b", second_reference_time)],
        ],
    )
    first_target_range = (first_target_time, first_target_time + timedelta(hours=1))
    second_target_range = (second_target_time, second_target_time + timedelta(hours=1))
    target_data = WindowLayerData(
        "lai",
        [
            [make_item("lai_unused", unused_target_time)],
            [make_item("lai_b", second_target_time)],
            [make_item("lai_a", first_target_time)],
        ],
        group_time_ranges=[
            (unused_target_time, unused_target_time + timedelta(hours=1)),
            second_target_range,
            first_target_range,
        ],
        materialized=True,
    )
    args = argparse.Namespace(
        match_mode="nearest_timestamp",
        max_delta=timedelta(hours=1),
        group_time_range_source="target",
    )

    aligned_data = align_items.align_target_layer(reference_data, target_data, args)

    assert aligned_data.layer_name == "lai"
    assert aligned_data.materialized is False
    assert [group[0]["name"] for group in aligned_data.serialized_item_groups] == [
        "lai_a",
        "lai_b",
    ]
    assert aligned_data.group_time_ranges == [
        first_target_range,
        second_target_range,
    ]


def test_align_target_layer_matches_name_regex() -> None:
    reference_data = WindowLayerData(
        "sentinel2",
        [[make_item("S2A_scene_20240101", datetime(2024, 1, 1, tzinfo=UTC))]],
    )
    target_data = WindowLayerData(
        "lai",
        [
            [make_item("LAI_scene_20240102", datetime(2024, 1, 2, tzinfo=UTC))],
            [make_item("LAI_scene_20240101", datetime(2024, 1, 1, tzinfo=UTC))],
        ],
    )
    args = argparse.Namespace(
        match_mode="name_regex",
        reference_name_regex=r"^S2A",
        target_name_replacement="LAI",
        group_time_range_source="none",
    )

    aligned_data = align_items.align_target_layer(reference_data, target_data, args)

    assert aligned_data.serialized_item_groups[0][0]["name"] == "LAI_scene_20240101"
    assert aligned_data.group_time_ranges is None


def test_nearest_timestamp_match_respects_max_delta() -> None:
    reference_group = [[make_item("sentinel2", datetime(2024, 1, 1, tzinfo=UTC))]][0]
    target_data = WindowLayerData(
        "lai",
        [[make_item("lai", datetime(2024, 1, 3, tzinfo=UTC))]],
    )

    assert (
        align_items.nearest_timestamp_match(
            reference_group,
            align_items.make_candidates(target_data),
            timedelta(hours=1),
        )
        is None
    )


def test_parse_timedelta_rejects_invalid_duration() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="invalid duration"):
        align_items.parse_timedelta("not-a-duration")
