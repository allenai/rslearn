"""Align prepared rslearn item groups between layers by item name.

This example utility runs after ``rslearn dataset prepare`` and before materialization.
It rewrites one or more target layers' prepared item groups so their group order follows
a reference layer. Each target group is selected by formatting the first item name in
the corresponding reference group.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from upath import UPath

from rslearn.dataset import Dataset, WindowLayerData


@dataclass
class CandidateGroup:
    """A prepared item group that can be selected for a target layer."""

    serialized_group: list[dict[str, Any]]
    group_time_range: tuple[datetime, datetime] | None
    names: set[str]


def get_item_name(serialized_item: dict[str, Any]) -> str:
    """Return a serialized item's name."""
    item_name = serialized_item.get("name")
    if not isinstance(item_name, str) or not item_name:
        raise ValueError("serialized item is missing a non-empty string name")
    return item_name


def make_candidates(layer_data: WindowLayerData) -> list[CandidateGroup]:
    """Build target candidates from prepared layer data."""
    candidates = []
    for group_idx, serialized_group in enumerate(layer_data.serialized_item_groups):
        group_time_range = (
            layer_data.group_time_ranges[group_idx]
            if layer_data.group_time_ranges is not None
            else None
        )
        names = {
            item["name"]
            for item in serialized_group
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        candidates.append(
            CandidateGroup(
                serialized_group=serialized_group,
                group_time_range=group_time_range,
                names=names,
            )
        )
    return candidates


def format_target_name(
    reference_group: list[dict[str, Any]],
    target_layer: str,
    target_name_template: str,
) -> str:
    """Format the target item name expected for a reference group."""
    if not reference_group:
        raise ValueError("reference group is empty")
    reference_item_name = get_item_name(reference_group[0])
    try:
        return target_name_template.format(
            reference_item_name=reference_item_name,
            target_layer=target_layer,
            target_layer_upper=target_layer.upper(),
        )
    except KeyError as e:
        supported_fields = [
            "reference_item_name",
            "target_layer",
            "target_layer_upper",
        ]
        raise ValueError(
            f"unknown target_name_template field {e.args[0]!r}; supported fields "
            f"are {supported_fields}"
        ) from e


def name_template_match(
    reference_group: list[dict[str, Any]],
    candidates: list[CandidateGroup],
    target_layer: str,
    target_name_template: str,
) -> CandidateGroup | None:
    """Find a target candidate by formatting the first reference item name."""
    target_name = format_target_name(
        reference_group,
        target_layer,
        target_name_template,
    )
    for candidate in candidates:
        if target_name in candidate.names:
            return candidate
    return None


def align_target_layer(
    reference_data: WindowLayerData,
    target_data: WindowLayerData,
    args: argparse.Namespace,
) -> WindowLayerData:
    """Align one target layer's prepared groups to a reference layer."""
    candidates = make_candidates(target_data)
    aligned_groups = []
    aligned_time_ranges = []

    for group_idx, reference_group in enumerate(reference_data.serialized_item_groups):
        candidate = name_template_match(
            reference_group,
            candidates,
            target_data.layer_name,
            args.target_name_template,
        )
        if candidate is None:
            target_name = format_target_name(
                reference_group,
                target_data.layer_name,
                args.target_name_template,
            )
            raise ValueError(
                f"no target item named {target_name!r} for reference group "
                f"{group_idx} in target layer {target_data.layer_name}"
            )

        aligned_groups.append(candidate.serialized_group)
        if args.group_time_range_source == "target":
            aligned_time_ranges.append(candidate.group_time_range)
        elif args.group_time_range_source == "reference":
            aligned_time_ranges.append(
                reference_data.group_time_ranges[group_idx]
                if reference_data.group_time_ranges is not None
                else None
            )
        elif args.group_time_range_source == "none":
            aligned_time_ranges.append(None)
        else:
            raise ValueError(
                f"unsupported group time range source: {args.group_time_range_source}"
            )

    group_time_ranges = (
        None if args.group_time_range_source == "none" else aligned_time_ranges
    )
    return WindowLayerData(
        layer_name=target_data.layer_name,
        serialized_item_groups=aligned_groups,
        group_time_ranges=group_time_ranges,
        materialized=False,
    )


def get_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Align prepared item groups in one or more target layers to a reference "
            "layer by transforming reference item names."
        )
    )
    parser.add_argument(
        "--root", required=True, help="Path to the rslearn dataset root."
    )
    parser.add_argument(
        "--reference-layer",
        required=True,
        help="Prepared layer whose item group order should be followed.",
    )
    parser.add_argument(
        "--target-layers",
        nargs="+",
        required=True,
        help="Prepared layer(s) to rewrite so they align with the reference layer.",
    )
    parser.add_argument(
        "--target-name-template",
        default="{reference_item_name}_{target_layer_upper}",
        help=(
            "Template for target item names. Supported fields: "
            "{reference_item_name}, {target_layer}, {target_layer_upper}."
        ),
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Optional window groups to process.",
    )
    parser.add_argument(
        "--windows",
        nargs="*",
        default=None,
        help="Optional window names to process.",
    )
    parser.add_argument(
        "--group-time-range-source",
        choices=["target", "reference", "none"],
        default="target",
        help=(
            "Which prepared request time ranges to write on the aligned target layer. "
            "The default keeps the matched target layer's prepared ranges."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many windows would be updated without writing changes.",
    )
    return parser


def main() -> None:
    """Run the item-group alignment utility."""
    parser = get_parser()
    args = parser.parse_args()

    dataset = Dataset(UPath(args.root))
    windows = dataset.load_windows(groups=args.groups, names=args.windows)
    updated_windows = 0

    for window in windows:
        layer_datas = window.load_layer_datas()
        if args.reference_layer not in layer_datas:
            raise ValueError(
                f"window {window.group}/{window.name} is missing prepared "
                f"reference layer {args.reference_layer}"
            )
        reference_data = layer_datas[args.reference_layer]

        changed = False
        for target_layer in args.target_layers:
            if target_layer not in layer_datas:
                raise ValueError(
                    f"window {window.group}/{window.name} is missing prepared "
                    f"target layer {target_layer}"
                )
            layer_datas[target_layer] = align_target_layer(
                reference_data,
                layer_datas[target_layer],
                args,
            )
            changed = True

        if changed:
            updated_windows += 1
            if not args.dry_run:
                window.save_layer_datas(layer_datas)

    action = "would update" if args.dry_run else "updated"
    print(f"{action} {updated_windows} windows")


if __name__ == "__main__":
    main()
