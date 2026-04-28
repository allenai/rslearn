"""Align prepared rslearn item groups between layers.

This example utility runs after ``rslearn dataset prepare`` and before materialization.
It rewrites one or more target layers' prepared item groups so their group order follows
a reference layer. It is useful when two independent data sources expose related items
that should be materialized in matching temporal slots.
"""

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pytimeparse
from upath import UPath

from rslearn.dataset import Dataset, WindowLayerData


@dataclass
class CandidateGroup:
    """A prepared item group that can be selected for a target layer."""

    serialized_group: list[dict[str, Any]]
    group_time_range: tuple[datetime, datetime] | None
    midpoint: datetime | None
    names: set[str]


def parse_timedelta(value: str) -> timedelta:
    """Parse a human-friendly duration such as ``1h`` or ``2 days``."""
    seconds = pytimeparse.parse(value)
    if seconds is None:
        raise argparse.ArgumentTypeError(f"invalid duration: {value}")
    return timedelta(seconds=seconds)


def item_time_range(
    serialized_item: dict[str, Any],
) -> tuple[datetime, datetime] | None:
    """Return an item's serialized geometry time range."""
    geometry = serialized_item.get("geometry")
    if not isinstance(geometry, dict):
        return None
    time_range = geometry.get("time_range")
    if not time_range:
        return None
    return (
        datetime.fromisoformat(time_range[0]),
        datetime.fromisoformat(time_range[1]),
    )


def time_range_midpoint(time_range: tuple[datetime, datetime]) -> datetime:
    """Return the midpoint of a time range."""
    return time_range[0] + (time_range[1] - time_range[0]) / 2


def group_midpoint(serialized_group: list[dict[str, Any]]) -> datetime | None:
    """Return the average midpoint of items in a serialized item group."""
    midpoints = []
    for serialized_item in serialized_group:
        time_range = item_time_range(serialized_item)
        if time_range is not None:
            midpoints.append(time_range_midpoint(time_range))
    if not midpoints:
        return None

    timestamps = [midpoint.timestamp() for midpoint in midpoints]
    return datetime.fromtimestamp(
        sum(timestamps) / len(timestamps), tz=midpoints[0].tzinfo
    )


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
                midpoint=group_midpoint(serialized_group),
                names=names,
            )
        )
    return candidates


def nearest_timestamp_match(
    reference_group: list[dict[str, Any]],
    candidates: list[CandidateGroup],
    max_delta: timedelta,
) -> CandidateGroup | None:
    """Find the target candidate closest in time to a reference group."""
    reference_midpoint = group_midpoint(reference_group)
    if reference_midpoint is None:
        return None

    best_candidate = None
    best_delta = None
    for candidate in candidates:
        if candidate.midpoint is None:
            continue
        delta = abs(candidate.midpoint - reference_midpoint)
        if best_delta is None or delta < best_delta:
            best_candidate = candidate
            best_delta = delta

    if best_candidate is None or best_delta is None:
        return None
    if best_delta > max_delta:
        return None
    return best_candidate


def name_regex_match(
    reference_group: list[dict[str, Any]],
    candidates: list[CandidateGroup],
    reference_name_regex: str,
    target_name_replacement: str,
) -> CandidateGroup | None:
    """Find a target candidate by transforming the first reference item name."""
    if not reference_group:
        return None
    reference_name = reference_group[0].get("name")
    if not isinstance(reference_name, str):
        return None
    target_name = re.sub(reference_name_regex, target_name_replacement, reference_name)
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
        if args.match_mode == "nearest_timestamp":
            candidate = nearest_timestamp_match(
                reference_group, candidates, args.max_delta
            )
        elif args.match_mode == "name_regex":
            candidate = name_regex_match(
                reference_group,
                candidates,
                args.reference_name_regex,
                args.target_name_replacement,
            )
        else:
            raise ValueError(f"unsupported match mode: {args.match_mode}")

        if candidate is None:
            raise ValueError(
                f"no {args.match_mode} match for reference group {group_idx} "
                f"in target layer {target_data.layer_name}"
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
            "layer by nearest timestamp or name transformation."
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
        "--match-mode",
        choices=["nearest_timestamp", "name_regex"],
        default="nearest_timestamp",
        help="How to select a target group for each reference group.",
    )
    parser.add_argument(
        "--max-delta",
        type=parse_timedelta,
        default=timedelta(days=1),
        help="Maximum timestamp difference for nearest_timestamp matching.",
    )
    parser.add_argument(
        "--reference-name-regex",
        default=None,
        help="Regex to replace in the first reference item name for name_regex matching.",
    )
    parser.add_argument(
        "--target-name-replacement",
        default=None,
        help="Replacement string used with --reference-name-regex.",
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
    if args.match_mode == "name_regex" and (
        args.reference_name_regex is None or args.target_name_replacement is None
    ):
        parser.error(
            "--reference-name-regex and --target-name-replacement are required for "
            "--match-mode name_regex"
        )

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
