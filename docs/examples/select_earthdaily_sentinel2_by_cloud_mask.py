"""Select EarthDaily Sentinel-2 items using prepared EDA cloud-mask clear cover.

Run this after ``rslearn dataset prepare`` and before ``rslearn dataset ingest`` or
``rslearn dataset materialize``. It scores prepared cloud-mask candidates for each
window, selects the clearest candidate over the window AOI, resolves the related
Sentinel-2 L2A item from STAC metadata, and rewrites the prepared Sentinel-2 layer to
that single selected item group.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from earthdaily import EDSClient, EDSConfig
from rasterio.enums import Resampling
from upath import UPath

from rslearn.data_sources.earthdaily import Sentinel2EDACloudMask
from rslearn.dataset import Dataset, Window, WindowLayerData


@dataclass
class ScoredCloudMask:
    """Clear-cover score for a prepared cloud-mask item."""

    serialized_group: list[dict]
    group_time_range: tuple[datetime, datetime] | None
    cloud_mask_item_name: str
    source_collection: str
    source_item_id: str
    clear_cover: float
    valid_cover: float
    clear_pixels: int
    valid_pixels: int
    total_pixels: int


def score_cloud_mask_item(
    *,
    cloud_mask_source: Sentinel2EDACloudMask,
    cloud_mask_collection: Any,
    window: Window,
    serialized_group: list[dict],
    group_time_range: tuple[datetime, datetime] | None,
) -> ScoredCloudMask:
    """Score a prepared cloud-mask group over a window."""
    if len(serialized_group) != 1:
        raise ValueError(
            "expected each cloud-mask candidate group to contain one item; configure "
            "the cloud-mask layer with query_config.space_mode='INTERSECTS'"
        )

    item = cloud_mask_source.deserialize_item(serialized_group[0])
    raster = cloud_mask_source.read_raster(
        layer_name="cloud_mask",
        item=item,
        bands=["cloud-mask"],
        projection=window.projection,
        bounds=window.bounds,
        resampling=Resampling.nearest,
    )
    cloud_mask = raster.get_chw_array()[0]

    total_pixels = cloud_mask.size
    valid_pixels = int((cloud_mask != 0).sum())
    clear_pixels = int((cloud_mask == 1).sum())
    clear_cover = clear_pixels / total_pixels if total_pixels else 0.0
    valid_cover = valid_pixels / total_pixels if total_pixels else 0.0

    stac_item = cloud_mask_collection.get_item(item.name)
    if stac_item is None:
        raise KeyError(f"cloud-mask STAC item not found: {item.name}")

    source_collection = stac_item.properties.get("eda:derived_from_collection_id")
    source_item_id = stac_item.properties.get("eda:derived_from_item_id")
    if not isinstance(source_collection, str) or not source_collection:
        raise ValueError(f"cloud-mask item {item.name} missing source collection")
    if not isinstance(source_item_id, str) or not source_item_id:
        raise ValueError(f"cloud-mask item {item.name} missing source item id")

    return ScoredCloudMask(
        serialized_group=serialized_group,
        group_time_range=group_time_range,
        cloud_mask_item_name=item.name,
        source_collection=source_collection,
        source_item_id=source_item_id,
        clear_cover=clear_cover,
        valid_cover=valid_cover,
        clear_pixels=clear_pixels,
        valid_pixels=valid_pixels,
        total_pixels=total_pixels,
    )


def score_window_cloud_masks(
    *,
    cloud_mask_source: Sentinel2EDACloudMask,
    cloud_mask_collection: Any,
    window: Window,
    cloud_mask_data: WindowLayerData,
) -> list[ScoredCloudMask]:
    """Score all prepared cloud-mask candidates for a window."""
    scored = []
    for group_idx, serialized_group in enumerate(
        cloud_mask_data.serialized_item_groups
    ):
        group_time_range = (
            cloud_mask_data.group_time_ranges[group_idx]
            if cloud_mask_data.group_time_ranges is not None
            else None
        )
        scored.append(
            score_cloud_mask_item(
                cloud_mask_source=cloud_mask_source,
                cloud_mask_collection=cloud_mask_collection,
                window=window,
                serialized_group=serialized_group,
                group_time_range=group_time_range,
            )
        )

    scored.sort(
        key=lambda candidate: (
            candidate.clear_cover,
            candidate.valid_cover,
            candidate.clear_pixels,
        ),
        reverse=True,
    )
    return scored


def get_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Select prepared EarthDaily Sentinel-2 item groups by scoring prepared "
            "sentinel-2-eda-cloud-mask candidates over each window."
        )
    )
    parser.add_argument(
        "--root", required=True, help="Path to the rslearn dataset root."
    )
    parser.add_argument(
        "--cloud-mask-layer",
        default="cloud_mask",
        help="Prepared cloud-mask layer name.",
    )
    parser.add_argument(
        "--sentinel2-layer",
        default="sentinel2",
        help="Prepared Sentinel-2 layer name to rewrite.",
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
        "--min-clear-cover",
        type=float,
        default=None,
        help="Optional minimum clear-cover fraction required to select a scene.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report selected scenes without writing prepared item groups.",
    )
    parser.add_argument(
        "--keep-cloud-mask-candidates",
        action="store_true",
        help=(
            "Only rewrite the Sentinel-2 layer. By default the cloud-mask layer is "
            "also rewritten to the selected cloud-mask item so it can be materialized "
            "alongside the selected Sentinel-2 scene."
        ),
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional dotenv file containing EarthDaily credentials.",
    )
    return parser


def main() -> None:
    """Run cloud-mask clear-cover selection."""
    args = get_parser().parse_args()
    load_dotenv(args.env_file)

    dataset = Dataset(UPath(args.root))
    if args.sentinel2_layer not in dataset.layers:
        raise ValueError(f"dataset has no layer named {args.sentinel2_layer!r}")
    if args.cloud_mask_layer not in dataset.layers:
        raise ValueError(f"dataset has no layer named {args.cloud_mask_layer!r}")

    cloud_mask_source = Sentinel2EDACloudMask(assets=["cloud-mask"], cache_dir=None)
    cloud_mask_collection = EDSClient(
        EDSConfig()
    ).platform.pystac_client.get_collection(Sentinel2EDACloudMask.COLLECTION_NAME)
    sentinel2_source: Any = dataset.layers[
        args.sentinel2_layer
    ].instantiate_data_source(dataset.path)

    windows = dataset.load_windows(groups=args.groups, names=args.windows)
    updated = 0
    for window in windows:
        layer_datas = window.load_layer_datas()
        if args.cloud_mask_layer not in layer_datas:
            raise ValueError(
                f"window {window.group}/{window.name} is missing prepared layer "
                f"{args.cloud_mask_layer}"
            )

        scored = score_window_cloud_masks(
            cloud_mask_source=cloud_mask_source,
            cloud_mask_collection=cloud_mask_collection,
            window=window,
            cloud_mask_data=layer_datas[args.cloud_mask_layer],
        )
        if not scored:
            raise ValueError(
                f"window {window.group}/{window.name} has no cloud-mask candidates"
            )

        best = scored[0]
        if args.min_clear_cover is not None and best.clear_cover < args.min_clear_cover:
            raise ValueError(
                f"window {window.group}/{window.name} best clear_cover "
                f"{best.clear_cover:.3f} is below {args.min_clear_cover:.3f}"
            )

        if getattr(sentinel2_source, "collection_name", None) != best.source_collection:
            raise ValueError(
                f"selected cloud-mask item {best.cloud_mask_item_name} derives from "
                f"{best.source_collection}, but layer {args.sentinel2_layer} uses "
                f"{getattr(sentinel2_source, 'collection_name', None)}"
            )

        sentinel2_item = sentinel2_source.get_item_by_name(best.source_item_id)
        sentinel2_time_range = sentinel2_item.geometry.time_range

        print(
            f"{window.group}/{window.name}: selected {best.source_item_id} from "
            f"{best.cloud_mask_item_name} clear_cover={best.clear_cover:.3f} "
            f"valid_cover={best.valid_cover:.3f}"
        )

        if args.dry_run:
            continue

        layer_datas[args.sentinel2_layer] = WindowLayerData(
            layer_name=args.sentinel2_layer,
            serialized_item_groups=[[sentinel2_item.serialize()]],
            group_time_ranges=[sentinel2_time_range],
            materialized=False,
        )
        if not args.keep_cloud_mask_candidates:
            layer_datas[args.cloud_mask_layer] = WindowLayerData(
                layer_name=args.cloud_mask_layer,
                serialized_item_groups=[best.serialized_group],
                group_time_ranges=[best.group_time_range],
                materialized=False,
            )
        window.save_layer_datas(layer_datas)
        updated += 1

    action = "would update" if args.dry_run else "updated"
    print(f"{action} {updated if not args.dry_run else len(windows)} windows")


if __name__ == "__main__":
    main()
