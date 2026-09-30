#!/usr/bin/env python3
"""Run a small, live Sentinel-3 rslearn workflow and export a GeoTIFF."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio
import shapely
from dotenv import load_dotenv
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.dataset.manage import (
    ingest_dataset_windows,
    materialize_dataset_windows,
    prepare_dataset_windows,
)
from rslearn.dataset.window_data_storage.per_item_group import (
    per_item_group_raster_dir,
)
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.raster_format import GeotiffRasterFormat


@dataclass(frozen=True)
class SourceSpec:
    """Configuration that differs between the Sentinel-3 test cases."""

    class_name: str
    band: str
    resolution: float


SOURCES = {
    "olci": SourceSpec("Sentinel3OlciEFR", "Oa08_reflectance", 300),
    "slstr-reflectance": SourceSpec("Sentinel3SlstrRBT", "S1_reflectance", 500),
    "slstr-bt": SourceSpec("Sentinel3SlstrRBT", "S8_BT", 1000),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES, default="olci")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sentinel3_burner.tif"),
        help="GeoTIFF to create (default: sentinel3_burner.tif)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="keep the intermediate rslearn dataset at this new directory",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--longitude", type=float, default=12.46)
    parser.add_argument("--latitude", type=float, default=41.89)
    parser.add_argument("--size", type=int, default=64, help="square window size")
    parser.add_argument(
        "--start", type=datetime.fromisoformat, default="2024-04-24T00:00:00+00:00"
    )
    parser.add_argument(
        "--end", type=datetime.fromisoformat, default="2024-04-30T23:59:59+00:00"
    )
    return parser.parse_args()


def require_credentials(env_file: Path) -> None:
    """Load CDSE credentials without logging their values."""
    if env_file.exists():
        load_dotenv(env_file, override=False)
    has_token = bool(os.environ.get("COPERNICUS_ACCESS_TOKEN"))
    has_login = bool(
        os.environ.get("COPERNICUS_USERNAME") and os.environ.get("COPERNICUS_PASSWORD")
    )
    if not has_token and not has_login:
        raise SystemExit(
            "Set COPERNICUS_ACCESS_TOKEN or COPERNICUS_USERNAME and "
            f"COPERNICUS_PASSWORD in the environment or {env_file}."
        )


def create_dataset(root: Path, args: argparse.Namespace, spec: SourceSpec) -> Dataset:
    """Create a one-layer dataset and one spatiotemporal window."""
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "layers": {
            "sentinel3": {
                "type": "raster",
                "band_sets": [
                    {
                        "bands": [spec.band],
                        "dtype": "float32",
                        "nodata_value": -9999.0,
                    }
                ],
                "data_source": {
                    "class_path": (
                        "rslearn.data_sources.copernicus." + spec.class_name
                    ),
                    "init_args": {"band_names": [spec.band], "timeout": 120},
                    "query_config": {
                        "space_mode": "INTERSECTS",
                        "max_matches": 1,
                        "min_matches": 1,
                    },
                },
            }
        }
    }
    with (root / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    dataset = Dataset(UPath(root))
    projection = get_utm_ups_projection(
        args.longitude, args.latitude, spec.resolution, -spec.resolution
    )
    center = STGeometry(
        WGS84_PROJECTION, shapely.Point(args.longitude, args.latitude), None
    ).to_projection(projection)
    center_x, center_y = int(center.shp.x), int(center.shp.y)
    half = args.size // 2
    bounds = (
        center_x - half,
        center_y - half,
        center_x - half + args.size,
        center_y - half + args.size,
    )
    Window(
        storage=dataset.storage,
        group="burner",
        name=args.source,
        projection=projection,
        bounds=bounds,
        time_range=(args.start.astimezone(UTC), args.end.astimezone(UTC)),
        data_factory=dataset.window_data_storage_factory,
    ).save()
    return dataset


def run_workflow(dataset: Dataset, spec: SourceSpec, output: Path) -> None:
    """Prepare, ingest, materialize, validate, and export the result."""
    windows = dataset.load_windows(groups=["burner"])
    print("Preparing window against the public CDSE catalogue...", flush=True)
    prepare_dataset_windows(dataset, windows)
    print("Downloading and ingesting the selected SAFE product...", flush=True)
    ingest_dataset_windows(dataset, windows)
    print("Materializing the rslearn window...", flush=True)
    materialize_dataset_windows(dataset, windows)

    raster_dir = per_item_group_raster_dir(
        windows[0].window_root, "sentinel3", [spec.band], group_idx=0
    )
    source_tif = Path(str(raster_dir / GeotiffRasterFormat.fname))
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_tif, output)

    with rasterio.open(output) as dataset_reader:
        array = dataset_reader.read(1)
        valid = np.isfinite(array)
        if dataset_reader.nodata is not None:
            valid &= array != dataset_reader.nodata
        values = array[valid]
        if not values.size:
            raise RuntimeError("materialized GeoTIFF contains no valid pixels")
        print(
            f"Saved {output} ({dataset_reader.width}x{dataset_reader.height}, "
            f"{dataset_reader.crs}, {values.size} valid pixels, "
            f"range {values.min():.6g} to {values.max():.6g})",
            flush=True,
        )


def main() -> None:
    """Run the burner workflow."""
    args = parse_args()
    if args.size <= 0:
        raise SystemExit("--size must be positive")
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    if args.dataset_dir is not None and args.dataset_dir.exists():
        raise SystemExit(f"dataset directory already exists: {args.dataset_dir}")
    require_credentials(args.env_file)
    spec = SOURCES[args.source]

    if args.dataset_dir is not None:
        dataset = create_dataset(args.dataset_dir, args, spec)
        run_workflow(dataset, spec, args.output)
        print(f"Kept intermediate dataset at {args.dataset_dir.resolve()}")
        return

    with tempfile.TemporaryDirectory(prefix="rslearn-sentinel3-burner-") as tmp_dir:
        dataset = create_dataset(Path(tmp_dir), args, spec)
        run_workflow(dataset, spec, args.output)


if __name__ == "__main__":
    main()
