"""Create stadium windows with label rasters from CSV.

Assigns each stadium to train or val based on a hash of the state name (~80/20 split).
Skips stadiums in WA (reserved for prediction).
"""

import argparse
import csv
import hashlib
from datetime import UTC, datetime

import numpy as np
import shapely
import tqdm
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset.storage.file import FileWindowStorage
from rslearn.dataset.window import Window
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat

WINDOW_SIZE = 256
RESOLUTION = 10
TIME_RANGE = (
    datetime(2025, 6, 1, tzinfo=UTC),
    datetime(2025, 8, 30, tzinfo=UTC),
)
VAL_RATIO = 0.2


def state_to_split(state: str) -> str:
    """Convert from two-letter state to the train/val split."""
    h = int(hashlib.sha256(state.encode()).hexdigest(), 16)
    return "val" if (h % 100) < VAL_RATIO * 100 else "train"


def main() -> None:
    """Main entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_path", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--group", type=str, default="default")
    args = parser.parse_args()

    ds_path = UPath(args.ds_path)
    storage = FileWindowStorage(ds_path)
    raster_format = GeotiffRasterFormat()
    half = WINDOW_SIZE // 2

    with open(args.csv_path) as f:
        rows = [row for row in csv.DictReader(f) if row["state"].strip() != "WA"]

    split_counts: dict[str, int] = {}
    for row in tqdm.tqdm(rows):
        lon = float(row["longitude"])
        lat = float(row["latitude"])
        team = row["team"].strip().replace(" ", "_")
        split = state_to_split(row["state"].strip())
        split_counts[split] = split_counts.get(split, 0) + 1

        projection = get_utm_ups_projection(lon, lat, RESOLUTION, -RESOLUTION)
        src_geom = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None)
        dst_geom = src_geom.to_projection(projection)
        cx, cy = int(dst_geom.shp.x), int(dst_geom.shp.y)
        bounds = (cx - half, cy - half, cx + half, cy + half)

        window = Window(
            storage=storage,
            group=args.group,
            name=team,
            projection=projection,
            bounds=bounds,
            time_range=TIME_RANGE,
            options={"split": split},
        )
        window.save()

        # Build label raster: 2=background, 1=center 5x5, 0=nodata ring 40x40 excluding 5x5.
        label = np.full((1, WINDOW_SIZE, WINDOW_SIZE), 2, dtype=np.uint8)
        mid = WINDOW_SIZE // 2
        label[:, mid - 20 : mid + 20, mid - 20 : mid + 20] = 0
        label[:, mid - 2 : mid + 3, mid - 2 : mid + 3] = 1

        raster_dir = window.get_raster_dir("label", ["label"])
        raster_format.encode_raster(
            raster_dir, projection, bounds, RasterArray(chw_array=label)
        )
        window.mark_layer_completed("label")

    print(f"Done: {split_counts}")


if __name__ == "__main__":
    main()
