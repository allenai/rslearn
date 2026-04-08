#!/usr/bin/env python3
"""Compare Sentinel-2 bi-weekly best scenes using SCL vs OmniCloudMask ranking.

This script:
- queries the same Sentinel-2 candidate pool per window (eo:cloud_cover < threshold),
- computes two cloudiness scores per candidate over the AOI:
  - Sentinel-2 SCL score (lower is better),
  - OmniCloudMask score (lower is better),
- picks the best candidate per method,
- saves one 2xN figure per AOI.

Row 0: best by SCL score
Row 1: best by OmniCloudMask score
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import rasterio
from omnicloudmask import predict_from_array
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from shapely.geometry import box

from rslearn.config import QueryConfig, SpaceMode
from rslearn.data_sources.planetary_computer import Sentinel2
from rslearn.data_sources.stac import SourceItem
from rslearn.utils.geometry import Projection, STGeometry


@dataclass
class AOI:
    """A fixed AOI in UTM zone 30N."""

    name: str
    slug: str
    center_x: float
    center_y: float
    half_size: float = 250.0  # meters


@dataclass
class CandidateMetrics:
    """Scores for one candidate scene."""

    item: SourceItem
    scl_score: float
    ocm_score: float


AOIS = [
    AOI("Ely, Cambridgeshire", "ely", 638_000, 5_808_000),
    AOI("North York Moors", "nymoors", 650_000, 6_015_000),
    AOI("Dartmoor, Devon", "dartmoor", 429_000, 5_589_000),
    AOI("Rannoch Moor, Scotland", "rannoch", 402_000, 6_270_000),
    AOI("Brecon Beacons, Wales", "brecon", 466_000, 5_739_000),
    AOI("Lake District", "lakedistrict", 500_000, 6_020_000),
    AOI("Snowdonia", "snowdonia", 430_000, 5_930_000),
    AOI("Norfolk Broads", "broads", 660_000, 5_860_000),
    AOI("Pembrokeshire", "pembrokeshire", 380_000, 5_730_000),
    AOI("South Downs", "southdowns", 500_000, 5_640_000),
    AOI("Yorkshire Dales", "ydales", 530_000, 5_990_000),
    AOI("Cairngorms", "cairngorms", 470_000, 6_350_000),
    AOI("Outer Hebrides", "hebrides", 300_000, 6_400_000),
    AOI("Donegal Hills", "donegal", 320_000, 6_100_000),
    AOI("Kerry Mountains", "kerry", 410_000, 5_760_000),
    AOI("Connemara", "connemara", 360_000, 5_950_000),
    AOI("Mayo Bogs", "mayo", 330_000, 6_030_000),
    AOI("Shropshire Hills", "shropshire", 420_000, 5_820_000),
    AOI("Lincolnshire Wolds", "lincolnwolds", 590_000, 5_930_000),
    AOI("Galloway Forest", "galloway", 380_000, 6_130_000),
    AOI("Suffolk Coast", "suffolkcoast", 640_000, 5_750_000),
    AOI("Anglesey", "anglesey", 400_000, 5_960_000),
    AOI("Mourne Mountains", "mournes", 360_000, 6_020_000),
    AOI("Sligo Uplands", "sligo", 350_000, 6_010_000),
    AOI("Wicklow Mountains", "wicklow", 470_000, 5_940_000),
]

CRS_UTM30N = CRS.from_epsg(32630)
PROJ_UTM30N = Projection(CRS_UTM30N, x_resolution=20.0, y_resolution=-20.0)


def biweekly_windows(year: int) -> list[tuple[datetime, datetime]]:
    """Return bi-weekly windows over the requested year in UTC."""
    start = datetime(year, 1, 1, tzinfo=UTC)
    end = datetime(year + 1, 1, 1, tzinfo=UTC)
    windows: list[tuple[datetime, datetime]] = []
    t = start
    while t < end:
        t_next = min(t + timedelta(days=14), end)
        windows.append((t, t_next))
        t = t_next
    return windows


def aoi_bounds(aoi: AOI) -> tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) in UTM30N meters."""
    return (
        aoi.center_x - aoi.half_size,
        aoi.center_y - aoi.half_size,
        aoi.center_x + aoi.half_size,
        aoi.center_y + aoi.half_size,
    )


def aoi_geometry(aoi: AOI, start: datetime, end: datetime) -> STGeometry:
    """Build rslearn STGeometry in projection pixel coordinates."""
    minx, miny, maxx, maxy = aoi_bounds(aoi)
    shp = box(
        minx / 20.0,
        maxy / -20.0,  # y_resolution is negative
        maxx / 20.0,
        miny / -20.0,
    )
    return STGeometry(PROJ_UTM30N, shp, (start, end))


def get_candidates(
    ds: Sentinel2,
    aoi: AOI,
    start: datetime,
    end: datetime,
    max_candidates: int,
) -> list[SourceItem]:
    """Fetch candidate scenes that CONTAIN the AOI window."""
    geom = aoi_geometry(aoi, start, end)
    query_config = QueryConfig(space_mode=SpaceMode.CONTAINS, max_matches=max_candidates)
    groups = ds.get_items([geom], query_config)[0]
    items = [group.items[0] for group in groups if group.items]

    # Deduplicate by scene ID while preserving order.
    seen: set[str] = set()
    deduped: list[SourceItem] = []
    for item in items:
        if item.name in seen:
            continue
        deduped.append(item)
        seen.add(item.name)
    return deduped


def read_bands(
    item: SourceItem,
    aoi: AOI,
    asset_keys: list[str],
    size: int,
    resampling: Resampling,
) -> np.ndarray:
    """Read requested bands over AOI to (C, H, W) float32."""
    minx, miny, maxx, maxy = aoi_bounds(aoi)
    out_transform = from_bounds(minx, miny, maxx, maxy, size, size)
    bands: list[np.ndarray] = []
    for key in asset_keys:
        if key not in item.asset_urls:
            raise KeyError(f"item {item.name} missing asset '{key}'")
        url = planetary_computer.sign(item.asset_urls[key])
        with rasterio.open(url) as src:
            with WarpedVRT(
                src,
                crs=CRS_UTM30N,
                transform=out_transform,
                width=size,
                height=size,
                resampling=resampling,
            ) as vrt:
                bands.append(vrt.read(1).astype(np.float32))
    return np.stack(bands, axis=0)


def scl_score(item: SourceItem, aoi: AOI, size: int) -> float:
    """Compute Sentinel-2 SCL score (lower is better).

    Default weighting mirrors Sentinel2SCLFirstValid:
    score = shadow + medium_cloud + 5*high_cloud + cirrus + snow_ice
    """
    scl = read_bands(
        item=item,
        aoi=aoi,
        asset_keys=["SCL"],
        size=size,
        resampling=Resampling.nearest,
    )[0]

    shadow_frac = float((scl == 3).mean())
    medium_cloud_frac = float((scl == 8).mean())
    high_cloud_frac = float((scl == 9).mean())
    cirrus_frac = float((scl == 10).mean())
    snow_ice_frac = float((scl == 11).mean())

    return (
        shadow_frac
        + medium_cloud_frac
        + 5.0 * high_cloud_frac
        + cirrus_frac
        + snow_ice_frac
    )


def omnicloudmask_score(item: SourceItem, aoi: AOI, size: int) -> float:
    """Compute OmniCloudMask score (lower is better).

    Default weighting mirrors OmniCloudMaskFirstValid:
    score = 5*thick_cloud + thin_cloud + cloud_shadow
    """
    scene = read_bands(
        item=item,
        aoi=aoi,
        asset_keys=["B04", "B03", "B8A"],
        size=size,
        resampling=Resampling.bilinear,
    )
    _, h, w = scene.shape
    if h < 32 or w < 32:
        scene = np.pad(
            scene,
            ((0, 0), (0, max(0, 32 - h)), (0, max(0, 32 - w))),
            mode="constant",
        )

    mask = predict_from_array(input_array=scene)
    mask = mask[:h, :w]

    thick_frac = float((mask == 1).mean())
    thin_frac = float((mask == 2).mean())
    shadow_frac = float((mask == 3).mean())
    return 5.0 * thick_frac + thin_frac + shadow_frac


def to_rgb(chw: np.ndarray) -> np.ndarray:
    """Percentile-stretch (3, H, W) to (H, W, 3) in [0, 1]."""
    rgb = chw.transpose(1, 2, 0)
    valid = rgb[rgb > 0]
    lo, hi = (np.percentile(valid, [2, 98]) if valid.size else (0.0, 1.0))
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def summarize_item(item: SourceItem) -> str:
    """Build a compact label with date + tile."""
    if item.geometry.time_range is not None:
        date_str = item.geometry.time_range[0].strftime("%d %b")
    else:
        date_str = item.name[:8]
    parts = item.name.split("_")
    tile_id = next((p for p in parts if p.startswith("T") and len(p) == 6), "")
    return f"{date_str} {tile_id}".strip()


def process_aoi(
    aoi: AOI,
    ds: Sentinel2,
    windows: list[tuple[datetime, datetime]],
    score_px: int,
    thumb_px: int,
    max_windows: int,
    max_candidates: int,
    out_dir: Path,
) -> None:
    """Compare SCL vs OmniCloudMask ranking for one AOI and save one figure.

    Only windows where the selected scene differs are plotted.
    """
    print(f"\n{'=' * 60}")
    print(aoi.name)
    print(f"{'=' * 60}")

    selected_windows = windows
    if max_windows > 0 and len(windows) > max_windows:
        step = len(windows) / max_windows
        selected_windows = [windows[int(i * step)] for i in range(max_windows)]

    rows: list[dict[str, object]] = []
    same_count = 0
    diff_count = 0
    for start, end in selected_windows:
        label = f"{start.strftime('%d')}–{end.strftime('%d %b')}"
        print(f"  {label}: querying...", end="", flush=True)

        candidates = get_candidates(ds, aoi, start, end, max_candidates=max_candidates)
        print(f" {len(candidates)} candidates", end="", flush=True)
        if not candidates:
            print(" (skip)")
            continue

        scored: list[CandidateMetrics] = []
        for item in candidates:
            try:
                scored.append(
                    CandidateMetrics(
                        item=item,
                        scl_score=scl_score(item, aoi, size=score_px),
                        ocm_score=omnicloudmask_score(item, aoi, size=score_px),
                    )
                )
            except Exception as exc:
                print(f"\n    skip {item.name}: {type(exc).__name__}: {exc}")

        if not scored:
            print(" (no valid candidates)")
            continue

        best_scl = min(scored, key=lambda m: m.scl_score)
        best_ocm = min(scored, key=lambda m: m.ocm_score)
        same_selection = best_scl.item.name == best_ocm.item.name
        print(
            f" -> best SCL={best_scl.scl_score:.3f}, best OCM={best_ocm.ocm_score:.3f}"
        )
        if same_selection:
            same_count += 1
            print("    same scene selected, skip plot row")
            continue

        diff_count += 1
        rows.append(
            {
                "label": label,
                "scl": best_scl,
                "ocm": best_ocm,
            }
        )

    if not rows:
        print(
            "  No differing selections for this AOI "
            f"(same={same_count}, different={diff_count})."
        )
        return

    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.1, 5.0))
    if n == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        "Sentinel-2 L2A comparison: windows where SCL and OmniCloudMask disagree\n"
        f"{aoi.name} · Same candidate pool per window",
        fontsize=9,
    )

    for col, row in enumerate(rows):
        label = row["label"]
        scl_best = row["scl"]
        ocm_best = row["ocm"]
        assert isinstance(label, str)
        assert isinstance(scl_best, CandidateMetrics)
        assert isinstance(ocm_best, CandidateMetrics)

        for r, (best, method_name) in enumerate(
            [
                (scl_best, "SCL"),
                (ocm_best, "OmniCloudMask"),
            ]
        ):
            ax = axes[r, col]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(label, fontsize=7)
            if col == 0:
                ax.set_ylabel(method_name, fontsize=8)

            try:
                chw = read_bands(
                    best.item,
                    aoi,
                    asset_keys=["B04", "B03", "B02"],
                    size=thumb_px,
                    resampling=Resampling.bilinear,
                )
                ax.imshow(to_rgb(chw), origin="upper")
                ax.set_title(summarize_item(best.item), fontsize=6, pad=2)
                score_text = f"scl={best.scl_score:.3f}\nocm={best.ocm_score:.3f}"
                ax.text(
                    0.01,
                    0.01,
                    score_text,
                    transform=ax.transAxes,
                    va="bottom",
                    ha="left",
                    fontsize=6,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.45, "pad": 2},
                )
            except Exception as exc:
                ax.set_facecolor("#1a1a1a")
                ax.text(
                    0.5,
                    0.5,
                    f"error\n{type(exc).__name__}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=5,
                    transform=ax.transAxes,
                )
                print(f"  warning {label} row {r}: {exc}")

    plt.tight_layout()
    out_path = out_dir / f"sentinel2_scl_vs_omnicloudmask_{aoi.slug}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(
        f"  saved -> {out_path} ({n} differing windows, same={same_count}, different={diff_count})"
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Compare Sentinel-2 SCL vs OmniCloudMask scene ranking."
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--cloud-cover-lt", type=float, default=80.0)
    parser.add_argument("--score-px", type=int, default=128)
    parser.add_argument("--thumb-px", type=int, default=256)
    parser.add_argument("--max-windows", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    """Run comparison across configured AOIs."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cloud_filter = {"eo:cloud_cover": {"lt": args.cloud_cover_lt}}
    ds = Sentinel2(
        assets=["B04", "B03", "B02", "B8A"],
        query=cloud_filter,
        # Do not pre-sort by cloud metric; we want a shared candidate pool.
        sort_by=None,
    )

    windows = biweekly_windows(args.year)
    for aoi in AOIS:
        process_aoi(
            aoi=aoi,
            ds=ds,
            windows=windows,
            score_px=args.score_px,
            thumb_px=args.thumb_px,
            max_windows=args.max_windows,
            max_candidates=args.max_candidates,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
