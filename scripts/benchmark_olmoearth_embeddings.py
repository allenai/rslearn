#!/usr/bin/env python3
"""Benchmark OlmoEarth embedding generation on CPU and MPS.

This script builds a synthetic rslearn dataset with Sentinel-2 inputs and benchmarks
`trainer.predict` using OlmoEarth + EmbeddingHead. It is intended for quick, local
device comparisons (e.g. `cpu` vs `mps`) without requiring remote data downloads.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
import numpy as np
import torch
from olmoearth_pretrain_minimal import ModelID
from upath import UPath

from rslearn.config import DType
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.models.olmoearth_pretrain.norm import OlmoEarthNormalize
from rslearn.models.singletask import SingleTaskModel
from rslearn.train.all_crops_dataset import IterableAllCropsDataset
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.dataset import DataInput, SplitConfig
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.train.optimizer import AdamW
from rslearn.train.tasks.embedding import EmbeddingHead, EmbeddingTask
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import NumpyRasterFormat

SENTINEL2_BANDS = [
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    "B01",
    "B09",
]


@dataclass
class BenchmarkResult:
    """Benchmark stats for one model/device pair."""

    model_id: str
    device: str
    num_runs: int
    warmup_runs: int
    num_windows: int
    timesteps: int
    window_size: int
    crop_size: int
    overlap_pixels: int
    batch_size: int
    num_predict_crops: int
    run_seconds: list[float]
    mean_seconds: float
    std_seconds: float
    min_seconds: float
    max_seconds: float
    mean_crops_per_second: float


def _write_dataset_config(dataset_root: Path) -> None:
    dataset_config = {
        "layers": {
            "sentinel2_l2a": {
                "type": "raster",
                "band_sets": [
                    {
                        "bands": SENTINEL2_BANDS,
                        "dtype": "uint16",
                        "format": {
                            "class_path": "rslearn.utils.raster_format.NumpyRasterFormat"
                        },
                    }
                ],
            }
        }
    }
    dataset_root.mkdir(parents=True, exist_ok=True)
    with (dataset_root / "config.json").open("w") as f:
        json.dump(dataset_config, f, indent=2)


def _build_synthetic_dataset(
    dataset_root: Path,
    num_windows: int,
    timesteps: int,
    window_size: int,
    seed: int,
    rebuild: bool,
) -> None:
    if rebuild and dataset_root.exists():
        shutil.rmtree(dataset_root)
    if (dataset_root / "config.json").exists() and not rebuild:
        return

    _write_dataset_config(dataset_root)
    dataset = Dataset(UPath(dataset_root))
    raster_format = NumpyRasterFormat()
    start = datetime(2024, 1, 1, tzinfo=UTC)

    for window_idx in range(num_windows):
        window = Window(
            storage=dataset.storage,
            group="default",
            name=f"window_{window_idx:03d}",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, window_size, window_size),
            time_range=(start, start + timedelta(days=365)),
        )
        window.save()

        for timestep_idx in range(timesteps):
            rng = np.random.default_rng(seed + window_idx * 10_000 + timestep_idx)
            array = rng.integers(
                low=0,
                high=10_000,
                size=(len(SENTINEL2_BANDS), window_size, window_size),
                dtype=np.uint16,
            )
            ts_start = start + timedelta(days=30 * timestep_idx)
            ts_end = ts_start + timedelta(days=1)
            raster = RasterArray(chw_array=array, time_range=(ts_start, ts_end))
            raster_dir = window.get_raster_dir(
                "sentinel2_l2a", SENTINEL2_BANDS, group_idx=timestep_idx
            )
            raster_format.encode_raster(
                path=raster_dir,
                projection=window.projection,
                bounds=window.bounds,
                raster=raster,
            )
            window.mark_layer_completed("sentinel2_l2a", group_idx=timestep_idx)


def _make_data_module(
    dataset_root: Path,
    crop_size: int,
    overlap_pixels: int,
    batch_size: int,
    num_workers: int,
) -> RslearnDataModule:
    task = EmbeddingTask()
    data_input = DataInput(
        data_type="raster",
        layers=["sentinel2_l2a"],
        bands=SENTINEL2_BANDS,
        passthrough=True,
        dtype=DType.FLOAT32,
        load_all_layers=True,
        load_all_item_groups=True,
    )
    predict_config = SplitConfig(
        groups=["default"],
        transforms=[
            OlmoEarthNormalize(
                band_names={"sentinel2_l2a": SENTINEL2_BANDS},
            )
        ],
        load_all_crops=True,
        crop_size=crop_size,
        overlap_pixels=overlap_pixels,
    )
    return RslearnDataModule(
        path=str(dataset_root),
        inputs={"sentinel2_l2a": data_input},
        task=task,
        batch_size=batch_size,
        num_workers=num_workers,
        predict_config=predict_config,
    )


def _make_module(task: EmbeddingTask, model_id: ModelID, patch_size: int) -> RslearnLightningModule:
    model = SingleTaskModel(
        encoder=[
            OlmoEarth(
                patch_size=patch_size,
                model_id=model_id,
                autocast_dtype=None,
                use_legacy_timestamps=False,
            )
        ],
        decoder=[EmbeddingHead()],
    )
    return RslearnLightningModule(
        model=model,
        task=task,
        optimizer=AdamW(),
    )


def _count_predict_crops(predict_dataset: object) -> int:
    if isinstance(predict_dataset, IterableAllCropsDataset):
        return sum(
            predict_dataset.get_window_num_crops(window.bounds)
            for window in predict_dataset.windows
        )
    return len(predict_dataset)


def _sync_if_needed(device: Literal["cpu", "mps"]) -> None:
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _benchmark_device(
    device: Literal["cpu", "mps"],
    data_module: RslearnDataModule,
    model_id: ModelID,
    patch_size: int,
    num_runs: int,
    warmup_runs: int,
    num_windows: int,
    timesteps: int,
    window_size: int,
    crop_size: int,
    overlap_pixels: int,
    batch_size: int,
) -> BenchmarkResult:
    pl_module = _make_module(data_module.task, model_id=model_id, patch_size=patch_size)
    data_module.setup("predict")
    predict_dataset = data_module.datasets["predict"]
    num_predict_crops = _count_predict_crops(predict_dataset)
    predict_dataloader = data_module.predict_dataloader()

    trainer = pl.Trainer(
        accelerator=device,
        devices=1,
        strategy="auto",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    for _ in range(warmup_runs):
        trainer.predict(
            pl_module, dataloaders=predict_dataloader, return_predictions=False
        )

    run_seconds: list[float] = []
    for _ in range(num_runs):
        _sync_if_needed(device)
        start = time.perf_counter()
        trainer.predict(
            pl_module, dataloaders=predict_dataloader, return_predictions=False
        )
        _sync_if_needed(device)
        run_seconds.append(time.perf_counter() - start)

    mean_seconds = statistics.mean(run_seconds)
    std_seconds = statistics.pstdev(run_seconds) if len(run_seconds) > 1 else 0.0
    mean_crops_per_second = num_predict_crops / mean_seconds if mean_seconds > 0 else 0.0

    return BenchmarkResult(
        model_id=model_id.name,
        device=device,
        num_runs=num_runs,
        warmup_runs=warmup_runs,
        num_windows=num_windows,
        timesteps=timesteps,
        window_size=window_size,
        crop_size=crop_size,
        overlap_pixels=overlap_pixels,
        batch_size=batch_size,
        num_predict_crops=num_predict_crops,
        run_seconds=run_seconds,
        mean_seconds=mean_seconds,
        std_seconds=std_seconds,
        min_seconds=min(run_seconds),
        max_seconds=max(run_seconds),
        mean_crops_per_second=mean_crops_per_second,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark OlmoEarth embedding generation on CPU and MPS."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("benchmarks/olmoearth_benchmark_dataset"),
        help="Path for synthetic benchmark dataset.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=["cpu", "mps"],
        default=["cpu", "mps"],
        help="Devices to benchmark.",
    )
    model_choices = [m.name for m in ModelID]
    parser.add_argument(
        "--model-id",
        type=str,
        choices=model_choices,
        default=None,
        help="Single model identifier (legacy alias for --model-ids).",
    )
    parser.add_argument(
        "--model-ids",
        nargs="+",
        choices=model_choices + ["all"],
        default=None,
        help="One or more model identifiers, or 'all'.",
    )
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--num-windows", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--overlap-pixels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-rebuild-dataset",
        action="store_true",
        help="Reuse existing dataset_root if present.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/olmoearth_benchmark_results.json"),
        help="Path to save benchmark results JSON.",
    )
    return parser


def main() -> None:
    """Run the benchmark and write summarized results."""
    args = _build_parser().parse_args()
    if args.model_id is not None and args.model_ids is not None:
        raise ValueError("Specify only one of --model-id or --model-ids.")

    if args.model_ids is None and args.model_id is None:
        model_id_names = [ModelID.OLMOEARTH_V1_NANO.name]
    elif args.model_ids is not None:
        if "all" in args.model_ids:
            model_id_names = [m.name for m in ModelID]
        else:
            model_id_names = args.model_ids
    else:
        model_id_names = [args.model_id]
    model_ids = [ModelID[name] for name in model_id_names]
    rebuild_dataset = not args.no_rebuild_dataset

    _build_synthetic_dataset(
        dataset_root=args.dataset_root,
        num_windows=args.num_windows,
        timesteps=args.timesteps,
        window_size=args.window_size,
        seed=args.seed,
        rebuild=rebuild_dataset,
    )

    chosen_devices: list[Literal["cpu", "mps"]] = []
    for device in args.devices:
        if device == "mps" and not torch.backends.mps.is_available():
            print("Skipping mps benchmark (MPS is not available).")
            continue
        chosen_devices.append(device)

    if not chosen_devices:
        raise RuntimeError("No benchmark devices available.")

    results: list[BenchmarkResult] = []
    for model_id in model_ids:
        for device in chosen_devices:
            data_module = _make_data_module(
                dataset_root=args.dataset_root,
                crop_size=args.crop_size,
                overlap_pixels=args.overlap_pixels,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            result = _benchmark_device(
                device=device,
                data_module=data_module,
                model_id=model_id,
                patch_size=args.patch_size,
                num_runs=args.num_runs,
                warmup_runs=args.warmup_runs,
                num_windows=args.num_windows,
                timesteps=args.timesteps,
                window_size=args.window_size,
                crop_size=args.crop_size,
                overlap_pixels=args.overlap_pixels,
                batch_size=args.batch_size,
            )
            results.append(result)

    system_info = {
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
    }
    payload = {
        "system_info": system_info,
        "args": {
            "dataset_root": str(args.dataset_root),
            "devices": args.devices,
            "model_ids": model_id_names,
            "patch_size": args.patch_size,
            "num_windows": args.num_windows,
            "timesteps": args.timesteps,
            "window_size": args.window_size,
            "crop_size": args.crop_size,
            "overlap_pixels": args.overlap_pixels,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "warmup_runs": args.warmup_runs,
            "num_runs": args.num_runs,
            "seed": args.seed,
        },
        "results": [asdict(result) for result in results],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(payload, f, indent=2)

    print("\nBenchmark summary")
    print(
        "| model_id | device | mean_s | std_s | min_s | max_s | crops | mean_crops_per_s |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for result in results:
        print(
            f"| {result.model_id} | {result.device} "
            f"| {result.mean_seconds:.3f} | {result.std_seconds:.3f} "
            f"| {result.min_seconds:.3f} | {result.max_seconds:.3f} "
            f"| {result.num_predict_crops} | {result.mean_crops_per_second:.2f} |"
        )
    print(f"\nWrote JSON results to {args.output_json}")


if __name__ == "__main__":
    main()
