## OlmoEarth CPU/MPS Benchmarks

This benchmark measures `rslearn` embedding generation throughput for all
OlmoEarth v1 model sizes (`NANO`, `TINY`, `BASE`, `LARGE`) on local `cpu` and
`mps` devices using the same synthetic Sentinel-2 workload.

## Benchmark Script

Use:

```bash
uv run python scripts/benchmark_olmoearth_embeddings.py \
  --devices cpu mps \
  --model-ids all \
  --num-runs 2 \
  --warmup-runs 1 \
  --num-windows 1 \
  --timesteps 2 \
  --window-size 128 \
  --crop-size 64 \
  --overlap-pixels 32 \
  --batch-size 1 \
  --num-workers 0 \
  --output-json benchmarks/olmoearth_all_models_cpu_mps_benchmark_2026-04-01.json
```

The script:

1. Creates a synthetic rslearn dataset at `benchmarks/olmoearth_benchmark_dataset`.
2. Runs `trainer.predict` with `EmbeddingTask` and `OlmoEarthNormalize`.
3. Benchmarks per-device wall-clock time across repeated runs (after one setup pass).
4. Writes machine metadata and results to JSON.

`--model-id` is still supported for single-model runs, but `--model-ids` is
recommended for side-by-side comparisons.

## Results (April 1, 2026)

Machine/runtime:

- `platform`: `macOS-15.7.4-arm64-arm-64bit`
- `python`: `3.11.13`
- `torch`: `2.9.1`
- `mps_available`: `true`

Benchmark setup:

- Models: `OLMOEARTH_V1_NANO`, `OLMOEARTH_V1_TINY`, `OLMOEARTH_V1_BASE`, `OLMOEARTH_V1_LARGE` (`patch_size=4`)
- Windows: `1`
- Timesteps/window: `2`
- Window size: `128x128`
- Predict crop size: `64`
- Overlap: `32`
- Batch size: `1`
- Predict crops/run: `9`
- Warmup runs: `1`
- Timed runs: `2`

Measured timings:

| Model | Device | Mean (s) | Std (s) | Min (s) | Max (s) | Mean crops/s |
|---|---|---:|---:|---:|---:|---:|
| `OLMOEARTH_V1_NANO` | cpu | 0.357 | 0.014 | 0.343 | 0.371 | 25.19 |
| `OLMOEARTH_V1_NANO` | mps | 0.349 | 0.010 | 0.339 | 0.359 | 25.76 |
| `OLMOEARTH_V1_TINY` | cpu | 0.831 | 0.020 | 0.811 | 0.851 | 10.83 |
| `OLMOEARTH_V1_TINY` | mps | 0.551 | 0.020 | 0.531 | 0.571 | 16.32 |
| `OLMOEARTH_V1_BASE` | cpu | 4.134 | 0.039 | 4.094 | 4.173 | 2.18 |
| `OLMOEARTH_V1_BASE` | mps | 1.803 | 0.008 | 1.795 | 1.810 | 4.99 |
| `OLMOEARTH_V1_LARGE` | cpu | 13.557 | 0.041 | 13.516 | 13.598 | 0.66 |
| `OLMOEARTH_V1_LARGE` | mps | 5.495 | 0.018 | 5.478 | 5.513 | 1.64 |

Relative speedup:

- `OLMOEARTH_V1_NANO`: `1.02x` faster on `mps` (`25.76 / 25.19`)
- `OLMOEARTH_V1_TINY`: `1.51x` faster on `mps` (`16.32 / 10.83`)
- `OLMOEARTH_V1_BASE`: `2.29x` faster on `mps` (`4.99 / 2.18`)
- `OLMOEARTH_V1_LARGE`: `2.47x` faster on `mps` (`1.64 / 0.66`)

Raw results are stored in:

- `benchmarks/olmoearth_all_models_cpu_mps_benchmark_2026-04-01.json`
