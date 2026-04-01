## OlmoEarth CPU/MPS Benchmarks

This benchmark measures `rslearn` embedding generation throughput for all
OlmoEarth v1 model sizes (`NANO`, `TINY`, `BASE`, `LARGE`) on local `cpu` and
`mps` devices using the same synthetic Sentinel-2 workload.

## Benchmark Script

Use:

```bash
DS=$(mktemp -d /tmp/olmoearth_bench_500px_XXXXXX)
uv run python scripts/benchmark_olmoearth_embeddings.py \
  --dataset-root "$DS" \
  --devices cpu mps \
  --model-ids all \
  --num-runs 1 \
  --warmup-runs 1 \
  --num-windows 1 \
  --timesteps 1 \
  --window-size 500 \
  --crop-size 64 \
  --overlap-pixels 32 \
  --batch-size 1 \
  --num-workers 0 \
  --output-json benchmarks/olmoearth_all_models_cpu_mps_500px_2026-04-01.json
```

The script:

1. Creates a synthetic rslearn dataset in a temporary directory under `/tmp`.
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
- Timesteps/window: `1`
- Window size: `500x500`
- Predict crop size: `64`
- Overlap: `32`
- Batch size: `1`
- Predict crops/run: `225`
- Warmup runs: `1`
- Timed runs: `1`

Measured timings:

| Model | Device | Mean (s) | Std (s) | Min (s) | Max (s) | Mean crops/s |
|---|---|---:|---:|---:|---:|---:|
| `OLMOEARTH_V1_NANO` | cpu | 3.911 | 0.000 | 3.911 | 3.911 | 57.54 |
| `OLMOEARTH_V1_NANO` | mps | 3.141 | 0.000 | 3.141 | 3.141 | 71.62 |
| `OLMOEARTH_V1_TINY` | cpu | 11.656 | 0.000 | 11.656 | 11.656 | 19.30 |
| `OLMOEARTH_V1_TINY` | mps | 4.694 | 0.000 | 4.694 | 4.694 | 47.93 |
| `OLMOEARTH_V1_BASE` | cpu | 42.507 | 0.000 | 42.507 | 42.507 | 5.29 |
| `OLMOEARTH_V1_BASE` | mps | 17.937 | 0.000 | 17.937 | 17.937 | 12.54 |
| `OLMOEARTH_V1_LARGE` | cpu | 134.551 | 0.000 | 134.551 | 134.551 | 1.67 |
| `OLMOEARTH_V1_LARGE` | mps | 48.916 | 0.000 | 48.916 | 48.916 | 4.60 |

`Mean crops/s` is throughput: the average number of inference crops processed per
second. It is computed as:

`num_predict_crops / mean_seconds`

For this run, `num_predict_crops=225` per window.

Relative speedup:

- `OLMOEARTH_V1_NANO`: `1.24x` faster on `mps` (`71.62 / 57.54`)
- `OLMOEARTH_V1_TINY`: `2.48x` faster on `mps` (`47.93 / 19.30`)
- `OLMOEARTH_V1_BASE`: `2.37x` faster on `mps` (`12.54 / 5.29`)
- `OLMOEARTH_V1_LARGE`: `2.75x` faster on `mps` (`4.60 / 1.67`)

Raw results are stored in:

- `benchmarks/olmoearth_all_models_cpu_mps_500px_2026-04-01.json`
