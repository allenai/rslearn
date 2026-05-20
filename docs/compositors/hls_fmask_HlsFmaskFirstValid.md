## rslearn.dataset.hls_fmask.HlsFmaskFirstValid

Ranks each item in a group using HLS `Fmask`/`fmask` QA bits over the requested
window, then applies FIRST_VALID in ranked order.

This compositor is HLS-specific and requires access to the QA band:

- `fmask` when using `rslearn.data_sources.nasa_hls.Hls2`
- `Fmask` when using `rslearn.data_sources.nasa_hls.Hls2S30` or `Hls2L30`

### Configuration

```jsonc
{
  "compositing_method": {
    "class_path": "rslearn.dataset.hls_fmask.HlsFmaskFirstValid",
    "init_args": {
      "fmask_band": "fmask",
      "fmask_nodata_value": 255,
      "on_missing_fmask": "error",
      "cloud_weight": 5,
      "cirrus_weight": 1
    }
  }
}
```

### Scoring

For each candidate item, the compositor reads `fmask_band` over the requested window
and computes cloudiness on valid pixels only (`fmask != fmask_nodata_value`).

It derives two fractions:

- `bit 0`: cirrus
- `bit 1`: cloud

Each fraction is:

`count(valid pixels where bit is set) / count(valid pixels)`

Then the item score is:

`score = cirrus_weight * cirrus_fraction + cloud_weight * cloud_fraction`

With defaults, this is:

`score = 1 * cirrus_fraction + 5 * cloud_fraction`

Lower score is better.

Ranking is best-to-worst by score, then `FIRST_VALID` is applied in that ranked order.
If an item has no valid pixels in the window, it is skipped for ranking.

### Execution Notes

- Ranking runs during **materialize**, not **prepare**.
- It runs only for item groups with more than one item.
- This works for both `ingest: true` and `ingest: false`.
- If `on_missing_fmask: "error"` and `fmask_band` is unavailable for an item,
  materialization raises an error.
