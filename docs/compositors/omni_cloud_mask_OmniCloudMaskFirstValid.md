## rslearn.dataset.omni_cloud_mask.OmniCloudMaskFirstValid

Ranks each item in a group using pixel-level
[`omnicloudmask` (OmniCloudMask)](https://github.com/DPIRD-DMA/OmniCloudMask)
inference over the requested window, then applies FIRST_VALID in ranked order.

### Configuration

Configure as a custom compositor in `compositing_method`:

```jsonc
{
  "compositing_method": {
    "class_path": "rslearn.dataset.omni_cloud_mask.OmniCloudMaskFirstValid",
    "init_args": {
      "red_band": "B04",
      "green_band": "B03",
      "nir_band": "B8A",
      "scoring_resolution": 20.0,
      "clear_weight": 0,
      "thick_cloud_weight": 5,
      "thin_cloud_weight": 1,
      "cloud_shadow_weight": 1
    }
  }
}
```

### Scoring

- OmniCloudMask classes: `0=clear`, `1=thick cloud`, `2=thin cloud`,
  `3=cloud shadow`.
- Score is a weighted sum of class fractions.
- Lower score is better.
- With defaults: `5*thick + thin + shadow` (`clear_weight=0`).
- If `scoring_resolution` is unset, ranking is evaluated on each band set's
  materialization grid.
- If `scoring_resolution` is set, ranking is evaluated once on a window-level
  grid at that resolution and reused across band sets.
- For Sentinel-2 with `nir_band="B8A"`, `scoring_resolution: 20.0` is a good
  speed-focused choice. It is typically faster, but can trade away a small
  amount of ranking accuracy compared to scoring on the materialization grid.
- For finer-than-10 m sensors, a good explicit choice is often
  `scoring_resolution: 10.0`.
- For coarser-than-10 m sensors, a good explicit choice is usually the native
  resolution.

### Execution Notes

- Ranking runs during **materialize**, not **prepare**.
- It runs only for item groups with more than one item.
- This works for both `ingest: true` and `ingest: false`.
- The scoring bands are read in addition to output bands.
- For reliable ranking quality, create windows with at least `96x96` pixels.
- `min_inference_size` only pads small windows; it does not add context from
  outside the window, so very small windows can still have lower accuracy.

Requires the optional `omnicloudmask` package (`pip install .[extra]` in this repo).
