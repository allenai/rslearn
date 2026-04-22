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
      "use_canonical_b8a_20m_grid": true,
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
- By default, if `nir_band="B8A"`, scoring is run once on a canonical 20 m
  RGB+NIR grid and that item ordering is reused across all band-set
  materialization passes for the window.
- Set `use_canonical_b8a_20m_grid: false` to disable that reuse and score on
  each band set's materialization grid instead.
- Setting `use_canonical_b8a_20m_grid: true` requires `nir_band="B8A"` and
  requires `B8A` scoring data to be available during materialization.

### Execution Notes

- Ranking runs during **materialize**, not **prepare**.
- It runs only for item groups with more than one item.
- This works for both `ingest: true` and `ingest: false`.
- The scoring bands are read in addition to output bands.
- For reliable ranking quality, create windows with at least `96x96` pixels.
- `min_inference_size` only pads small windows; it does not add context from
  outside the window, so very small windows can still have lower accuracy.

Requires the optional `omnicloudmask` package (`pip install .[extra]` in this repo).
