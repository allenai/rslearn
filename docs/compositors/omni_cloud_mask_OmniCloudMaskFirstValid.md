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

### Execution Notes

- Ranking runs during **materialize**, not **prepare**.
- It runs only for item groups with more than one item.
- This works for both `ingest: true` and `ingest: false`.
- The scoring bands are read in addition to output bands.

Requires the optional `omnicloudmask` package (`pip install .[extra]` in this repo).
