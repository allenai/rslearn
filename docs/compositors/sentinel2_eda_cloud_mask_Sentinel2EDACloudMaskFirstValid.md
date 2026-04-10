## rslearn.dataset.sentinel2_eda_cloud_mask.Sentinel2EDACloudMaskFirstValid

Ranks each item in a group using Sentinel-2 EDA cloud-mask values over the requested
window, then applies FIRST_VALID in ranked order.

This compositor is Sentinel-2/EarthDaily specific and requires an EDA cloud-mask band
(default: `eda_cloud_mask`).

### Configuration

```jsonc
{
  "compositing_method": {
    "class_path": "rslearn.dataset.sentinel2_eda_cloud_mask.Sentinel2EDACloudMaskFirstValid",
    "init_args": {
      "cloud_mask_band": "eda_cloud_mask",
      "null_weight": 5,
      "clear_weight": 0,
      "cloud_weight": 5,
      "shadow_weight": 1,
      "thin_cloud_weight": 1,
      "unknown_weight": 5
    }
  }
}
```

### Scoring

The score is a weighted sum of class fractions:

- `0`: null / nodata
- `1`: clear
- `2`: cloud
- `3`: cloud shadow
- `4`: thin cloud
- values outside `0..4`: unknown

Lower score is better.

With defaults, score is:
`5*null + 5*cloud + shadow + thin_cloud + 5*unknown` (`clear_weight=0`).

### Execution Notes

- Ranking runs during **materialize**, not **prepare**.
- It runs only for item groups with more than one item.
- This works for both `ingest: true` and `ingest: false`.
- If `cloud_mask_band` is missing for an item, materialization raises an error.
