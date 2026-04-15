## rslearn.dataset.sentinel2_scl.Sentinel2SCLFirstValid

Ranks each item in a group using Sentinel-2 Scene Classification Layer (SCL) values
over the requested window, then applies FIRST_VALID in ranked order.

This compositor is Sentinel-2-specific and requires access to an SCL band.

### Configuration

```jsonc
{
  "compositing_method": {
    "class_path": "rslearn.dataset.sentinel2_scl.Sentinel2SCLFirstValid",
    "init_args": {
      "scl_band": "SCL",
      "cloud_shadow_weight": 1,
      "medium_cloud_weight": 1,
      "high_cloud_weight": 5,
      "cirrus_weight": 1,
      "snow_ice_weight": 1
    }
  }
}
```

### Scoring

The score is a weighted sum of SCL class fractions:

- `3`: cloud shadow
- `8`: medium probability cloud
- `9`: high probability cloud
- `10`: thin cirrus
- `11`: snow/ice

Lower score is better.

With defaults, score is:
`shadow + medium_cloud + 5*high_cloud + cirrus + snow_ice`.

### Execution Notes

- Ranking runs during **materialize**, not **prepare**.
- It runs only for item groups with more than one item.
- This works for both `ingest: true` and `ingest: false`.
- If `scl_band` is not available for an item, materialization raises an error.
