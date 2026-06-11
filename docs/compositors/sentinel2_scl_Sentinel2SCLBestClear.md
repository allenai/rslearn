## rslearn.dataset.sentinel2_scl.Sentinel2SCLBestClear

Selects the single Sentinel-2 item in a group with the highest clear-cover fraction
using Sentinel-2 Scene Classification Layer (SCL) values over the requested window.

This compositor is Sentinel-2-specific and requires access to an SCL band.

### Configuration

```jsonc
{
  "compositing_method": {
    "class_path": "rslearn.dataset.sentinel2_scl.Sentinel2SCLBestClear",
    "init_args": {
      "scl_band": "SCL",
      "clear_values": [4, 5, 6],
      "min_clear_fraction": 0.8,
      "min_valid_cover": 0.5
    }
  }
}
```

`min_clear_fraction` and `min_valid_cover` default to `0.0`, which preserves the
default behavior of selecting the clearest available scene without filtering.

### Scoring

For each candidate item, the compositor reads `scl_band` over the requested window
and computes:

```text
clear_cover = count(SCL in clear_values) / total_window_pixels
clear_fraction = count(SCL in clear_values) / count(SCL != 0)
valid_cover = count(SCL != 0) / total_window_pixels
```

Sentinel-2 SCL class `0` is treated as nodata. Candidates below
`min_clear_fraction` or `min_valid_cover` are skipped. Remaining candidates are
sorted by highest `clear_cover`, then highest `valid_cover`, then raw clear pixel
count. Only the selected item is materialized.

Default clear SCL values are:

- `4`: vegetation
- `5`: bare soil / not vegetated
- `6`: water

You can include additional classes such as `11` snow/ice by overriding
`clear_values`. SCL class `0` cannot be included because it is the fixed nodata
class.

### Thresholds

Use `min_clear_fraction` to require a minimum clear fraction among valid pixels. For
example, `0.8` requires at least 80% of non-nodata SCL pixels to be clear.

Use `min_valid_cover` when you also want to reject scenes that only cover a small
part of the requested window. For example, `0.5` requires at least half of the
window to have non-nodata SCL pixels.

For a 100-pixel window:

| Clear Pixels | Cloud/Other Valid Pixels | Nodata Pixels | `clear_fraction` | `valid_cover` |
|---:|---:|---:|---:|---:|
| 80 | 20 | 0 | 80 / 100 = 0.80 | 100 / 100 = 1.00 |
| 8 | 2 | 90 | 8 / 10 = 0.80 | 10 / 100 = 0.10 |
| 40 | 40 | 20 | 40 / 80 = 0.50 | 80 / 100 = 0.80 |

With `min_clear_fraction: 0.8` and `min_valid_cover: 0.5`, the first scene passes,
the second is rejected because it covers too little of the window, and the third is
rejected because too few valid pixels are clear.

### Execution Notes

- Selection runs during **materialize**, not **prepare**.
- It runs only for item groups with more than one item.
- Use `query_config.space_mode: "SINGLE_COMPOSITE"` or another grouping mode that
  creates multi-item groups; `INTERSECTS` creates separate one-item groups, so there is
  nothing to rank.
- Unlike `Sentinel2SCLFirstValid`, this compositor does not fill nodata pixels from
  lower-ranked items. The output comes from one selected scene only.
- This works for both `ingest: true` and `ingest: false`.
- If `scl_band` is not available for an item, materialization raises an error.
