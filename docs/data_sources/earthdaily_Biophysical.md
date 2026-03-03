## rslearn.data_sources.earthdaily.Biophysical

Biophysical variables on [EarthDaily](https://earthdaily.com/) platform (EDAgro layers).

See [EarthDaily Setup](earthdaily.md) for required dependency/credentials.

By default, this data source applies per-asset scale/offset values from STAC
`raster:bands` metadata (`apply_scale_offset: true`) to convert raw pixel values into
physical units using `physical = raw * scale + offset`. Set `apply_scale_offset: false`
to keep raw values.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.earthdaily.Biophysical",
  "init_args": {
    // Required: which biophysical variable to fetch.
    // One of: "lai", "fapar", "fcover".
    "variable": "lai",
    // Whether to apply STAC `raster:bands` scale/offset (default true). Set to false to
    // keep raw values.
    "apply_scale_offset": true,
    // Optional: STAC API `query` filter passed to searches.
    "query": null,
    // Optional: STAC item property to sort by before grouping/matching (default null).
    "sort_by": null,
    // Whether to sort ascending when sort_by is set (default true).
    "sort_ascending": true,
    // Optional cache directory for cached item metadata.
    "cache_dir": null,
    // Timeout for HTTP asset downloads.
    "timeout": "10s",
    // Retry settings for EarthDaily API client requests (search/get item).
    "max_retries": 3,
    "retry_backoff_factor": 5.0
  }
}
```

### Available Bands

Band names correspond 1:1 with the selected `variable`:
- `variable: "lai"` → `lai`
- `variable: "fapar"` → `fapar`
- `variable: "fcover"` → `fcover`
