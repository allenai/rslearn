## rslearn.data_sources.earthdaily.Sentinel2

Sentinel-2 L2A data on [EarthDaily](https://earthdaily.com/) platform (collection: `sentinel-2-c1-l2a`).
For EarthDaily collection `sentinel-2-l2a` with Planetary Computer-style asset keys and
optional harmonization, use `rslearn.data_sources.earthdaily.Sentinel2L2A`.

See [EarthDaily Setup](earthdaily.md) for required dependency/credentials.

By default, this data source applies per-asset scale/offset values from STAC
`raster:bands` metadata (`apply_scale_offset: true`) to convert raw pixel values into
physical units using `physical = raw * scale + offset`. Set `apply_scale_offset: false`
to keep raw values.
For Sentinel-2 spectral bands, this physical unit is reflectance (typically BOA
reflectance for L2A products), e.g. raw `10000` with scale `0.0001` maps to `1.0`.

When `apply_scale_offset: true`, configure the target `band_sets[].dtype` as `float32`.
rslearn will raise during initialization if a non-float dtype is configured through the
layer context. Nodata is read from STAC `raster:bands` metadata and preserved during
scale/offset application (for this collection, nodata is typically `0`).

Note: EarthDaily may include a preview `thumbnail` asset; rslearn does not ingest/materialize it.

### Collection Status

According to the Earth Search project, `sentinel-2-c1-l2a` is intended to eventually
replace `sentinel-2-l2a`, and contains COG assets processed to at least baseline 5.0.
As noted there (as of April 2024), ESA archive reprocessing to baseline 5.0 is still
incomplete, with known gaps for Nov 2016 to Nov 2019 and for 2022, and no published
completion date.

Source: <https://github.com/Element84/earth-search>

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.earthdaily.Sentinel2",
  "init_args": {
    // Whether to apply STAC `raster:bands` scale/offset (default true). Set to false to
    // keep raw values.
    "apply_scale_offset": true,
    // Optional: EarthDaily Sentinel-2 STAC asset keys to fetch (default null).
    // If null and the layer config is available, assets are inferred from the layer's
    // requested band names.
    //
    // Note: this is different from the "Available bands" list below:
    // - "assets" uses EarthDaily STAC asset keys (e.g. "red", "nir", "visual", "scl").
    // - "band_sets[].bands" uses rslearn band names (e.g. "B04", "B08", "R", "scl").
    //
    // Example: ["red", "green", "blue", "nir", "swir16", "swir22", "visual", "scl"]
    "assets": null,
    // Optional: maximum cloud cover (%) to filter items at search time.
    // If set, it takes precedence over cloud_cover_threshold and overrides any
    // `eo:cloud_cover` filter in `query`.
    "cloud_cover_max": null,
    // Optional: default max cloud cover (%) to apply when cloud_cover_max is not set.
    // If set, it overrides any `eo:cloud_cover` filter in `query`.
    // If both cloud_cover_max and cloud_cover_threshold are null, `query` (including
    // any `eo:cloud_cover` filter) is passed through unchanged.
    "cloud_cover_threshold": null,
    // Maximum number of STAC items to fetch per window before rslearn grouping/matching.
    "search_max_items": 500,
    // Optional ordering of items before grouping (useful with SpaceMode.COMPOSITE +
    // CompositingMethod.FIRST_VALID): "cloud_cover" (default), "datetime", or null.
    "sort_items_by": "cloud_cover",
    // Optional: STAC API `query` filter passed to searches.
    // Example: {"s2:product_type": {"eq": "S2MSI2A"}}
    // Note: if cloud_cover_max/cloud_cover_threshold is set, the effective query also
    // includes an `eo:cloud_cover` upper bound.
    "query": null,
    // Optional: STAC item property to sort by before grouping/matching (default null).
    // If set, it takes precedence over sort_items_by.
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

Example layer snippet:

```jsonc
{
  "type": "raster",
  "band_sets": [{
    "bands": ["B02", "B03", "B04"],
    "dtype": "float32",
    // Optional: override nodata explicitly if needed.
    // "nodata_vals": [0, 0, 0]
  }],
  "data_source": {
    "class_path": "rslearn.data_sources.earthdaily.Sentinel2",
    "init_args": {
      "apply_scale_offset": true
    }
  }
}
```

### Available Bands

Available rslearn band names (select via `band_sets[].bands`; rslearn infers required
EarthDaily assets when `assets` is null):
- B01
- B02
- B03
- B04
- B05
- B06
- B07
- B08
- B09
- B11
- B12
- B8A
- R, G, B (from the `visual` asset)
- scl, aot, wvp

Common EarthDaily asset key to rslearn band name mapping:
- coastal → B01
- blue → B02
- green → B03
- red → B04
- rededge1 → B05
- rededge2 → B06
- rededge3 → B07
- nir → B08
- nir08 → B8A
- nir09 → B09
- swir16 → B11
- swir22 → B12
- visual → R, G, B
- scl → scl
- aot → aot
- wvp → wvp
