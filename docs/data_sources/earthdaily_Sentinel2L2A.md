## rslearn.data_sources.earthdaily.Sentinel2L2A

Sentinel-2 L2A data on [EarthDaily](https://earthdaily.com/) platform using the
`sentinel-2-l2a` collection.

**This is the recommended source for users migrating from Planetary Computer.** It
exposes the same asset keys and band names as
`rslearn.data_sources.planetary_computer.Sentinel2` (`B01`–`B12` except `B10`, plus
`B8A`, `SCL`, and `visual`), so existing layer configs and band references can be reused
without changes.

For the newer EarthDaily Collection 1 (`sentinel-2-c1-l2a`) with different asset keys
and scale/offset-applied reflectance, use
[`rslearn.data_sources.earthdaily.Sentinel2C1L2A`](earthdaily_Sentinel2C1L2A.md).

Authentication and dependency requirements are the same as `Sentinel2C1L2A` (optional
`earthdaily[platform]`, `EDS_CLIENT_ID`, `EDS_SECRET`, `EDS_AUTH_URL`, `EDS_API_URL`).
See [EarthDaily Setup](earthdaily.md).

For collection lifecycle context (`sentinel-2-c1-l2a` replacing `sentinel-2-l2a`) and
known archive gaps, see [earthdaily.Sentinel2C1L2A](earthdaily_Sentinel2C1L2A.md#collection-status).

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.earthdaily.Sentinel2L2A",
  "init_args": {
    // Flag (default false) to harmonize pixel values across different processing
    // baselines.
    "harmonize": false,
    // Optional: list of Sentinel-2 asset keys to fetch.
    // Example: ["B02", "B03", "B04", "B08"]
    "assets": null,
    // Optional: maximum cloud cover (%) to filter items at search time. If set,
    // injects an `eo:cloud_cover` upper bound into the STAC query.
    "cloud_cover_max": null,
    // Maximum number of STAC items to fetch per window before rslearn grouping/matching.
    "search_max_items": 500,
    // Optional ordering of items before grouping (useful with SpaceMode.COMPOSITE +
    // CompositingMethod.FIRST_VALID): "cloud_cover" (default), "datetime", or null.
    "sort_items_by": "cloud_cover",
    // Optional: STAC API `query` filter passed to searches.
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

### Available Bands

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
- SCL
- R, G, B (from the `visual` asset)

### Harmonization

Harmonization first tries to build the callback from Sentinel-2 metadata XML
(`product_metadata`) using `get_harmonize_callback(...)` when that asset is present.

If metadata is missing or cannot be read while `harmonize=true`, rslearn raises
that error instead of silently skipping harmonization.

If metadata XML is available but does not itself indicate whether an offset is
present, rslearn falls back to the processing baseline encoded in STAC
`properties["sentinel:product_id"]` when available, then to the item ID, and
otherwise to the acquisition date:

- scenes with baseline 04.00+ include a +1000 DN offset for reflectance bands,
- harmonization undoes this by subtracting 1000 DN (with clipping at zero),
- fallback applies this adjustment when the scene ID encodes baseline 04.00+,
- if the scene ID does not expose a processing baseline, fallback applies the
  adjustment for acquisitions on/after 2022-01-25,
- `SCL` and `visual` (TCI RGB) are not harmonized.
