## rslearn.data_sources.earthdaily.Sentinel2EDACloudMask

Sentinel-2 EDA cloud-mask data on the [EarthDaily](https://earthdaily.com/)
platform using the `sentinel-2-eda-cloud-mask` collection.

See [EarthDaily Setup](earthdaily.md) for required dependency/credentials.

The source exposes the first band of the `cloud-mask` STAC asset as a single-band
categorical raster. Use `resampling_method: "nearest"` for this layer to preserve
class values.

For a workflow that ranks cloud-mask items by AOI clear cover and then retrieves the
related Sentinel-2 L2A item, see
[EarthDaily Cloud-Mask Clear-Cover Selection](../examples/EarthDailyCloudMaskClearCover.md).

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.earthdaily.Sentinel2EDACloudMask",
  "init_args": {
    // Optional: EarthDaily STAC asset keys to fetch (default null).
    // If null and the layer config is available, assets are inferred from the layer's
    // requested band names. The only supported asset is "cloud-mask".
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

Example layer snippet:

```jsonc
{
  "type": "raster",
  "band_sets": [{
    "bands": ["cloud-mask"],
    "dtype": "uint8",
    "nodata_value": 0,
    "zoom_offset": 0
  }],
  "resampling_method": "nearest",
  "data_source": {
    "class_path": "rslearn.data_sources.earthdaily.Sentinel2EDACloudMask"
  }
}
```

### Available Band

The `cloud-mask` band contains categorical cloud-mask values:

- `0`: nodata
- `1`: clear
- `2`: cloud
- `3`: cloud shadow
- `4`: thin cloud

The STAC items include `eda:derived_from_collection_id` and
`eda:derived_from_item_id` properties that identify the source Sentinel-2 item.
