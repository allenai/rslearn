## rslearn.data_sources.gcp_public_data.Sentinel2

This data source is for Sentinel-2 L1C data on Google Cloud Storage.

Sentinel-2 imagery is available on Google Cloud Storage as part of the Google
Public Cloud Data Program. The images are added with a 1-2 day latency after
becoming available on Copernicus.

See https://cloud.google.com/storage/docs/public-datasets/sentinel-2 for details.

The bucket is public and free so no credentials are needed.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.gcp_public_data.Sentinel2",
  "init_args": {
    // Required cache directory to cache product metadata files and the optional rtree
    // index.
    "index_cache_dir": "cache/sentinel2",
    // Sort by this attribute, either null (default, meaning arbitrary ordering) or
    // "cloud_cover".
    "sort_by": null,
    // Flag (default true) to build an rtree index to speed up product lookups. This can
    // be set false to avoid lengthy (multiple hours) rtree creation time if you are only
    // using a few windows.
    "use_rtree_index": true,
    // Flag (default false) to harmonize pixel values across different processing
    // baselines (recommended), see
    // https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
    "harmonize": false,
    // When using rtree index, only create it for products within this time range. It
    // defaults to null, meaning to create rtree index for entire time range.
    // Example: ["2024-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00"]
    "rtree_time_range": null,
    // By default, if use_rtree_index is true, the rtree index is stored in the
    // index_cache_dir. Set this to override the path for the rtree index and only use
    // index_cache_dir for the product metadata files.
    "rtree_cache_dir": null
  }
}
```

### Available Bands

uint16 bands:

- B01
- B02
- B03
- B04
- B05
- B06
- B07
- B08
- B09
- B10
- B11
- B12
- B8A

uint8 bands:

- R (from TCI asset; derived from B04)
- G (from TCI asset; derived from B03)
- B (from TCI asset; derived from B02)

### Example

See [planetary_computer_Sentinel2](./planetary_computer_Sentinel2.md) for example usage
of a related data source.
