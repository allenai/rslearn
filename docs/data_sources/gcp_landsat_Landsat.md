## rslearn.data_sources.gcp_landsat.Landsat

This data source is for Landsat imagery on GCP's public Landsat GCS bucket
(`gs://gee-public-data-landsat`). It contains Collection 2 data (L1 and L2) for all
Landsat missions (1-9) as cloud-optimized GeoTIFFs. The bucket is requester-pays, so a
GCP project must be configured for billing.

Scene discovery uses the bucket's `index.csv.gz` (~23M rows), which is parsed once and
stored in an on-disk rtree index for fast spatial lookups. Scene geometry uses the
bounding box from the index. Use `rtree_time_range` to restrict the index to a date
range, which significantly speeds up rtree creation.

This data source supports direct materialization: if the "ingest" flag is set false,
then ingestion will be skipped and windows will be directly populated from windowed
reads of the underlying COG files via `gs://` URLs.

### Environment Variables

The following environment variables must be set:

- `GOOGLE_APPLICATION_CREDENTIALS`: path to a GCP service account JSON key file.
- `GS_USER_PROJECT`: GCP project for requester-pays billing (e.g. `my-gcp-project`).

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.gcp_landsat.Landsat",
  "init_args": {
    // Required cache directory for index and rtree files. Unless prefixed by a
    // protocol (like "file://..."), it is joined with the dataset path.
    "index_cache_dir": "cache/gcp_landsat",
    // Filter by spacecraft. null means all missions (Landsat 1-9).
    // Values: "LANDSAT_1" through "LANDSAT_9".
    "spacecraft_id": ["LANDSAT_8", "LANDSAT_9"],
    // Filter by collection tier. null means all.
    // Values: "T1", "T2", "RT".
    "collection_category": ["T1"],
    // Filter by processing level. null means all.
    // Values: "L1GS", "L1GT", "L1TP", "L2SP", "L2SR".
    "data_type": ["L1TP"],
    // Which bands to expose. null defaults to layer config bands or OLI-TIRS bands.
    "bands": null,
    // Sort by this attribute. null (default) for arbitrary ordering, or "cloud_cover".
    "sort_by": "cloud_cover",
    // GCP project for requester-pays billing (used for downloading index.csv.gz).
    "gcp_project": "my-gcp-project",
    // Only index scenes within this time range. Highly recommended to speed up
    // rtree creation. null means all scenes.
    "rtree_time_range": ["2025-01-01T00:00:00", "2025-12-31T23:59:59"]
  }
}
```

### Available Bands

Bands depend on the sensor. Pixel values are uint16.

**OLI-TIRS (Landsat 8/9):** B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11

**ETM+ (Landsat 7):** B1, B2, B3, B4, B5, B6_VCID_1, B6_VCID_2, B7, B8

**TM (Landsat 4/5):** B1, B2, B3, B4, B5, B6, B7

**MSS (Landsat 1-5):** B4, B5, B6, B7
