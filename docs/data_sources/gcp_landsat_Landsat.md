## rslearn.data_sources.gcp_landsat.Landsat

This data source is for Landsat imagery on GCP's public Landsat GCS bucket
(`gs://gee-public-data-landsat`). It contains Collection 2 data (L1 and L2) for all
Landsat missions (1-9) as cloud-optimized GeoTIFFs. The bucket is requester-pays, so a
GCP project must be configured for billing.

Scene discovery uses BigQuery table
`earth-engine-public-data.geo_index.landsat_c2_index`.

Two discovery modes are supported:

- `use_rtree_index=true` (default): one BigQuery scan builds an on-disk rtree index
  for fast repeated lookups.
- `use_rtree_index=false`: each `get_items` call runs one BigQuery query filtered by
  time, geometry bounding box, and WRS path/row.

Scene geometry uses the bounding box from the BigQuery index. Use
`rtree_time_range` to restrict rtree construction to a date range, which
significantly speeds up index creation.

This data source supports direct materialization: if the "ingest" flag is set false,
then ingestion will be skipped and windows will be directly populated from windowed
reads of the underlying COG files via `gs://` URLs.

### Environment Variables

The following environment variables must be set:

- `GOOGLE_APPLICATION_CREDENTIALS`: path to a GCP service account JSON key file.
- `GS_USER_PROJECT`: GCP project for requester-pays billing (e.g. `my-gcp-project`).
  This is used both for BigQuery/GCS calls made directly by the data source, and by
  GDAL for `gs://` URLs.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.gcp_landsat.Landsat",
  "init_args": {
    // Required cache directory for index and rtree files. Unless prefixed by a
    // protocol (like "file://..."), it is joined with the dataset path.
    "index_cache_dir": "cache/gcp_landsat",
    // Required sensor filter. Available bands differ across sensors. Multiple sensors
    // are supported only for Level-1 processing levels; requested bands must be
    // available for every configured sensor.
    // Values: "OLI_TIRS" (Landsat 8/9), "OLI" (Landsat 8/9 OLI-only),
    // "TIRS" (Landsat 8/9 TIRS-only), "ETM" (Landsat 7), "TM" (Landsat 4/5),
    // "MSS" (Landsat 1-5).
    "sensor_ids": ["OLI_TIRS"],
    // Required processing-level filter. Available bands differ across processing
    // levels. Multiple processing levels are supported only when every configured
    // value is Level-1 ("L1GS", "L1GT", or "L1TP").
    // Values: "L1GS", "L1GT", "L1TP", "L2SP", "L2SR".
    "processing_levels": ["L1TP"],
    // Optional spacecraft filter. null means all missions with the configured sensors.
    // Values: "LANDSAT_1" through "LANDSAT_9".
    "spacecraft_ids": null,
    // Filter by collection tier. null means all.
    // Values: "T1", "T2", "RT".
    "collection_category": ["T1"],
    // Sort by this attribute. null (default) for arbitrary ordering, or "cloud_cover".
    "sort_by": "cloud_cover",
    // Whether to use local rtree mode or one-query-per-get_items BigQuery mode.
    "use_rtree_index": true,
    // Only index scenes within this time range. Highly recommended to speed up
    // rtree creation. null means all scenes.
    "rtree_time_range": ["2025-01-01T00:00:00", "2025-12-31T23:59:59"]
  }
}
```

### Available Bands

Bands depend on both the sensor and the processing level.
Pixel values are uint16.
When multiple sensors or multiple Level-1 processing levels are configured, only bands
available for every configured sensor and processing level are valid; if `bands` is
omitted, the data source defaults to those common bands.

**Level-1 (`L1GS`, `L1GT`, `L1TP`)**

These three Level-1 product types have the same band naming for a given sensor; they
mainly differ in geometric correction quality. `L1TP` is terrain precision corrected
using ground control points and DEM data, and is generally the best Level-1 choice for
pixel-level time series. `L1GT` is systematic terrain corrected using DEM data but
without the same ground-control precision correction. `L1GS` is systematic corrected
using spacecraft/sensor information only, without terrain correction.

- **OLI-TIRS (Landsat 8/9):** B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11
- **OLI (Landsat 8/9 OLI-only):** B1, B2, B3, B4, B5, B6, B7, B8, B9
- **TIRS (Landsat 8/9 TIRS-only):** B10, B11
- **ETM+ (Landsat 7):** B1, B2, B3, B4, B5, B6_VCID_1, B6_VCID_2, B7, B8
- **TM (Landsat 4/5):** B1, B2, B3, B4, B5, B6, B7
- **MSS (Landsat 1-5):** B4, B5, B6, B7

**Level-2 Surface Reflectance + Surface Temperature (`L2SP`)**

- **OLI-TIRS (Landsat 8/9):** B1, B2, B3, B4, B5, B6, B7, B10
- **ETM+ (Landsat 7):** B1, B2, B3, B4, B5, B6, B7
- **TM (Landsat 4/5):** B1, B2, B3, B4, B5, B6, B7

For `L2SP`, non-thermal bands are surface reflectance (`SR_B*` on GCS). The thermal
band is surface temperature (`ST_B10` for OLI-TIRS, `ST_B6` for ETM+ and TM).

**Level-2 Surface Reflectance Only (`L2SR`)**

- **OLI-TIRS (Landsat 8/9):** B1, B2, B3, B4, B5, B6, B7
- **ETM+ (Landsat 7):** B1, B2, B3, B4, B5, B7
- **TM (Landsat 4/5):** B1, B2, B3, B4, B5, B7

`L2SR` products do not include surface temperature assets, so OLI-TIRS B10 and
ETM+/TM B6 are not available for `L2SR`.
