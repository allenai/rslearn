## rslearn.data_sources.aws_landsat.LandsatOliTirs

This data source is for Landsat 8/9 OLI-TIRS imagery on AWS. It uses the usgs-landsat
S3 bucket maintained by USGS. It includes Tier 1/2 scenes but not Real-Time scenes. See
https://aws.amazon.com/marketplace/pp/prodview-ivr4jeq6flk7u for details about the
bucket.

This data source supports direct materialization: if the "ingest" flag is set false,
then ingestion will be skipped and windows will be directly populated from windowed
reads of the underlying cloud-optimized GeoTIFFs on S3.

Progress while scanning metadata is logged at DEBUG level; set `RSLEARN_LOGLEVEL=DEBUG`
to enable debug logging.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.aws_landsat.LandsatOliTirs",
  "init_args": {
    // Required cache directory to cache product metadata files. Unless prefixed by a
    // protocol (like "file://..."), it is joined with the dataset path (i.e., specifies
    // a sub-directory within the dataset folder.
    "metadata_cache_dir": "cache/landsat",
    // Sort by this attribute, either null (default, meaning arbitrary ordering) or
    // "cloud_cover".
    "sort_by": null
  }
}
```

### Available Bands

Pixel values are uint16.

- B1
- B2
- B3
- B4
- B5
- B6
- B7
- B8
- B9
- B10
- B11
