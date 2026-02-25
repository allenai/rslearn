## rslearn.data_sources.aws_open_data.Naip

This data source is for NAIP imagery on AWS. It uses the naip-source requester pays
bucket maintained by Esri. See https://registry.opendata.aws/naip/ for more
information. AWS credentials must be configured for use with boto3.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.aws_open_data.Naip",
  "init_args": {
    // Required cache directory to cache index shapefiles. Unless prefixed by a protocol
    // (like "file://..."), it is joined with the dataset path.
    "index_cache_dir": "cache/naip",
    // Whether to build an rtree index to accelerate prepare lookups, default false. It
    // is recommended to set this true when processing more than a few windows.
    "use_rtree_index": false,
    // Limit the search to these states (list of their two-letter codes). This can
    // substantially accelerate lookups when the rtree index is disabled, since by
    // default (null) it has to scan through all of the states.
    // Example: ["wa", "or"]
    "states": null,
    // Limit the search to these years. Like with states, this can speed up lookups when
    // the rtree index is disabled.
    // Example: [2023, 2024]
    "years": null
  }
}
```

### Available Bands

Pixel values are uint8.

- R
- G
- B
- IR
