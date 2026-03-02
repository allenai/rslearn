## rslearn.data_sources.aws_google_satellite_embedding_v1.GoogleSatelliteEmbeddingV1

Google Satellite Embedding v1 (annual satellite embeddings from AlphaEarth) at 10 m/pixel
resolution on AWS Open Data. The data is stored as Cloud-Optimized GeoTIFFs organized by
year and UTM zone.

Available years: 2018-2024.

See https://registry.opendata.aws/aef-source/ for details. No credentials are needed (public bucket).

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.aws_google_satellite_embedding_v1.GoogleSatelliteEmbeddingV1",
  "init_args": {
    // Required cache directory for the index file. Unless prefixed by a protocol
    // (like "file://..."), it is joined with the dataset path.
    "metadata_cache_dir": "cache/gse",
    // Whether to apply de-quantization (default true). The raw data is int8; with
    // de-quantization enabled, values are mapped to float32 in [-1, 1] using:
    //   ((value / 127.5) ** 2) * sign(value)
    // Raw nodata is -128; after de-quantization it becomes -1.0.
    // Set false to keep raw int8 values.
    "apply_dequantization": true
  }
}
```

This data source supports direct materialization (`"ingest": false`).

### Available Bands

The band names for the 64 embedding channels are "A00" through "A63".

- When `apply_dequantization` is true (default): float32 values in [-1, 1] (nodata = -1).
- When `apply_dequantization` is false: raw int8 values (nodata = -128).
