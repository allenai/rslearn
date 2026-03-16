## rslearn.data_sources.alphaearth.AlphaEarth

AlphaEarth annual satellite embeddings at 10 m/pixel resolution hosted on Source
Cooperative. This data source reads the public STAC GeoParquet index and accesses the
64-band Cloud-Optimized GeoTIFF assets over HTTPS.

Available years: 2018-2024.

This data source requires the optional `duckdb` dependency, which is included in
`rslearn[extra]`.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.alphaearth.AlphaEarth",
  "init_args": {
    // Required cache directory for the GeoParquet index. Unless prefixed by a protocol
    // (like "file://..."), it is joined with the dataset path.
    "metadata_cache_dir": "cache/alphaearth",
    // Optional alternate GeoParquet index URL or local file path.
    "index_url": "https://data.source.coop/tge-labs/aef/v1/annual/aef_index_stac_geoparquet.parquet",
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

The band names for the 64 embedding channels are `"A00"` through `"A63"`.

- When `apply_dequantization` is true (default): float32 values in [-1, 1] (nodata = -1).
- When `apply_dequantization` is false: raw int8 values (nodata = -128).
