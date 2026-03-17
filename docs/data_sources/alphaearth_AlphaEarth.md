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
  "type": "raster",
  "band_sets": [{
    "dtype": "float32",
    // Generate A00..A63 without listing all 64 bands explicitly.
    "num_bands": 64,
    "band_prefix": "A",
    "band_zero_pad": 2
  }],
  "data_source": {
    "class_path": "rslearn.data_sources.alphaearth.AlphaEarth",
    // Recommended: use direct materialization instead of ingesting annual TIFFs.
    "ingest": false,
    "init_args": {
      // Required cache directory for the GeoParquet index. Unless prefixed by a protocol
      // (like "file://..."), it is joined with the dataset path.
      "metadata_cache_dir": "cache/alphaearth",
      // Optional alternate GeoParquet index URL or local file path.
      "index_url": "https://data.source.coop/tge-labs/aef/v1/annual/aef_index_stac_geoparquet.parquet",
      // Whether to apply de-quantization (default true). The raw data is int8; with
      // de-quantization enabled, values are approximately mapped back into the
      // original embedding space as float32 in [-1, 1] using:
      //   ((value / 127.5) ** 2) * sign(value)
      // Raw nodata is -128; after de-quantization it becomes -2.0 so nodata stays
      // outside the valid embedding range.
      // Set false only if you explicitly want the raw quantized int8 values.
      "apply_dequantization": true
    }
  }
}
```

This data source supports direct materialization (`"ingest": false`), and that is the
recommended mode. Setting `"ingest": true` is not recommended because the annual
AlphaEarth TIFFs are large, around 3 GB each.

### Available Bands

The band names for the 64 embedding channels are `"A00"` through `"A63"`.

- When `apply_dequantization` is true (default): float32 values in [-1, 1] (nodata = -2).
- When `apply_dequantization` is false: raw int8 values (nodata = -128).

If `nodata_vals` is omitted in the layer band set, AlphaEarth supplies a default:

- `-2` when `apply_dequantization` is true
- `-128` when `apply_dequantization` is false

You usually do not need to set `nodata_vals` for AlphaEarth explicitly. Leave it
unset unless you intentionally want to override these datasource defaults.

`-2` is used for the dequantized path because it lies outside the valid embedding
range and avoids colliding with legitimate values near or at `-1` that may arise in
downstream normalization or post-processing workflows.

### Embedding Semantics

AlphaEarth embeddings are designed to live in a semantically meaningful embedding
space that is consistent across years. The original embeddings are unit-length and can
be compared with cosine similarity, dot product, or angular distance for tasks such as
clustering and temporal change detection.

In rslearn, `apply_dequantization=true` approximately decodes the stored int8 values
back into that embedding space. Because this reconstruction happens after
quantization, exact unit norm is not preserved mathematically. For most workflows this
approximation is appropriate, but if your downstream method assumes exact unit-length
vectors, L2-normalize the embeddings before analysis.

The embeddings are also intended to be linearly composable: averaging or aggregating
them can produce useful coarser-resolution representations. When doing this with
dequantized embeddings, optional re-normalization afterward is recommended for
cosine-based comparisons.

### Model Input Config

In model config, you can avoid listing all 64 embedding bands by using
`use_all_bands_in_order_of_band_set_idx`:

```yaml
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    path: ${DATASET_PATH}
    inputs:
      embeddings:
        data_type: raster
        layers: ["alphaearth"]
        use_all_bands_in_order_of_band_set_idx: 0
        dtype: FLOAT32
        passthrough: true
```
