# Zarr Data Source Example

The snippet below demonstrates how to reference a spatio-temporal Zarr cube from a
raster layer. Install the optional dependencies before running the dataset workflow:

```bash
uv pip install -e ".[extra]"
```

Add a layer similar to the following in your dataset's `config.json`:

```jsonc
"sentinel2": {
  "type": "raster",
  "bands": [
    {
      "name": "B02",
      "dtype": "float32",
      "nodata": 0.0
    },
    {
      "name": "B03",
      "dtype": "float32",
      "nodata": 0.0
    },
    {
      "name": "B04",
      "dtype": "float32",
      "nodata": 0.0
    }
  ],
  "data_source": {
    "name": "rslearn.data_sources.zarr.ZarrDataSource",
    "store_uri": "s3://bucket/path/to/datacube.zarr",
    "data_variable": "reflectance",
    "crs": "EPSG:32633",
    "pixel_size": 10,
    "origin": [500000.0, 4200000.0],
    "axis_names": {"x": "x", "y": "y", "time": "time", "band": "band"},
    "bands": ["B02", "B03", "B04"],
    "dtype": "float32",
    "nodata": 0.0,
    "chunk_shape": {"y": 1024, "x": 1024},
    "storage_options": {"anon": true}
  },
  // Set to false to stream directly from the cube instead of ingesting.
  "ingest": true
}
```

When `ingest` is left at the default `true`, run `rslearn dataset ingest` to cache each
chunk into your tile store. If you flip `ingest` to `false`, `rslearn dataset
materialize` will read the necessary portions directly from the Zarr store instead.

