## rslearn.data_sources.aws_glo30.CopernicusGLO30

Copernicus GLO-30 DEM (30m) elevation data, served directly from the original public
AWS S3 bucket at `s3://copernicus-dem-30m` (no credentials required).

The data is split into 1x1-degree Cloud-Optimized GeoTIFF tiles covering global land
areas, so elevation is read on demand at materialize time and there is no ingest step.
Tile paths are constructed deterministically from latitude/longitude; the bucket's
`tileList.txt` index is downloaded and cached so that we only create items for tiles
that actually exist.

This data source provides the raw `elevation` band only. Slope and aspect are derived
on the fly with
[`rslearn.train.transforms.terrain.ElevationToSlopeAspect`](../Transforms.md), matching
how other rslearn data sources materialize raw bands and leave derived quantities to
transforms.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.aws_glo30.CopernicusGLO30",
  "init_args": {
    // Directory to cache the tileList.txt index.
    "metadata_cache_dir": "cache/glo30",
    // Band name to use when the layer config is unavailable (default: elevation).
    "band_name": "elevation"
  },
  // Recommended query configuration.
  "query_config": {
    "space_mode": "MOSAIC",
    "max_matches": 1
  },
  "ingest": false
}
```

### Available Bands

The data source should be configured with a single band set containing a single band,
the elevation band (named `elevation` by default), in meters. The data type should be
`float32`.

```jsonc
{
  "type": "raster",
  "band_sets": [{
    "bands": ["elevation"],
    "dtype": "float32"
  }],
  "data_source": {
    "class_path": "rslearn.data_sources.aws_glo30.CopernicusGLO30",
    "init_args": {"metadata_cache_dir": "cache/glo30"},
    "query_config": {
      "space_mode": "MOSAIC",
      "max_matches": 1
    },
    "ingest": false
  }
}
```

Items from this data source do not come with a time range (the DEM is static).

### Deriving Slope and Aspect

Use `ElevationToSlopeAspect` in the model's transform pipeline to turn the elevation
band into elevation/slope/aspect channels. Because the transform runs after
materialization, the image is already in the window's projection, so pass the window's
pixel size in meters:

```jsonc
{
  "class_path": "rslearn.train.transforms.terrain.ElevationToSlopeAspect",
  "init_args": {
    // The window resolution in meters.
    "pixel_size_m": 10,
    "input_selector": "elevation",
    // Any subset/order of elevation, slope, and aspect.
    "bands": ["elevation", "slope", "aspect"]
  }
}
```

The emitted bands are:

- `elevation` — raw DEM value in meters
- `slope` — terrain slope in degrees [0, 90)
- `aspect` — compass direction of steepest descent in degrees [0, 360), -1 for flat

### Notes

- Only land tiles are published. Because we filter items against `tileList.txt`,
  windows over ocean simply match fewer (or no) tiles instead of failing to read.
- This data source supports direct materialization only. Setting `"ingest": true`
  raises an error.
- Slope and aspect use central differences in the interior and one-sided differences
  on the image border, so the outermost pixel ring of each window is less accurate.
  Pixels with NaN elevation produce NaN slope and aspect.
- See also `rslearn.data_sources.planetary_computer.CopDemGlo30` for the same dataset
  served via Microsoft Planetary Computer's STAC API.
