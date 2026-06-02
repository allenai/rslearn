## rslearn.data_sources.copernicus_glo30.CopernicusGLO30

Copernicus GLO-30 DEM (30m) elevation data, served directly from the original public
AWS S3 bucket at `s3://copernicus-dem-30m` (no credentials required).

The data is split into 1x1-degree COG tiles covering global land areas. No file list
is needed — tile paths are constructed deterministically from latitude/longitude.

In addition to raw elevation, this data source can compute **slope** and **aspect**
as derived bands during ingest. Slope and aspect are computed in the geographic
(EPSG:4326) coordinate system with per-row latitude correction for proper metric
gradients.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.copernicus_glo30.CopernicusGLO30",
  "init_args": {
    // Timeout for HTTP requests (default: 30s).
    "timeout": "30s"
  },
  // Recommended query configuration.
  "query_config": {
    "space_mode": "MOSAIC",
    "max_matches": 1
  },
  "ingest": true
}
```

### Available Bands

The data source should be configured with a single band set containing one or more of:

- `elevation` — raw DEM value in meters
- `slope` — terrain slope in degrees [0, 90)
- `aspect` — compass direction of steepest descent in degrees [0, 360), -1 for flat

The data type should be `float32`.

Example with all three bands:

```jsonc
{
  "type": "raster",
  "band_sets": [{
    "bands": ["elevation", "slope", "aspect"],
    "dtype": "float32"
  }],
  "data_source": {
    "class_path": "rslearn.data_sources.copernicus_glo30.CopernicusGLO30",
    "query_config": {
      "space_mode": "MOSAIC",
      "max_matches": 1
    },
    "ingest": true
  }
}
```

Example with elevation only:

```jsonc
{
  "type": "raster",
  "band_sets": [{
    "bands": ["elevation"],
    "dtype": "float32"
  }],
  "data_source": {
    "class_path": "rslearn.data_sources.copernicus_glo30.CopernicusGLO30",
    "query_config": {
      "space_mode": "MOSAIC",
      "max_matches": 1
    },
    "ingest": true
  }
}
```

Items from this data source do not come with a time range (the DEM is static).

### Notes

- Tiles over open ocean do not exist in the source dataset. These are skipped
  gracefully during ingest (a warning is logged).
- Slope and aspect are computed from the raw geographic grid using `numpy.gradient`
  with per-row latitude correction. At tile boundaries, gradients use one-sided
  differences (a 1-pixel-wide edge artifact).
- See also `rslearn.data_sources.planetary_computer.CopDemGlo30` for the same dataset
  served via Microsoft Planetary Computer's STAC API (supports direct materialization
  but only provides the raw elevation band).
