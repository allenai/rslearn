## rslearn.data_sources.earthdatahub.ERA5LandDailyUTCv1

ERA5-Land daily UTC (v1) hosted on EarthDataHub.

See https://earthdatahub.destine.eu/collections/era5/datasets/era5-land-daily for details.

Authentication requires configuring the netrc file (`~/.netrc` on Linux and MacOS) as follows:

```
machine data.earthdatahub.destine.eu
  password <write your personal access token here>
```

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.earthdatahub.ERA5LandDailyUTCv1",
  "init_args": {
    // Optional: URL/path to the EarthDataHub Zarr store.
    "zarr_url": "https://data.earthdatahub.destine.eu/era5/era5-land-daily-utc-v1.zarr",
    // Optional bounding box as [min_lon, min_lat, max_lon, max_lat] (WGS84).
    // Recommended for performance. Dateline-crossing bounds (min_lon > max_lon) are not
    // supported.
    "bounds": null,
    // Whether to allow the underlying HTTP client to read environment configuration
    // (including netrc) for auth/proxies (default true).
    "trust_env": true
  }
}
```

### Recommended materialized time-series layer

For small spatial windows with many daily timesteps, use the data source with
`SINGLE_COMPOSITE`, `SPATIAL_MOSAIC_TEMPORAL_STACK`, and `NumpyRasterFormat`.
This materializes one `(C, T, H, W)` NumPy array per window/layer/band set instead
of one GeoTIFF per timestep.

```jsonc
{
  "layers": {
    "era5": {
      "type": "raster",
      "compositing_method": "SPATIAL_MOSAIC_TEMPORAL_STACK",
      "band_sets": [
        {
          "dtype": "float32",
          "bands": ["t2m", "tp"],
          "nodata_value": -9999.0,
          // Optional, but useful for point-like windows where ERA5 should be
          // loaded as one pixel per timestep.
          "spatial_size": [1, 1],
          "format": {
            "class_path": "rslearn.utils.raster_format.NumpyRasterFormat"
          }
        }
      ],
      "data_source": {
        "class_path": "rslearn.data_sources.earthdatahub.ERA5LandDailyUTCv1",
        "init_args": {
          "band_names": ["t2m", "tp"],
          "trust_env": true
        },
        "query_config": {
          "space_mode": "SINGLE_COMPOSITE"
        }
      }
    }
  }
}
```

To materialize temporal aggregates instead, keep the same `NumpyRasterFormat` and
`SINGLE_COMPOSITE` setup but replace the compositing method with one of:

- `TEMPORAL_MEAN`
- `TEMPORAL_MAX`
- `TEMPORAL_MIN`

Each reducer first builds the same clipped spatial mosaic temporal stack, then
reduces across the T dimension to one timestep. For example, if dataset windows are
bi-weekly, changing `SPATIAL_MOSAIC_TEMPORAL_STACK` to `TEMPORAL_MEAN` writes one
mean aggregate per bi-weekly window.

### Available Bands

- `d2m`: 2m dewpoint temperature (units: K)
- `e`: evaporation (units: m of water equivalent)
- `pev`: potential evaporation (units: m)
- `ro`: runoff (units: m)
- `sp`: surface pressure (units: Pa)
- `ssr`: surface net short-wave (solar) radiation (units: J m-2)
- `ssrd`: surface short-wave (solar) radiation downwards (units: J m-2)
- `str`: surface net long-wave (thermal) radiation (units: J m-2)
- `swvl1`: volumetric soil water layer 1 (units: m3 m-3)
- `swvl2`: volumetric soil water layer 2 (units: m3 m-3)
- `t2m`: 2m temperature (units: K)
- `tp`: total precipitation (units: m)
- `u10`: 10m U wind component (units: m s-1)
- `v10`: 10m V wind component (units: m s-1)
