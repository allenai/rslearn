## rslearn.data_sources.planetary_computer.Sentinel3SlstrLST

[Sentinel-3 SLSTR Level-2 Land Surface Temperature (LST) data on Microsoft Planetary
Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-3-slstr-lst-l2-netcdf).
This dataset is provided as netCDF swaths; the data source uses the `lst-in`
asset for measurements and the `slstr-geodetic-in` asset for geolocation. During
ingestion it interpolates the swath onto a regular lat/lon grid using linear
weights (this is an approximation; for precise geolocation you may need a custom
workflow).

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.planetary_computer.Sentinel3SlstrLST",
  "init_args": {
    // Stride for sampling geolocation arrays when estimating grid resolution.
    "sample_step": 20,
    // Nodata value used for missing pixels (default 0.0).
    "nodata_value": 0.0,
    // Optional output grid resolution in degrees. If omitted, estimate from geodetic arrays.
    "grid_resolution": null,
    // See rslearn.data_sources.planetary_computer.PlanetaryComputer.
    "query": null,
    "sort_by": null,
    "sort_ascending": true,
    "timeout_seconds": 10
  }
}
```

### Available Bands

- LST
