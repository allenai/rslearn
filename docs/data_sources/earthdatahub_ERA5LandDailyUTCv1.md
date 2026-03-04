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
    // Units to return for `t2m` (default "kelvin"): "celsius" or "kelvin".
    "temperature_unit": "kelvin",
    // Whether to allow the underlying HTTP client to read environment configuration
    // (including netrc) for auth/proxies (default true).
    "trust_env": true
  }
}
```

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
- `t2m`: 2m temperature (units: K or °C; see `temperature_unit`)
- `tp`: total precipitation (units: m)
- `u10`: 10m U wind component (units: m s-1)
- `v10`: 10m V wind component (units: m s-1)
