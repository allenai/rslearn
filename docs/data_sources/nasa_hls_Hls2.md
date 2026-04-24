## rslearn.data_sources.nasa_hls.Hls2S30

NASA LP DAAC HLS v2.0 Sentinel-2 ([HLSS30](https://hls.gsfc.nasa.gov/)) data on
CMR STAC / LP DAAC cloud storage. Direct materialization is supported.

This data source prefers LP DAAC `s3_*` assets and automatically fetches temporary
AWS credentials from the LP DAAC `s3credentials` endpoint. Set either:

- `EARTHDATA_TOKEN` (preferred), or
- `EARTHDATA_USERNAME` + `EARTHDATA_PASSWORD`

The aliases `NASA_EARTHDATA_TOKEN`, `NASA_EARTHDATA_USERNAME`, and
`NASA_EARTHDATA_PASSWORD` are also supported.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.nasa_hls.Hls2S30",
  "init_args": {
    // Optional list of bands to expose.
    "band_names": null,
    // Optional STAC query filter.
    "query": null,
    // Optional STAC sort property, e.g. "eo:cloud_cover".
    "sort_by": null,
    "sort_ascending": true,
    // Optional Earthdata auth overrides (otherwise env vars are used).
    "earthdata_token": null,
    "earthdata_username": null,
    "earthdata_password": null,
    "timeout_seconds": 30
  }
}
```

### Available Bands

The default band set includes:
- B01 (coastal)
- B02 (blue)
- B03 (green)
- B04 (red)
- B05 (rededge1)
- B06 (rededge2)
- B07 (rededge3)
- B08 (nir)
- B8A (nir_narrow)
- B11 (swir16)
- B12 (swir22)

Additional supported raster assets:
- B09 (water_vapor)
- B10 (cirrus)
- Fmask (fmask / qa)
- SAA (solar_azimuth)
- SZA (solar_zenith)
- VAA (view_azimuth)
- VZA (view_zenith)

Band names may be provided as either asset keys or the aliases above.

## rslearn.data_sources.nasa_hls.Hls2L30

NASA LP DAAC HLS v2.0 Landsat ([HLSL30](https://hls.gsfc.nasa.gov/)) data on
CMR STAC / LP DAAC cloud storage. Direct materialization is supported.

Authentication and temporary S3 credential handling are the same as for
`rslearn.data_sources.nasa_hls.Hls2S30`.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.nasa_hls.Hls2L30",
  "init_args": {
    "band_names": null,
    "query": null,
    "sort_by": null,
    "sort_ascending": true,
    "earthdata_token": null,
    "earthdata_username": null,
    "earthdata_password": null,
    "timeout_seconds": 30
  }
}
```

### Available Bands

The default band set includes:
- B01 (coastal)
- B02 (blue)
- B03 (green)
- B04 (red)
- B05 (nir)
- B06 (swir16)
- B07 (swir22)
- B09 (cirrus)
- B10 (lwir11)
- B11 (lwir12)

Additional supported raster assets:
- Fmask (fmask / qa)
- SAA (solar_azimuth)
- SZA (solar_zenith)
- VAA (view_azimuth)
- VZA (view_zenith)

Band names may be provided as either asset keys or the aliases above.
