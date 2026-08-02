## rslearn.data_sources.nasa_ecostress.EcostressLSTE

NASA [ECOSTRESS](https://ecostress.jpl.nasa.gov/) Tiled Land Surface Temperature and
Emissivity Instantaneous L2 Global 70 m product (`ECO_L2T_LSTE`, version 002) served
from NASA's Earthdata Cloud (LP DAAC) as Cloud-Optimized GeoTIFFs. Discovery uses the
CMR STAC LPCLOUD endpoint, and direct materialization is supported.

Like the [NASA HLS](nasa_hls_Hls2.md) sources, this prefers LP DAAC `s3_*` assets and
automatically fetches temporary AWS credentials from the LP DAAC `s3credentials`
endpoint. It requires an Earthdata bearer token:

- `EARTHDATA_TOKEN`

### Region Behavior

Same as the HLS sources: LP DAAC temporary AWS credentials are intended for
same-region (`us-west-2`) direct S3 access. rslearn tries the `s3://` asset first and
falls back to the authenticated HTTPS asset URL when direct S3 access is unavailable
(e.g. local development outside `us-west-2`).

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.nasa_ecostress.EcostressLSTE",
  "init_args": {
    // Optional list of bands to expose. Defaults to ["LST"].
    "band_names": null,
    // Optional STAC query filter.
    "query": null,
    // Optional STAC sort property, e.g. "datetime".
    "sort_by": null,
    "sort_ascending": true,
    // Optional Earthdata bearer token override (otherwise EARTHDATA_TOKEN is used).
    "earthdata_token": null,
    "timeout": "30s"
  }
}
```

Because each layer is a separate COG, list the requested layers in `band_names`. For a
LST time series, set `max_matches > 1` in the layer's `query_config` (like other
rslearn sources, the default returns a single match).

### Available Bands

Default band:
- `LST` — land surface (skin) temperature, float32 **Kelvin**, `NaN` fill value.

Additional supported layers:
- `LST_err` — LST uncertainty (Kelvin)
- `EmisWB` — wideband emissivity
- `QC` — quality control bit field (uint16)
- `cloud` — cloud mask (uint8; 0 = clear, 1 = cloudy)
- `water` — water mask (uint8; 0 = land, 1 = water)
- `height` — surface elevation (meters)
- `view_zenith` — sensor view zenith angle (degrees)

Band names are matched to ECOSTRESS assets by their `_<band>` filename suffix.

### Cloud Masking

The LST retrieval runs on every pixel regardless of cloud cover, so **cloudy pixels
are not reliably set to `NaN`** in the `LST` layer — only pixels where no retrieval was
produced are filled with `NaN` (and read as the raster nodata value). For authoritative
cloud screening, NASA recommends the dedicated `cloud` layer: add `"cloud"` to
`band_names` and keep only pixels where it equals 0. (In v002, the QC layer no longer
carries cloud information — use the `cloud` layer instead.)
