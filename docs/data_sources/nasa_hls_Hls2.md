## rslearn.data_sources.nasa_hls.Hls2S30

NASA LP DAAC HLS v2.0 Sentinel-2 ([HLSS30](https://hls.gsfc.nasa.gov/)) data on
CMR STAC / LP DAAC cloud storage. Direct materialization is supported.

This data source prefers LP DAAC `s3_*` assets and automatically fetches temporary
AWS credentials from the LP DAAC `s3credentials` endpoint. Set:

- `EARTHDATA_TOKEN`

### Region Behavior

LP DAAC's temporary AWS credentials are intended for same-region direct S3 access,
with best support/performance in `us-west-2`.

- If your job is running in or near `us-west-2`, rslearn will try the LP DAAC
  `s3://` asset first.
- If direct S3 access is denied or unavailable, rslearn falls back to the
  authenticated HTTPS asset URL from STAC.
- The HTTPS fallback is intended to make local development and non-`us-west-2`
  runs work, but it will usually be slower than direct in-region S3 access.

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
    // Optional Earthdata bearer token override (otherwise EARTHDATA_TOKEN is used).
    "earthdata_token": null,
    "timeout": "30s"
  }
}
```

### Available Bands

The default band set includes:
- B01
- B02
- B03
- B04
- B05
- B06
- B07
- B08
- B8A
- B11
- B12

Additional supported raster assets:
- B09
- B10
- Fmask
- SAA
- SZA
- VAA
- VZA

Band names are the HLS asset keys listed above.

### HLS Fmask Cloud-Aware Compositing

To rank HLS items by per-window cloudiness before FIRST_VALID compositing, configure:

```jsonc
{
  "layers": {
    "hls": {
      "type": "raster",
      "band_sets": [
        {"bands": ["red", "green", "blue"], "dtype": "int16", "nodata_value": -9999},
        {"bands": ["fmask"], "dtype": "uint8"}
      ],
      "data_source": {
        "class_path": "rslearn.data_sources.nasa_hls.Hls2",
        "init_args": {
          "query": {"eo:cloud_cover": {"lt": 90}}
        },
        "query_config": {
          "max_matches": 8,
          "space_mode": "MOSAIC"
        }
      },
      "compositing_method": {
        "class_path": "rslearn.dataset.hls_fmask.HlsFmaskFirstValid",
        "init_args": {
          "fmask_band": "fmask"
        }
      }
    }
  }
}
```

When using `Hls2S30` or `Hls2L30`, use `fmask_band: "Fmask"` and include `Fmask` in a
configured `band_sets` entry.

## rslearn.data_sources.nasa_hls.Hls2

Combined NASA LP DAAC HLS v2.0 time-series datasource with a shared semantic band
schema across Sentinel-2 and Landsat observations.

This datasource is intended for single chronological time-series use. It does **not**
expose raw `Bxx` asset keys, because those keys do not mean the same thing across
HLSS30 and HLSL30. Instead it exposes semantic band names shared by both sensors.

Authentication, direct S3 preference, and HTTPS fallback behavior are the same as
for `rslearn.data_sources.nasa_hls.Hls2S30`.

For multi-item chronological series, configure the layer query with `max_matches > 1`.
Like other rslearn datasources, the default query behavior only returns a single match.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.nasa_hls.Hls2",
  "init_args": {
    // Optional semantic band list. Defaults to the common reflectance bands below.
    "band_names": null,
    // Optional source subset. Valid values are "sentinel" and "landsat".
    "sources": ["sentinel", "landsat"],
    "query": null,
    // Defaults to datetime so the merged series is chronological.
    "sort_by": "datetime",
    "sort_ascending": true,
    "earthdata_token": null,
    "timeout": "30s"
  }
}
```

### Available Bands

Default bands:
- coastal
- blue
- green
- red
- nir
- swir16
- swir22

Optional additional shared bands:
- cirrus
- fmask
- solar_azimuth
- solar_zenith
- view_azimuth
- view_zenith

Use `Hls2S30` or `Hls2L30` instead if you need source-specific raw bands like
Sentinel red-edge bands or Landsat thermal bands.

## rslearn.data_sources.nasa_hls.Hls2L30

NASA LP DAAC HLS v2.0 Landsat ([HLSL30](https://hls.gsfc.nasa.gov/)) data on
CMR STAC / LP DAAC cloud storage. Direct materialization is supported.

Authentication and temporary S3 credential handling are the same as for
`rslearn.data_sources.nasa_hls.Hls2S30`.

The same region behavior applies here: direct `s3://` reads are preferred when
same-region LP DAAC access is available, and rslearn falls back to authenticated
HTTPS otherwise.

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
    "timeout": "30s"
  }
}
```

### Available Bands

The default band set includes:
- B01
- B02
- B03
- B04
- B05
- B06
- B07
- B09
- B10
- B11

Additional supported raster assets:
- Fmask
- SAA
- SZA
- VAA
- VZA

Band names are the HLS asset keys listed above.
