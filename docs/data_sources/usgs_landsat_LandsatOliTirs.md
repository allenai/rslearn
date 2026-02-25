## rslearn.data_sources.usgs_landsat.LandsatOliTirs

This data source is for Landsat data from the USGS M2M API.

You can request access at https://m2m.cr.usgs.gov/.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.usgs_landsat.LandsatOliTirs",
  "init_args": {
    // M2M API username can be set here or via M2M_USERNAME environment variable.
    "username": null,
    // M2M API authentication token can be set here or via M2M_TOKEN environment variable.
    "token": null,
    // Sort by this attribute, either null (default, meaning arbitrary ordering) or
    // "cloud_cover".
    "sort_by": null
  }
}
```

### Available Bands

Pixel values are uint16.

- B1
- B2
- B3
- B4
- B5
- B6
- B7
- B8
- B9
- B10
- B11
