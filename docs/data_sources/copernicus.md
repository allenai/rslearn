## rslearn.data_sources.copernicus.Copernicus

This data source is for images from the ESA Copernicus OData API. See
https://documentation.dataspace.copernicus.eu/APIs/OData.html for details about the API
and how to get an access token.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.copernicus.Copernicus",
  "init_args": {
    // Required dictionary mapping from a filename or glob string of an asset inside the
    // product zip file, to the list of bands that the asset contains. An example for
    // Sentinel-2 images is shown.
    "glob_to_bands": {
      "*/GRANULE/*/IMG_DATA/*_B01.jp2": ["B01"],
      "*/GRANULE/*/IMG_DATA/*_TCI.jp2": ["R", "G", "B"]
    },
    // Optional API access token. See https://documentation.dataspace.copernicus.eu/APIs/OData.html
    // for how to get a token. If not set, it is read from the environment variable
    // COPERNICUS_ACCESS_TOKEN. If that environment variable doesn't exist, then we
    // attempt to read the username/password from COPERNICUS_USERNAME and
    // COPERNICUS_PASSWORD (this is useful since access tokens are only valid for an hour).
    "access_token": null,
    // Optional query filter string to include when searching for items. This will be
    // appended to other name, geographic, and sensing time filters where applicable. For
    // example, "Collection/Name eq 'SENTINEL-2'". See the API documentation for more
    // examples.
    "query_filter": null,
    // Optional order by string to include when searching for items. For example,
    // "ContentDate/Start asc". See the API documentation for more examples.
    "order_by": null,
    // Optional product attribute name to sort returned products by that attribute. If
    // set, attributes will be expanded when listing products. Note that while order_by
    // uses the API to order products, the API provides limited options, and sort_by
    // instead is done after the API call.
    "sort_by": null,
    // If sort_by is set, sort in descending order instead of ascending order.
    "sort_desc": false,
    // Timeout for requests in seconds.
    "timeout": 10
  }
}
```

## rslearn.data_sources.copernicus.Sentinel2

This data source is for Sentinel-2 images from the ESA Copernicus OData API.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.copernicus.Sentinel2",
  "init_args": {
    // Required product type, either "L1C" or "L2A".
    "product_type": "L1C",
    // Flag (default false) to harmonize pixel values across different processing
    // baselines (recommended), see
    // https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
    "harmonize": false,
    // See rslearn.data_sources.copernicus.Copernicus for details about the configuration
    // options below.
    "access_token": null,
    "order_by": null,
    "sort_by": null,
    "sort_desc": false,
    "timeout": 10
  }
}
```

### Available Bands

- B01
- B02
- B03
- B04
- B05
- B06
- B07
- B08
- B09
- B11
- B12
- B8A
- R, G, B (uint8)
- B10 (L1C only)
- AOT (L2A only)
- WVP (L2A only)
- SCL (L2A only)

## rslearn.data_sources.copernicus.Sentinel1

This data source is for Sentinel-1 images from the ESA Copernicus OData API. Currently
only IW GRDH VV+VH products are supported, even though all Sentinel-1 scenes are
available in the data source.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.copernicus.Sentinel1",
  "init_args": {
    // Required product type, must be "IW_GRDH".
    "product_type": "IW_GRDH",
    // Required polarisation, must be "VV_VH".
    "polarisation": "VV_VH",
    // Optional orbit direction to filter by, either "ASCENDING" or "DESCENDING". The
    // default is to not filter (so both types of scenes are included/mixed).
    "orbit_direction": null,
    // See rslearn.data_sources.copernicus.Copernicus for details about the configuration
    // options below.
    "access_token": null,
    "order_by": null,
    "sort_by": null,
    "sort_desc": false,
    "timeout": 10
  }
}
```
