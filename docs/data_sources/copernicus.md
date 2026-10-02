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

## rslearn.data_sources.copernicus.Sentinel3OlciEFR

This data source retrieves [Sentinel-3 OLCI Level-1](https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel3.html#sentinel-3-olci-level-1) Earth Observation Full Resolution
(`OL_1_EFR___`) products from Copernicus Data Space. During ingestion it converts the
21 top-of-atmosphere radiance channels to unitless top-of-atmosphere reflectance using
the product solar flux, detector index, and solar-zenith annotations. Reflectance is
not clipped. The geolocated swath is interpolated to an intermediate WGS84 grid before
normal rslearn materialization.

It requires the optional NetCDF/xarray/scipy dependencies included in
`rslearn[extra]`. Direct materialization is not supported; keep ingestion enabled.

!!! warning "Changing windows after ingestion"

    Ingestion crops each swath to the requested windows plus `swath_padding`, and the
    tile store does not expand an item that is already marked complete. If windows are
    added, moved, or resized after ingestion, clear this layer's cached tile-store
    entries and ingest it again. With the default tile store, remove the corresponding
    layer (or layer alias) directory under `<dataset_path>/tiles/`. Then run the
    following commands for the affected windows and layer:

    ```bash
    rslearn dataset prepare --root <dataset_path> --enabled-layers <layer_name> --force
    rslearn dataset ingest --root <dataset_path> --enabled-layers <layer_name>
    ```

    Custom tile stores must be cleared through their configured backend.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.copernicus.Sentinel3OlciEFR",
  "init_args": {
    // Intermediate WGS84 grid resolution in degrees (approximately 300 m).
    "grid_resolution": 0.0027,
    "nodata_value": -9999.0,
    // Extra area retained around requested windows before swath interpolation.
    "swath_padding": 0.1,
    // See Copernicus above for authentication, ordering, and timeout options.
    "access_token": null,
    "order_by": null,
    "sort_by": null,
    "sort_desc": false,
    "timeout": 10
  }
}
```

### Available Bands

- `Oa01_reflectance` through `Oa21_reflectance` (`float32`, unitless)

## rslearn.data_sources.copernicus.Sentinel3SlstrRBT

This data source retrieves [Sentinel-3 SLSTR Level-1](https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel3.html#sentinel-3-slstr-level-1) Radiance and Brightness
Temperature (`SL_1_RBT___`) products from Copernicus Data Space. It converts the
nadir-view S1-S6 channels to unitless top-of-atmosphere reflectance and reads the
nadir-view S7-S9 brightness temperatures in kelvin.

The reflective and thermal channels are ingested independently because they use
different native grids. They may be placed in separate rslearn band sets with distinct
materialization resolutions. The source requires `rslearn[extra]` and ingestion.

!!! warning "Changing windows after ingestion"

    Ingestion crops each swath to the requested windows plus `swath_padding`, and the
    tile store does not expand an item that is already marked complete. If windows are
    added, moved, or resized after ingestion, clear this layer's cached tile-store
    entries and ingest it again. With the default tile store, remove the corresponding
    layer (or layer alias) directory under `<dataset_path>/tiles/`. Then run the
    following commands for the affected windows and layer:

    ```bash
    rslearn dataset prepare --root <dataset_path> --enabled-layers <layer_name> --force
    rslearn dataset ingest --root <dataset_path> --enabled-layers <layer_name>
    ```

    Custom tile stores must be cleared through their configured backend.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.copernicus.Sentinel3SlstrRBT",
  "init_args": {
    // Intermediate WGS84 grids, approximately 500 m and 1 km respectively.
    "reflectance_grid_resolution": 0.0045,
    "bt_grid_resolution": 0.009,
    "nodata_value": -9999.0,
    "swath_padding": 0.1,
    // See Copernicus above for authentication, ordering, and timeout options.
    "access_token": null,
    "order_by": null,
    "sort_by": null,
    "sort_desc": false,
    "timeout": 10
  }
}
```

### Available Bands

- `S1_reflectance` through `S6_reflectance` (`float32`, unitless)
- `S7_BT`, `S8_BT`, and `S9_BT` (`float32`, kelvin)

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
