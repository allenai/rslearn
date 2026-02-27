## rslearn.data_sources.soilgrids.SoilGrids

This data source provides access to [ISRIC SoilGrids](https://www.isric.org/explore/soilgrids)
via the public WCS endpoints (e.g. `https://maps.isric.org/mapserv?map=/map/clay.map`).

This source is intended for **direct materialization** (set `"ingest": false` in the
layer's `data_source` config), since data is fetched on-demand per window.

### Configuration

Example (clay, user-provided WCS subset parameters):

```jsonc
{
  "class_path": "rslearn.data_sources.soilgrids.SoilGrids",
  "init_args": {
    "service_id": "clay",
    "coverage_id": "clay_0-5cm_mean"
    // Optional request CRS, defaults to EPSG:3857. You can specify either "EPSG:3857"
    // or the URN form "urn:ogc:def:crs:EPSG::3857".
    // "crs": "EPSG:3857"
  }
}
```

If `"width"`/`"height"` and `"resx"`/`"resy"` are omitted, rslearn will default to
requesting at ~250 m resolution in the request CRS and then reprojecting to the window
grid. For EPSG:4326 requests, SoilGrids requires `"width"`/`"height"` so rslearn will
default those to the window pixel size.

### Available Bands

- B1 (float32 recommended; scale/offset applied; set `nodata_vals` to `-32768`)
