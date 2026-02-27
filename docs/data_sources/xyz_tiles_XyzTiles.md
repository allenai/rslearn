## rslearn.data_sources.xyz_tiles.XyzTiles

This data source is for web xyz image tiles (slippy tiles).

These tiles are usually in WebMercator projection, but different CRS can be configured.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.xyz_tiles.XyzTiles",
  "init_args": {
    // Required list of URL templates. The templates must include placeholders for {x}
    // (column), {y} (row), and {z} (zoom level).
    // Example: ["https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.jpg"]
    "url_templates": null,
    // Required list of time ranges. It should match the list of URL templates. This is
    // primarily useful with multiple URL templates, to distinguish which one should be
    // used depending on the window time range. If time is not important, then you can
    // set it arbitrarily.
    // Example: [["2024-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00"]]
    "time_ranges": null,
    // Required zoom level. Currently, a single zoom level must be specified, and tiles
    // will always be read at that zoom level, rather than varying depending on the
    // window resolution.
    // Example: 17 to use zoom level 17.
    "zoom": null,
    // The CRS of the xyz image tiles. Defaults to WebMercator.
    "crs": "EPSG:3857",
    // The total projection units along each axis. Defaults to 40075016.6856 which
    // corresponds to WebMercator. This is used to compute the pixel resolution, i.e. the
    // tiles split the world into 2^zoom tiles along each axis so the resolution is
    // (total_units / 2^zoom / tile_size) units/pixel.
    "total_units": 40075016.6856,
    // Apply an offset to the projection units when converting tile positions. Without an
    // offset, the WebMercator tile columns and rows would range from -2^(zoom-1) to
    // 2^(zoom-1). The default offset is half the default total units so that it
    // corresponds to the standard range from 0 to 2^zoom.
    "offset": 20037508.3428,
    // The size of tiles. The default is 256x256 which is typical.
    "tile_size": 256
  }
}
```

### Available Bands

The bands are named "R", "G", and "B" and are typically uint8 but the data type would
depend on the images returned by the URL template.

### Example

Here is a dataset configuration to show Google Maps Satellite images with a dummy time
range.

```json
{
  "layers": {
    "google_maps_satellite": {
      "band_sets": [{
          "bands": ["R", "G", "B"],
          "dtype": "uint8"
      }],
      "data_source": {
        "class_path": "rslearn.data_sources.xyz_tiles.XyzTiles",
        "init_args": {
          "url_templates": ["http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}"],
          "time_ranges": [["1900-01-01T00:00:00Z", "2100-01-01T00:00:00Z"]],
          "zoom": 17
        },
        "ingest": false
      },
      "type": "raster"
    }
  }
}
```

Save this to a dataset folder like `/path/to/dataset/config.json`. Then we can create a
sample window, and then run prepare and materialize.

```
export DATASET_PATH=/path/to/dataset
# This will create one 1024x1024 window at 1 m/pixel, which roughly corresponds to the
# zoom 17 resolution (see https://wiki.openstreetmap.org/wiki/Zoom_levels).
rslearn dataset add_windows --root $DATASET_PATH --group default --name seattle --box=-122.337,47.616,-122.337,47.616 --src_crs EPSG:4326 --window_size 1024 --utm --resolution 1 --start 2025-07-01T00:00:00Z --end 2025-08-01T00:00:00Z
rslearn dataset prepare --root $DATASET_PATH
rslearn dataset materialize --root $DATASET_PATH
```

You can then visualize the image in qgis:

```
qgis $DATASET_PATH/windows/default/seattle/layers/google_maps_satellite/R_G_B/geotiff.tif
```
