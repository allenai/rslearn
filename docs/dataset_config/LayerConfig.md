[Back to DatasetConfig.md](../DatasetConfig.md)

## Layer Config

An individual layer is configured with a layer config dict as follows:

```jsonc
{
  // Layer type is "raster" or "vector".
  "type": "raster",
  // Optional layer alias. If set, it overrides the name of this layer in the tile
  // store, which simply defaults to the layer name.
  "alias": "optional alias",

  // Raster layers options.
  // The band sets specify the groups of bands that are present in this layer.
  "band_sets": [...],

  // Vector layer options.
  // The vector format configures how features are stored (default GeoJSON).
  "vector_format": {...},

  // Remaining options are specific to layers that are automatically populated from a
  // data source. These options are documented in DataSourceConfig.md.
  "data_source": {...},
  "resampling_method": "...",
  "compositing_method": "..."
}
```

## Alias

The alias overrides the name of the layer in the tile store.

This is primarily useful when you have a dataset where you find it necessary to define
multiple layers that reference the same data source. Without an alias, items for each
layer will be written to separate folders in the tile store (based on the layer names).
This means that, if the same item appears in both layers across the dataset windows, it
would be ingested once for each layer into the tile store. Setting the alias for both
layers to the same value ensures that their items are written to the same location in
the tile store, avoiding this duplicate ingestion.

Here is an example for Sentinel-2 L1C from GCS, where there are two layers. Each layer
creates a mosaic, but the second layer creates a mosaic 60 days in the future. The
duration of the layers is controlled by the duration of the window's time range.

```jsonc
{
  "layers": {
    "sentinel2_current": {
      "type": "raster",
      "band_sets": [{
        "dtype": "uint8",
        "bands": ["R", "G", "B"]
      }],
      "data_source": {
        "class_path": "rslearn.data_sources.gcp_public_data.Sentinel2",
        "init_args": {
          "index_cache_dir": "cache/sentinel2/",
          "sort_by": "cloud_cover",
          "use_rtree_index": false
        }
      },
      "alias": "sentinel2"
    },
    "sentinel2_future": {
      "type": "raster",
      "band_sets": [{
        "dtype": "uint8",
        "bands": ["R", "G", "B"]
      }],
      "data_source": {
        "class_path": "rslearn.data_sources.gcp_public_data.Sentinel2",
        "init_args": {
          "index_cache_dir": "cache/sentinel2/",
          "sort_by": "cloud_cover",
          "use_rtree_index": false
        },
        // The time offset is documented later.
        "time_offset": "60d"
      },
      "alias": "sentinel2"
    }
  }
}
```

## Raster Layers


Raster layers have additional configuration:

```jsonc
{
  "type": "raster",
  // The band sets specify the groups of bands that are present in this layer. If there
  // is a data source, then these bands will be read from the data source (mixing bands
  // from multiple source assets as needed).
  "band_sets": [
    {
      // Required data type, one of "uint8", "uint16", "uint32", "int32", "float32".
      "dtype": "uint8",
      // Required list of band names.
      "bands": ["R", "G", "B"],
      // Optional raster format, defaults to GeoTIFF without additional options.
      // Example: {"class_path": "rslearn.utils.raster_format.SingleImageRasterFormat", "init_args": {"format": "png"}}
      "format": {
        "class_path": "rslearn.utils.raster_format.GeotiffRasterFormat",
        "init_args": {}
      },
      // Optional zoom offset (default 0).
      "zoom_offset": 0,
      // Optional remap configuration for remapping pixel values during
      // materialization (default is to not perform any remapping).
      "remap": null,
    },
    // ... (additional band sets)
  ]
}
```

Below, we document the `format`, `zoom_offset`, and `remap` arguments within the band
set config.

### Raster Format

The raster format specifies how to encode and decode the raster data in storage. The
default is to save as GeoTIFF but you can customize this to e.g. save as PNG instead,
or customize the GeoTIFF compression and other options.

The available formats are:

- `rslearn.utils.raster_format.GeotiffRasterFormat`: save the raster as a GeoTIFF (default).
- `rslearn.utils.raster_format.ImageTileRasterFormat`: split the raster into tiles along a grid, and store the
  tiles.
- `rslearn.utils.raster_format.SingleImageRasterFormat`: save the raster as a single PNG or JPEG.

GeotiffRasterFormat configuration:

```jsonc
{
  "class_path": "rslearn.utils.raster_format.GeotiffRasterFormat",
  "init_args": {
    // What block size to use in the output GeoTIFF. Tiling is only enabled if the size
    // of the GeoTIFF exceeds this block size on at least one dimension. The default is
    // 512.
    "block_size": 512,
    // Whether to always produce a tiled GeoTIFF (instead of only if the raster is large
    // enough). Default false.
    "always_enable_tiling": false,
    // Arbitrary options to pass to rasterio when encoding GeoTIFFs.
    // Example: {"compress": "zstd", "predictor": 2, "zstd_level": 1}
    "geotiff_options": {}
  }
}
```

ImageTileRasterFormat configuration:

```jsonc
{
  "class_path": "rslearn.utils.raster_format.ImageTileRasterFormat",
  "init_args": {
    // Required format to save the images as, one of "geotiff", "png", "jpeg".
    // With png and jpeg, only 1-band and 3-band band sets are supported.
    "format": "png",
    // The tile size, default 512.
    "tile_size": 512
  }
}
```

SingleImageRasterFormat configuration:

```jsonc
{
  "class_path": "rslearn.utils.raster_format.SingleImageRasterFormat",
  "init_args": {
    // Required format, either "png" or "jpeg".
    // With png and jpeg, only 1-band and 3-band band sets are supported.
    "format": "png"
  }
}
```

### Zoom Offset

A non-zero zoom offset specifies that rasters for this band set should be stored at a
different resolution than the window's resolution.

A positive zoom offset means the resolution will be 2^offset higher than the window
resolution. For example, if the window resolution is 10 m/pixel, and the zoom offset is
2, then the raster will be stored at 2.5 m/pixel.

A negative zoom offset means the resolution will be 2^offset lower than the window
resolution. For example, if the window resolution is 10 m/pixel, and the zoom offset is
-2, then the raster will be stored at 40 m/pixel.

### Remap

Remapping specifies a way to remap pixel values during materialization. The default is
to perform no remapping.

The available remappers are:
- "linear": linear remapping.

LinearRemapper configuration:

```jsonc
{
  "name": "linear",
  // Required source range. Source values outside this range will be clipped to the
  // range.
  "src": [0, 8000],
  // Required destination range to remap to. With the example values here, a source
  // value of 0 (or lower) would be remapped to 128, while 4000 would be mapped to 192,
  // and 8000 or higher would be mapped to 256.
  "dst": [128, 256]
}
```

## Vector Layers


Vector layers have additional configuration:

```jsonc
{
  "type": "vector",
  // Optional vector format, defaults to GeoJSON.
  "vector_format": {
    "class_path": "rslearn.utils.vector_format.GeojsonVectorFormat",
    "init_args": {}
  }
}
```

### Vector Format

The vector format specifies how to encode and decode the vector data in storage.

The available formats are:

- `rslearn.utils.vector_format.GeojsonVectorFormat`: save the vector as one GeoJSON (default).
- `rslearn.utils.vector_format.TileVectorFormat`: split the vector data into tiles and store each as a separate GeoJSON.

GeojsonVectorFormat configuration:

```jsonc
{
  "class_path": "rslearn.utils.vector_format.GeojsonVectorFormat",
  "init_args": {
    // The coordinate mode. It controls the projection to use for coordinates written to the
    // GeoJSON files. "pixel" (default) means we write them as is, "crs" means we just
    // undo the resolution in the Projection so they are in CRS coordinates, and "wgs84"
    // means we always write longitude/latitude. When using "pixel", the GeoJSON will not
    // be readable by GIS tools since it relies on a custom encoding.
    "coordinate_mode": "pixel"
  }
}
```

TileVectorFormat configuration:

```jsonc
{
  "class_path": "rslearn.utils.vector_format.TileVectorFormat",
  "init_args": {
    // The tile size, default 512.
    "tile_size": 512,
    // The Projection to use for tiling. Features will be re-projected to this
    // projection, and then only stored in the tiles that they intersect. Tiles are
    // aligned with pixel coordinates in the projection, e.g. tile (0, 0) with
    // tile_size=512 covers pixels from (0, 0) to (512, 512).
    // The default is to use the projection of the first feature passed to the
    // encode_vector function.
    // Example: "projection": {"crs": "EPSG:3857", "x_resolution": 10, "y_resolution": 10}
    "projection": null
  }
}
```
