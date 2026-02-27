## Dataset Configuration File

The dataset configuration file is a JSON file that specifies:

1. The layers in the dataset. Each layer contains a different raster or vector modality.
   For example, a dataset may have one layer for Sentinel-2 images, and another layer for
   building polygons from OpenStreetMap.
2. The configuration of the tile store, where items ingested from data sources are stored.
3. The configuration of the window storage, which tracks information like the name and
   bounds of each window.

The overall dataset configuration file looks like this:

```jsonc
{
  // The layers section is required and maps from layer name to layer config.
  "layers": {
    "layer_name": {
      // Layer config.

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

      // Layers to be populated automatically from a data source specify additional
      // configuration options.
      // The data source section is optional. If it is not set, it means that this layer
      // will be populated by the user, e.g. programmatically via the Python API.
      "data_source": {
        // The class path of the data source.
        "class_path": "rslearn.data_sources.gcp_public_data.Sentinel2",
        // Data sources may expose additional configuration options, passed via init_args.
        // class_path and init_args are handled by jsonargparse to instantiate the data
        // source class.
        "init_args": {
          // ...
        },
        // The query configuration specifies how items should be matched to windows. It is
        // optional, and the values below are defaults.,
        "query_config": {
          // The space mode must be "MOSAIC" (default), "CONTAINS", "INTERSECTS", or "SINGLE_COMPOSITE".
          "space_mode": "MOSAIC",
          // The max matches defaults to 1.
          "max_matches": 1,
          // For MOSAIC, the number of overlapping items wanted within each item group covering
          // the window (default 1). Set higher for compositing.
          "mosaic_compositing_overlaps": 1,
          // By default, the space mode controls how multiple item groups are created in case
          // max_matches > 1. If period_duration is set, the window time range is instead
          // divided into periods of this duration, and the space mode is applied within each
          // period to produce one item group per period.
          "period_duration": null,
          // When period_duration is set, whether to return item groups in reverse temporal
          // order (most recent first). Should always be set to false when setting period_duration.
          // The default is true for backwards compatibility (deprecated).
          "per_period_mosaic_reverse_time_order": false,
        },
        // The time offset is optional. It defaults to 0.
        "time_offset": "0d",
        // The duration is optional. It defaults to null.
        "duration": null,
        // The ingest flag is optional, and defaults to true.
        "ingest": true
      },
      // Re-sampling method to use during materialization. This only applies to raster
      // layers with a data source. It is used when there is a difference in CRS or
      // resolution between the item from the data source and the window's target.
      // It is one of "nearest", "bilinear" (default), "cubic", "cubic_spline".
      "resampling_method": "bilinear",
      // The compositing method to use, to handle raster item groups with more than one item.
      // It can be FIRST_VALID (default), MEAN, MEDIAN, or SPATIAL_MOSAIC_TEMPORAL_STACK.
      "compositing_method": "FIRST_VALID"
    },
    // ... (additional layers)
  },
  // The tile store config is optional; for most use cases, the default of using a
  // file-based tile store with GeoTIFFs and GeoJSONs works well.
  "tile_store": {
    // Tile store config.
  },
  // The window storage config is optional. It defaults to using a file-based
  // storage scheme (no database).
  "storage": {
    // Window storage config.
  }
}
```

The documents below detail the specification of different sections of the dataset
configuration file. Also see [Examples.md](Examples.md) for some examples of dataset
configuration files for different use cases.

- [LayerConfig](dataset_config/LayerConfig.md): the layer configuration, excluding
  data source options.
- [DataSourceConfig](dataset_config/DataSourceConfig.md): the portion of the layer
  configuration relating to configuring a data source (i.e., `data_source`,
  `resampling_method`, and `compositing_method` above).
- [TileStoreConfig](dataset_config/TileStoreConfig.md): the tile store config.
- [WindowStorageConfig](dataset_config/WindowStorageConfig.md): the window storage config.
