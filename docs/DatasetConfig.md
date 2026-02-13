Dataset Configuration File
--------------------------

The dataset configuration file is a JSON file that specifies the layers in the dataset,
and the configuration of the tile store.

Each layer contains a different raster or vector modality. For example, a dataset may
have one layer for Sentinel-2 images, and another layer for building polygons from
OpenStreetMap.

Layers may be populated manually, or populated automatically from data sources. rslearn
retrieves data from data sources in three steps: prepare, ingest, and materialize (see
[CoreConcepts](CoreConcepts.md)). The tile store is an intermediate storage to store
the ingested items.

Below, we detail the dataset configuration file specification. See
[Examples.md](Examples.md) for some examples of dataset configuration files for
different use cases.

The overall dataset configuration file looks like this:

```jsonc
{
  // The layers section is required and maps from layer name to layer config.
  "layers": {
    "layer_name": {
      // Layer config.
    },
    // ... (additional layers)
  },
  // The tile store config is optional; for most use cases, the default of using a
  // file-based tile store with GeoTIFFs and GeoJSONs works well.
  "tile_store": {
    // Tile store config.
  }
}
```


Layer Config
------------

An individual layer is configured with a layer config dict as follows:

```jsonc
{
  // The layer type must be "raster" or "vector".
  "type": "raster",
  // The alias is optional. It overrides the name of this layer in the tile store,
  // which simply defaults to the layer name.
  "alias": "optional alias",
  // The data source section is optional. If it is not set, it means that this layer
  // will be populated by the user, e.g. using a separate Python script.
  "data_source": {
    // Data source specification.
  },
  // Raster and vector layers have additional type-specific configuration.
}
```

### Alias

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


Raster Layers
-------------

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
  ],
  // Re-sampling method to use during materialization. This only applies to raster
  // layers with a data source. It is used when there is a difference in CRS or
  // resolution between the item from the data source and the window's target.
  // It is one of "nearest", "bilinear" (default), "cubic", "cubic_spline".
  "resampling_method": "bilinear",
  // Method how to select pixel values when multiple items in a group cover the same
  // pixel. One of "FIRST_VALID", "MEAN", "MEDIAN". Defaults to "FIRST_VALID".
  "compositing_method": "FIRST_VALID"
}
```

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
    // Required format to save the images as, one off "geotiff", "png", "jpeg".
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


Vector Layers
-------------

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
    // Example: "projection": {"crs": "EPSG:3857, "x_resolution": 10, "y_resolution": 10}
    "projection": null
  }
}
```


Data Source Specification
-------------------------

The data source specification looks like this:

```jsonc
{
  // The class path of the data source.
  "class_path": "rslearn.data_sources.gcp_public_data.Sentinel2",
  // Data sources may expose additional configuration options, passed via init_args.
  // class_path and init_args are handled by jsonargparse to instantiate the data
  // source class.
  "init_args": {
    // ...
  },
  // The query configuration specifies how items should be matched to windows. It is
  // optional, and the values below are defaults.
  "query_config": {
    // The space mode must be "MOSAIC" (default), "CONTAINS", "INTERSECTS", or "PER_PERIOD_MOSAIC".
    "space_mode": "MOSAIC",
    // The max matches defaults to 1.
    "max_matches": 1,
    // For MOSAIC and PER_PERIOD_MOSAIC modes, the number of overlapping items wanted
    // within each item group covering the window (default 1). Set higher for compositing.
    "mosaic_compositing_overlaps": 1,
    // For PER_PERIOD_MOSAIC mode, the duration of each sub-period (default "30d").
    "period_duration": "30d",
    // For PER_PERIOD_MOSAIC mode, whether to return item groups in reverse temporal
    // order (most recent first). Should always be set to false when using
    // PER_PERIOD_MOSAIC. The defaults is true for backwards compatibility (deprecated).
    "per_period_mosaic_reverse_time_order": false
  },
  // The time offset is optional. It defaults to 0.
  "time_offset": "0d",
  // The duration is optional. It defaults to null.
  "duration": null,
  // The ingest flag is optional, and defaults to true.
  "ingest": true
}
```

`class_path` and `init_args` configure the data source itself. See
[DataSources](DataSources.md) for details on all of the built-in data sources in rslearn.

`query_config`, `time_offset`, and `duration` specify how windows should be matched
to items during the prepare stage.

`ingest` specifies whether to ingest data into the tile store. Some data sources
support directly materializing from the source without using the tile store.

### Query Configuration

The query configuration specifies how items should be matched to windows.

For each window, the matching process starts with a list of items provided by the data
source that intersect the window's spatial extent and time range. The output from
matching is a `list[list[Item]]`. This is a list of item groups, where each item group
corresponds to the items that will be used to create one composite of raster or vector
data.

#### Space Mode

The space mode defines the matching strategy. It interacts with `max_matches`, which
specifies the maximum number of item groups to produce.

**CONTAINS.** Use items that fully contain the window bounds. The resulting item groups
will each consist of exactly one item. This strategy iterates over the items in the
order they are provided by the data source (some data sources provide sorting
options, e.g. sort by cloud cover), filtering ones that do not contain the window,
and creating single-item item groups for the rest, continuing until there are no more
items or `max_matches` item groups have been created.

**INTERSECTS.** Use items that intersect the window bounds. As with CONTAINS, the
resulting item groups will each consist of exactly one item.

**MOSAIC.** Create mosaics, where each item group combines multiple items from the data
source as needed to cover the entire window. In this case, each item group may
include multiple items. This strategy initializes a buffer of `max_matches` empty
item groups. It then iterates over the items, adding each item to the first group
that the item provides additional coverage for (skipping groups that already cover
all the portions of the window that the new item covers). Finally, the non-empty
groups are returned.

**PER_PERIOD_MOSAIC.** Create one mosaic per sub-period of the time range. When using
MOSAIC, each resulting item group could arbitrarily combine items from across the
window's time range. PER_PERIOD_MOSAIC is useful if you want each mosaic to
correspond to a sub-period, e.g. getting a mosaic for each month of the year. The
duration of the sub-periods is controlled by `period_duration`. This strategy starts
from the most recent sub-period, and finds all items temporally intersecting that
sub-period. If no items are found, the sub-period is skipped; otherwise, it iterates
over the items similar to MOSAIC, incorporating each item that covers new portions of
the window. It continues until either there are no more sub-periods (it reaches the
beginning of the window's time range) or it has created `max_matches` item groups.

**Example.**
Consider a window covering a 10km x 10km region with a time range of January 1 to April
1. The data source returns four items in order:

- Item A: covers the full window (10km x 10km), from January 15
- Item B: covers the left half of the window (5km x 10km), from January 20
- Item C: covers the right half of the window (5km x 10km), from March 10
- Item D: covers the full window (10km x 10km), from March 20

With `max_matches=2`:

- CONTAINS returns `[[A], [D]]`. Both A and D fully contain the window. B and C are
  skipped because they only partially cover the window.
- **INTERSECTS** returns `[[A], [B]]`. All four items intersect the window, but we stop
  at 2 due to max_matches. Each item becomes its own single-item group.
- **MOSAIC** returns `[[A], [B, C]]`. Item A covers the full window, completing the
  first mosaic. Item B doesn't add coverage to the first mosaic (A already covers it),
  so B starts the second mosaic. Item C adds the right half to the second mosaic. Item D
  doesn't add new coverage to either mosaic.
- **PER_PERIOD_MOSAIC** with `period_duration="29d"` returns `[[A], [C, D]]`. The time
  range is split into January, February, and March sub-periods. For March, items C and
  D are combined into one mosaic. February is skipped since there are no matching
  items. For January, item A covers the full window.

#### Compositing

For vector data, non-singleton item groups are handled by concatenating the vector
features across items in the group.

Compositing raster data is more complex, and a `compositing_method` option is provided
to control the behavior. By default, `compositing_method = FIRST_VALID`; for each
pixel and band, the value is set based on the first item that is not NODATA at that
pixel and band. The `compositing_method` can instead be set to MEAN or MEDIAN to
compute the mean or median across all items in the group that are not NODATA at that
pixel and band.

#### Compositing Overlaps

For MOSAIC and PER_PERIOD_MOSAIC, the default behavior is to create item groups that
cover the window's spatial extent once. `mosaic_compositing_overlaps` can be set
greater than 1 to have each item group cover the window multiple times. This is useful
when computing mean or median composites for each item group.

### Time Offset and Duration

By default, the time range used for requesting items from the data source and applying
the matching strategy is the time range of the window. The time range can be adjusted
by setting `time_offset` and/or `duration`.

`time_offset` specifies a positive or negative time delta. If set, the time delta is
added to the time range. It is parsed by [pytimeparse](https://github.com/wroberts/pytimeparse).
For example:

- "30d" means to adjust the time range 30 days into the future.
- "-30d" means to adjust the time range 30 days into the past.

`duration` specifies a positive time delta. If set, the end time of the time range is
set to the start time plus `duration`.

Suppose the window time range is [2024-01-01, 2024-02-01].

- With time_offset=30d, the matching time range is [2024-01-31, 2024-03-02].
- With duration=180d, the matching time range is [2024-01-01, 2024-06-29].
- With time_offset=30d AND duration=180d, the matching time range is [2024-01-31, 2024-07-29].

### Ingest Flag

The ingest flag specifies whether this data source should be ingested.

The default interface for data sources is represented as a collection of items, where
the items are matched to windows and then the items need to first be ingested before
they can be re-projected and cropped to align with individual windows. However, some
data sources support (or require) directly materializing data into the window.

For example, `XyzTiles` represents a slippy map tiles layer, i.e. a mosaic covering the
entire world that is broken up into tiles. Rather than representing each tile as a
separate item (which would be inefficient), it only supports directly materializing the
data into windows. Then, when using this data source, the ingest flag should be set to
false.

Other data sources like PlanetaryComputer (which uses COGs on Microsoft Planetary
Computer) support both approaches (download entire COGs and then align locally, or read
crops directly from the remote COGs). In this case, ingestion will be faster for dense
windows while direct materialization will be faster for sparse windows.

### rslearn.data_sources.earthdaily.Sentinel2

Sentinel-2 L2A data on EarthDaily platform (collection: `sentinel-2-c1-l2a`).

This data source requires the optional `earthdaily[platform]` dependency and EarthDaily
credentials in the environment (see [EarthDaily documentation](https://earthdaily.github.io/earthdaily-python-client/main/) for supported methods).

By default, this data source applies per-asset scale/offset values from STAC
`raster:bands` metadata (`harmonize: true`) using `physical = raw * scale + offset`. Set
`harmonize: false` to keep raw values.

The additional data source configuration looks like this:
Only the keys documented below are supported in `init_args`; unknown keys will raise an error.

```jsonc
{
  // Whether to apply STAC `raster:bands` scale/offset (default true). Set to false to
  // keep raw values.
  "harmonize": true,
  // Optional: EarthDaily Sentinel-2 asset keys to use (default null). If null and the
  // layer config is available, assets are inferred from the layer's band sets.
  // Example: ["red", "green", "blue", "nir", "swir16", "swir22"]
  "assets": null,
  // Optional: maximum cloud cover (%) to filter items at search time.
  // If set, it overrides any `eo:cloud_cover` filter in `query`.
  "cloud_cover_max": null,
  // Optional: default cloud cover threshold (%) to apply if cloud_cover_max is not set.
  "cloud_cover_threshold": null,
  // Maximum number of STAC items to fetch per window before rslearn grouping/matching.
  "search_max_items": 500,
  // Optional ordering of items before grouping (useful with SpaceMode.COMPOSITE +
  // CompositingMethod.FIRST_VALID): "cloud_cover" (default), "datetime", or null.
  "sort_items_by": "cloud_cover",
  // Whether to apply an SCL-based cloud mask during ingest (default false). If true,
  // cloudy pixels are set to mask_nodata_value before writing to the dataset tile store.
  // Note: this only applies when ingest is enabled (the default).
  "apply_cloud_mask": false,
  // Which asset key to use as the mask band (default "scl").
  "mask_band": "scl",
  // SCL values treated as cloudy (default [3, 8, 9, 10]); set to override.
  "exclude_scl_values": null,
  // Nodata value to write into cloudy pixels.
  "mask_nodata_value": 0,
  // Optional: STAC API `query` filter passed to searches.
  // Example: {"s2:product_type": {"eq": "S2MSI2A"}}
  // Note: if cloud_cover_max/cloud_cover_threshold is set, the effective query also
  // includes an `eo:cloud_cover` upper bound.
  "query": null,
  // Optional: STAC item property to sort by before grouping/matching (default null).
  // If set, it takes precedence over sort_items_by.
  "sort_by": null,
  // Whether to sort ascending when sort_by is set (default true).
  "sort_ascending": true,
  // Optional cache directory for cached item metadata.
  "cache_dir": null,
  // Timeout for HTTP asset downloads.
  "timeout": "10s",
  // Retry settings for EarthDaily API client requests (search/get item).
  "max_retries": 3,
  "retry_backoff_factor": 5.0,
  // EarthDaily service name (only "platform" is supported).
  "service_name": "platform"
}
```

Example:

```jsonc
{
  "layers": {
    "sentinel2": {
      "type": "raster",
      "band_sets": [{
        // When harmonize=true (default), reflectance bands are typically scaled to float.
        // Use float32 (recommended) or set harmonize=false to keep raw integer values.
        "dtype": "float32",
        "bands": ["B02", "B03", "B04", "B08"]
      }],
      "compositing_method": "FIRST_VALID",
      "data_source": {
        "class_path": "rslearn.data_sources.earthdaily.Sentinel2",
        "query_config": {
          "space_mode": "COMPOSITE",
          "time_mode": "WITHIN",
          "max_matches": 1
        },
        "init_args": {
          "cloud_cover_max": 15.0,
          "sort_items_by": "cloud_cover",
          "apply_cloud_mask": true,
          "mask_nodata_value": 0
        }
      }
    }
  }
}
```

Available bands:
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
- R, G, B (from the `visual` asset)
- scl, aot, wvp
