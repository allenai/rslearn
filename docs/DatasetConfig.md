Dataset Configuration File
--------------------------

The dataset configuration file is a JSON file that specifies the layers in the dataset,
and the configuration of the tile store.

Each layer contains a different raster or vector modality. For example, a dataset may
have one layer for Sentinel-2 images, and another layer for building polygons from
OpenStreetMap.

Layers may be populated manually, or populated automatically from data sources. rslearn
retrieves data from data sources in three steps:

1. Prepare: identify items in the data source that correspond to windows in the
   dataset.
2. Ingest: download those items to the dataset's tile store.
3. Materialize: crop and re-project the items relevant to each window from the tile
   store as needed to align them with the window.

The tile store is an intermediate storage to store the ingested items.

Below, we detail the dataset configuration file specification. See
[DatasetConfigExamples.md](DatasetConfigExamples.md) for some examples of dataset
configuration files for different use cases.

The overall dataset configuration file looks like this:

```jsonc
{
  // The layers section is required and maps from layer name to layer specification.
  "layers": {
    "layer_name": {
      // Layer specification.
    },
    // ... (additional layers)
  },
  // The tile store section is optional. It contains the tile store specification.
  "tile_store": {
    // Tile store specification.
  }
}
```


Layer Specification
-------------------

The general layer specification looks like this:

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
        "name": "rslearn.data_sources.gcp_public_data.Sentinel2",
        "index_cache_dir": "cache/sentinel2/",
        "sort_by": "cloud_cover",
        "use_rtree_index": false
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
        "name": "rslearn.data_sources.gcp_public_data.Sentinel2",
        "index_cache_dir": "cache/sentinel2/",
        "sort_by": "cloud_cover",
        "use_rtree_index": false,
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
      // Example: {"name": "single_image", "format": "png"}
      "format": null,
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
  "resampling_method": "bilinear"
}
```

### Raster Format

The raster format specifies how to encode and decode the raster data in storage. The
default is to save as GeoTIFF but you can customize this to e.g. save as PNG instead,
or customize the GeoTIFF compression and other options.

The available formats are:

- "geottiff": save the raster as a GeoTIFF (default).
- "image_tile": split the raster into tiles along a grid, and store the
  tiles.
- "single_image": save the raster as a single PNG or JPEG.

GeotiffRasterFormat configuration:

```jsonc
{
  "name": "geotiff",
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
```

ImageTileRasterFormat configuration:

```jsonc
{
  "name": "image_tile",
  // Required format to save the images as, one off "geotiff", "png", "jpeg".
  "format": "png",
  // The tile size, default 512.
  "tile_size": 512
}
```

SingleImageRasterFormat configuration:

```jsonc
{
  "name": "single_image",
  // Required format, either "png" or "jpeg".
  "format": "png"
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
  // value of 0 (or lower) would be remapped to 128, while 4000 would be mapped to 192.
  "dst": [128, 256]
}
```


Vector Layers
-------------

Vector layers have additional configuration:

```jsonc
{
  "type": "vector",
  // The zoom offset. This is similar to the raster band set zoom offset, and defaults
  // to 0.
  "zoom_offset": 0,
  // Optional vector format, defaults to GeoJSON.
  "format": {"name": "geojson"}
}
```

### Vector Format

The vector format specifies how to encode and decode the vector data in storage.

The available formats are:

- "geojson": save the vector as one GeoJSON (default).
- "tile": split the vector data into tiles and store each as a separate GeoJSON.

GeojsonVectorFormat configuration:

```jsonc
{
  "name": "geojson",
  // The coordinate mode. It controls the projection to use for coordinates written to the
  // GeoJSON files. "pixel" (default) means we write them as is, "crs" means we just
  // undo the resolution in the Projection so they are in CRS coordinates, and "wgs84"
  // means we always write longitude/latitude. When using "pixel", the GeoJSON will not
  // be readable by GIS tools since it relies on a custom encoding.
  "coordinate_mode": "pixel"
}
```

TileVectorFormat configuration:

```jsonc
{
  "name": "tile",
  // The tile size, default 512.
  "tile_size": 512
}
```


Data Source Specification
-------------------------

The data source specification looks like this:

```jsonc
{
  // The class path of the data source.
  "name": "rslearn.data_sources.gcp_public_data.Sentinel2",
  // The query configuration specifies how items should be matched to windows. It is
  // optional, and the values below are defaults.
  "query_config": {
    // The space mode must be "MOSAIC" (default), "CONTAINS", or "INTERSECTS".
    "space_mode": "MOSAIC",
    // The time mode must be "WITHIN" (default), "BEFORE", or "AFTER".
    "time_mode": "WITHIN",
    // The max matches defaults to 1.
    "max_matches": 1
  },
  // The time offset is optional. It defaults to 0.
  "time_offset": "0d",
  // The duration is optional. It defaults to null.
  "duration": null,
  // The ingest flag is optional, and defaults to true.
  "ingest": true,
  // Data sources may expose additional configuration options. These would also be
  // configured in this section.
  // ...
}
```

### Query Configuration

The query configuration specifies how items should be matched to windows.

For each window, the matching process yields a `list[list[Item]]`. This is a list of
item groups, where each item group corresponds to the items that will be used to create
one materialized piece of raster or vector data.

The space mode defines the spatial matching.

- MOSAIC means that one or more mosaics should be created, combining multiple items
  from the data source as needed to cover the entire window. In this case, each item
  group may include multiple items.
- CONTAINS means that only items that contain the window bounds should be used. In this
  case, each item group consists of exactly one item.
- INTERSECTS means that items that intersect the window bounds can be used. In this
  case, each item group consists of exactly one item.

For raster data, with MOSAIC, multiple items may be combined together to materialize a
raster aligned with the window, while CONTAINS and INTERSECTS means that each
materialized raster should correspond to one item (possibly after cropping and
re-projection).

The time mode defines the temporal matching.

- WITHIN means to use items with time ranges that are contained within the time range
  of the window, but to process them in the order provided by the data source. Note
  that, for most data sources, the item time range is a single point in time.
- BEFORE and AFTER still use items with time ranges that are contained within the time
  range of the window, but they affect the ordering of the items. BEFORE matches items
  in reverse temporal order, starting with items just before the window end time. AFTER
  matches items in temporal order, starting with items just after the window start
  time.

Finally, max matches is the maximum number of item groups that should be created. The
default is 1. For MOSAIC, this means to attempt to create one mosaic covering the
window; zero item groups will be returned only if there are zero items intersecting the
window. For CONTAINS and INTERSECTS, this means to select the first matching item.

If max matches is greater than one, then for MOSAIC, it will attempt to create multiple
mosaics up to that quantity of mosaics. However, it will only start the next mosaic
after the current mosaic fully covers the window. This means that, if there is no item
covering some corner of the window, then even if there are many items redundantly
covering the rest of the window, only one moasaic will be returned.

For CONTAINS and INTERSECTS, it will simply choose up to that many matching items.

Under WITHIN time mode, the order of the items is based on the ordering provided by the
data source. Some data sources provide options to, say, sort items by cloud cover.
Under BEFORE or AFTER time mode, the ordering from the data source is overwritten.

### Time Offset

By default, the time range used for matching is the time range of the window. The time
offset specifies a positive or negative time delta to apply to the window's time range
before matching.

It is parsed by [pytimeparse](https://github.com/wroberts/pytimeparse). For example:

- "30d" means to adjust the window time range 30 days into the future.
- "-30d" means to adjust the window time range 30 days into the past.

The duration of the window time range is not affected.

Then, the data source will look for items based on new time range.

### Duration

The optional duration overrides the duration of the window's time range. The new time
range will have the same start time as the window's start time, but the end time will
be computed by adding the specified duration to that start time.

It is also parsed by pytimeparse. For example, "30d" means to set the duration of the
time range to 30 days.

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


Source-Specific Configuration
-----------------------------

This section details the configuration of each data source.

We also include source-specific recommendations for settings for the `dataset prepare`,
`dataset ingest`, and `dataset materialize` commands below. Unless otherwise noted, it
is generally suggested to use:

```
rslearn dataset prepare --root ... --workers NUM_WORKERS
rslearn dataset ingest --root ... --workers NUM_WORKERS --no-use-initial-job --jobs-per-process 1
rslearn dataset materialize --root ... --workers NUM_WORKERS --no-use-initial-job
```

Replace NUM_WORKERS with a number of workers depending on the available system memory
(may require trial and error).

When using multiple workers, rslearn by default first processes one task in the main
thread before parallelizing the remaining tasks across workers, but
`--no-use-initial-job` disables this functionality. We use the default functionality
for `dataset prepare` since data sources often perform processing, like downloading and
caching an index file, that should not be parallellized.

`--jobs-per-process` indicates how many tasks a worker thread should process before
terminating. We use `--jobs-per-process 1` for `dataset ingest` since there seem to be
some memory leaks in rasterio that crops up for some data sources.

### rslearn.data_sources.aws_landsat.LandsatOliTirs

This data source is for Landsat 8/9 OLI-TIRS imagery on AWS. It uses the usgs-landsat
S3 bucket maintained by USGS. It includes Tier 1/2 scenes but not Real-Time scenes. See
https://aws.amazon.com/marketplace/pp/prodview-ivr4jeq6flk7u for details about the
bucket.

The additional data source configuration looks like this:

```jsonc
{
  // Required cache directory to cache product metadata files. Unless prefixed by a
  // protocol (like "file://..."), it is joined with the dataset path (i.e., specifies
  // a sub-directory within the dataset folder.
  "metadata_cache_dir": "cache/landsat",
  // Sort by this attribute, either null (default, meaning arbitrary ordering) or
  // "cloud_cover".
  "sort_by": null
}
```

Available bands:
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

### rslearn.data_sources.aws_open_data.Naip

This data source is for NAIP imagery on AWS. It uses the naip-source requester pays
bucket maintained by Esri. See https://registry.opendata.aws/naip/ for more
information. AWS credentials must be configured for use with boto3.

The additional data source configuration looks like this:

```jsonc
{
  // Required cache directory to cache index shapefiles. Unless prefixed by a protocol
  // (like "file://..."), it is joined with the dataset path.
  "index_cache_dir": "cache/naip",
  // Whether to build an rtree index to accelerate prepare lookups, default false. It
  // is recommended to set this true when processing more than a few windows.
  "use_rtree_index": false,
  // Limit the search to these states (list of their two-letter codes). This can
  // substantially accelerate lookups when the rtree index is disabled, since by
  // default (null) it has to scan through all of the states.
  // Example: ["wa", "or"]
  "states": null,
  // Limit the search to these years. Like with states, this can speed up lookups when
  // the rtree index is disabled.
  // Example: [2023, 2024]
  "years": null
}
```

Available bands:
- R
- G
- B
- IR

### rslearn.data_sources.aws_open_data.Sentinel2

This data source is for Sentinel-2 L1C and L2A imagery on AWS. It uses the
sentinel-s2-l1c and sentinel-s2-l2a S3 buckets maintained by Sinergise. They state the
data is "added regularly, usually within few hours after they are available on
Copernicus OpenHub".

See https://aws.amazon.com/marketplace/pp/prodview-2ostsvrguftb2 for details about the
buckets. AWS credentials must be configured for use with boto3.

The additional data source configuration looks like this:

```jsonc
{
  // Required modality, either "L1C" or "L2A".
  "modality": "L1C",
  // Required cache directory to cache product metadata files.
  "metadata_cache_dir": "cache/sentinel2",
  // Sort by this attribute, either null (default, meaning arbitrary ordering) or
  // "cloud_cover".
  "sort_by": null,
  // Flag (default false) to harmonize pixel values across different processing
  // baselines (recommended), see
  // https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
  "harmonize": false
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
- B10 (L1C only)
- B11
- B12
- B8A
- R (from TCI asset; derived from B04)
- G (from TCI asset; derived from B03)
- B (from TCI asset; derived from B02)

### rslearn.data_sources.gcp_public_data.Sentinel2

This data source is for Sentinel-2 data on Google Cloud Storage.

Sentinel-2 imagery is available on Google Cloud Storage as part of the Google
Public Cloud Data Program. The images are added with a 1-2 day latency after
becoming available on Copernicus.

See https://cloud.google.com/storage/docs/public-datasets/sentinel-2 for details.

The bucket is public and free so no credentials are needed.

```jsonc
{
  // Required cache directory to cache product metadata files and the optional rtree
  // index.
  "index_cache_dir": "cache/sentinel2",
  // Sort by this attribute, either null (default, meaning arbitrary ordering) or
  // "cloud_cover".
  "sort_by": null,
  // Flag (default true) to build an rtree index to speed up product lookups. This can
  // be set false to avoid length (multiple hours) rtree creation time if you are only
  // using a few windows.
  "use_rtree_index": true,
  // Flag (default false) to harmonize pixel values across different processing
  // baselines (recommended), see
  // https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
  "harmonize": false,
  // When using rtree index, only create it for products within this time range. It
  // defaults to null, meaning to create rtree index for entire time range.
  // Example: ["2024-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00"]
  "rtree_time_range": null,
  // By default, if use_rtree_index is true, the rtree index is stored in the
  // index_cache_dir. Set this to override the path for the rtree index and only use
  // index_cache_dir for the product metadata files.
  "rtree_cache_dir": null
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
- B10
- B11
- B12
- B8A
- R (from TCI asset; derived from B04)
- G (from TCI asset; derived from B03)
- B (from TCI asset; derived from B02)

### rslearn.data_sources.google_earth_engine.GEE

This data source is still experimental.

### rslearn.data_sources.local_files.LocalFiles

This data source supports ingesting data from local raster or vector files. It is
configured by a source directory that should be a flat structure with the raster or
vector files. Raster files must be readable by rasterio. Vector files must be readable
by fiona.

Each source file is treated as a separate item, so for raster files, each file must
contain the full range of bands, and different files should cover different locations.

```jsonc
{
  // Required source directory containing the flat structure of raster or vector files.
  // It is relative to the dataset root, so include a protocol if it is outside.
  // Example: "file:///path/to/files/".
  "src_dir": null
}
```

For raster data, the bands will be named "B1", "B2", and so on depending on the number
of bands in the source files.

The time range of all items is null (infinite).

For this dataset, use `--workers 0` (default) so that processing is done in the main
thread. This is because most of the work is spent initializing the data source, due to
the need for identifying the bounds of all of the local files, and so it is best to
just have this done once rather than once in each worker.

### rslearn.data_sources.openstreetmap.OpenStreetMap

This data source is for ingesting OpenStreetMap data from a PBF file.

An existing local PBF file can be used, or if the provided path doesn't exist, then the
global OSM PBF will be downloaded.

This data source uses a single item. If more windows are added, data in the TileStore
will need to be completely re-computed.

```jsonc
{
  // Required list of PBF filenames to read from.
  // If a single filename is provided and it doesn't exist, the latest planet PBF will
  // be downloaded there.
  "pbf_fnames": ["planet-latest.osm.pbf"],
  // Required file to cache the bounds of the different PBF files.
  "bounds_fname": "bounds.json",
  // Required map of categories to extract from the OSM data.
  // Each category specifies a set of restrictions that extract only a certain type of
  // OSM feature, and convert it to a GeoJSON feature.
  "categories": {
    // The key will be added as a "category" property in the resulting GeoJSON
    // features.
    "aerialway_pylon": {
      // Optional limit on the types of features to match. If set, valid list values
      // are "node", "way", "relation".
      // Example: ["node"] to only match nodes.
      "feature_types": null,
      // Optional tag conditions. For each entry (tag_name, values list), only match
      // OSM features with that tag, and if values list is not empty, only match if the
      // tag value matches one element of the values list.
      // The default is null. The example below will only match OSM features with the
      // "aerialway" tag set to "pylon".
      "tag_conditions": {
        "aerialway": [
          "pylon"
        ]
      },
      // Optional tag properties. This is used to save properties of the OSM feature in
      // the resulting GeoJSON feature. It is a list of [tag name, prop name]. If tag
      // tag name exists on the OSM feature, then it will be populated into the prop
      // name property on the GeoJSON feature.
      // Example: [["aerialway:heating", "aerialway:heating"]]
      "tag_properties": null,
      // Optionally convert the OpenStreetMap feature to the specified geometry type
      // (one of "Point", "LineString", "Polygon"). Otherwise, matching nodes result in
      // Points, matching ways result in LineStrings, and matching relations result in
      // Polygons. Note that nodes cannot be converted to LineString/Polygon.
      "to_geometry": "Point"
    },
  }
}
```

### rslearn.data_sources.planet.Planet

This data source is still experimental.

### rslearn.data_sources.planet_basemap.PlanetBasemap

This data source is still experimental.

### rslearn.data_sources.usgs_landsat.LandsatOliTirs

This data source is for Landsat data from the USGS M2M API.

You can request access at https://m2m.cr.usgs.gov/.

```jsonc
{
  // Required M2M API username.
  "username": null,
  // Required M2M API authentication token.
  "token": null,
  // Sort by this attribute, either null (default, meaning arbitrary ordering) or
  // "cloud_cover".
  "sort_by": null,
}
```

Available bands:
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

### rslearn.data_sources.xyz_tiles.XyzTiles

This data source is for web xyz image tiles (slippy tiles).

These tiles are usually in WebMercator projection, but different CRS can be configured.

```jsonc
{
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
```
