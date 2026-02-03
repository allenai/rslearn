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
  "class_path": "rslearn.data_sources.gcp_public_data.Sentinel2",
  // The query configuration specifies how items should be matched to windows. It is
  // optional, and the values below are defaults.
  "query_config": {
    // The space mode must be "MOSAIC" (default), "CONTAINS", "INTERSECTS", or "PER_PERIOD_MOSAIC".
    "space_mode": "MOSAIC",
    // The time mode must be "WITHIN" (default), "BEFORE", or "AFTER".
    "time_mode": "WITHIN",
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
  "ingest": true,
  // Data sources may expose additional configuration options, passed via init_args.
  // class_path and init_args are handled by jsonargparse to instantiate the data
  // source class.
  "init_args": {
    // ...
  }
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
- PER_PERIOD_MOSAIC means to create one mosaic per sub-period of the time range. The
  duration of the sub-periods is controlled by `period_duration`. When using this mode,
  always set `per_period_mosaic_reverse_time_order` to false.

For raster data, with MOSAIC, multiple items may be combined together to materialize a
raster aligned with the window, while CONTAINS and INTERSECTS means that each
materialized raster should correspond to one item (possibly after cropping and
re-projection).

**Compositing with mosaic_compositing_overlaps.**
A mosaic stitches adjacent images into one seamless picture, using one pixel value where
images overlap. To enable compositing (combining overlapping images with mean or median),
set `mosaic_compositing_overlaps` to a value greater than 1. This controls how many
overlapping items should be included in each item group (mosaic). Then set
`compositing_method` in the layer config to "MEAN" or "MEDIAN" to compute the per-pixel
mean or median across the overlapping items.

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
mosaics up to that quantity of mosaics.
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

This data source supports direct materialization: if the "ingest" flag is set false,
then ingestion will be skipped and windows will be directly populated from windowed
reads of the underlying cloud-optimized GeoTIFFs on S3.

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

### rslearn.data_sources.aws_sentinel1.Sentinel1

This data source is for Sentinel-1 GRD imagery on AWS. It uses the sentinel-s1-l1c S3
bucket maintained by Sinergise. See
https://aws.amazon.com/marketplace/pp/prodview-uxrsbvhd35ifw for details about the
bucket.

Although other Sentinel-1 scenes are available, the data source currently only supports
the GRD IW DV scenes (vv+vh bands). It uses the Copernicus API for metadata search
(prepare step).

The additional data source configuration looks like this:

```jsonc
{
  // Optional orbit direction to filter by, either "ASCENDING" or "DESCENDING". The
  // default is to not filter (so both types of scenes are included/mixed).
  "orbit_direction": null
}
```

Available bands:
- vv
- vh

### rslearn.data_sources.aws_sentinel2_element84.Sentinel2

This data source is for Sentinel-2 L2A imagery from the Element 84 Earth Search STAC
API on AWS. It uses the `s3://sentinel-cogs` S3 bucket which provides Cloud-Optimized
GeoTIFFs, enabling direct materialization without ingestion.

See https://aws.amazon.com/marketplace/pp/prodview-ykj5gyumkzlme for details.

The bucket is public and free so no credentials are needed.

```jsonc
{
  // Optional STAC query filter.
  // Example: {"eo:cloud_cover": {"lt": 20}}
  "query": null,
  // Sort by this STAC property, e.g. "eo:cloud_cover".
  "sort_by": null,
  // Whether to sort ascending or descending (default ascending).
  "sort_ascending": true,
  // Optional directory to cache discovered items.
  "cache_dir": null,
  // Flag (default false) to harmonize pixel values across different processing
  // baselines (recommended), see
  // https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
  "harmonize": false,
  // Timeout for requests.
  "timeout": "10s"
}
```

Available bands:
- B01 (uint16, from coastal asset)
- B02 (uint16, from blue asset)
- B03 (uint16, from green asset)
- B04 (uint16, from red asset)
- B05 (uint16, from rededge1 asset)
- B06 (uint16, from rededge2 asset)
- B07 (uint16, from rededge3 asset)
- B08 (uint16, from nir asset)
- B09 (uint16, from nir09 asset)
- B11 (uint16, from swir16 asset)
- B12 (uint16, from swir22 asset)
- B8A (uint16, from nir08 asset)
- R, G, B (uint8, from visual asset)

### rslearn.data_sources.climate_data_store.ERA5Land

Base class for ingesting ERA5 land data from the Copernicus Climate Data Store.

An API key must be provided either in the data source configuration or via the CDSAPI_KEY
environment variable. You can acquire an API key by registering at the Climate Data Store
website (https://cds.climate.copernicus.eu/).

Valid bands are the variable names listed on the CDS dataset page (use the **API request**
tool to check valid values). Note it is necessary to replace "_" with "-" in the variable
names, e.g. `total_precipitation` becomes `total-precipitation`.

**Performance:** Specifying the `bounds` parameter to limit the geographic extent of data
retrieval is highly recommended, especially for hourly data, as downloading global data can
be very slow and resource-intensive.

The additional data source configuration looks like this:

```jsonc
{
  // Optional API key. If not provided in the data source configuration, it must be set
  // via the CDSAPI_KEY environment variable.
  "api_key": null,
  // Optional bounding box as [min_lon, min_lat, max_lon, max_lat]. Recommended to speed
  // up ingestion, especially for hourly data.
  // Example: [-122.4, 47.6, -122.3, 47.7]
  "bounds": null
}
```

### rslearn.data_sources.climate_data_store.ERA5LandMonthlyMeans

This data source is for ingesting ERA5 land monthly averaged data from the Copernicus
Climate Data Store. This corresponds to the `reanalysis-era5-land-monthly-means` dataset.

See rslearn.data_sources.climate_data_store.ERA5Land above for common configuration
and usage information.

Valid bands: See the [CDS dataset page](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means?tab=download).

### rslearn.data_sources.climate_data_store.ERA5LandHourly

This data source is for ingesting ERA5 land hourly data from the Copernicus Climate Data
Store. This corresponds to the `reanalysis-era5-land` dataset.

See rslearn.data_sources.climate_data_store.ERA5Land above for common configuration
and usage information. **Note:** The `bounds` parameter is especially important for hourly
data to avoid very slow global downloads.

Valid bands: See the [CDS dataset page](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download).

### rslearn.data_sources.copernicus.Copernicus

This data source is for images from the ESA Copernicus OData API. See
https://documentation.dataspace.copernicus.eu/APIs/OData.html for details about the API
and how to get an access token.

The additional data source configuration looks like this:

```jsonc
{
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
```

### rslearn.data_sources.copernicus.Sentinel2

This data source is for Sentinel-2 images from the ESA Copernicus OData API.

The additional data source configuration looks like this:

```jsonc
{
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
- TCI
- B10 (L1C only)
- AOT (L2A only)
- WVP (L2A only)
- SCL (L2A only)

### rslearn.data_sources.copernicus.Sentinel1

This data source is for Sentinel-1 images from the ESA Copernicus OData API. Currently
only IW GRDH VV+VH products are supported, even though all Sentinel-1 scenes are
available in the data source.

The additional data source configuration looks like this:

```jsonc
{
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
```

### rslearn.data_sources.eurocrops.EuroCrops

This data source is for EuroCrops vector data (v11).

See https://zenodo.org/records/14094196 for details.

While the source data is split into country-level files, this data source uses one item
per year for simplicity. So each item corresponds to all of the country-level files for
that year.

Note that the RO_ny.zip file is not used.

There is no data-source-specific configuration.

The vector features should have `EC_hcat_c` and `EC_hcat_n` properties indicating the
HCAT category code and name respectively.

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
  // be set false to avoid lengthy (multiple hours) rtree creation time if you are only
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

### rslearn.data_sources.hf_srtm.SRTM

Elevation data from the Shuttle Radar Topography Mission (SRTM), served from the AI2
Hugging Face mirror at https://huggingface.co/datasets/allenai/srtm-global-void-filled.

The data is split into 1x1-degree tiles. SRTM1 (1 arc-second, ~30m resolution) is
available for some regions (primarily US territories), while SRTM3 (3 arc-second, ~90m
resolution) is available globally. By default, SRTM1 is preferred when available for
higher resolution.

No credentials are needed.

```jsonc
{
  // Timeout for requests.
  "timeout": "10s",
  // Optional directory to cache the file list.
  "cache_dir": null,
  // If true, always use 3 arc-second (SRTM3) data even when 1 arc-second (SRTM1) is
  // available. Defaults to false, which prefers SRTM1 for higher resolution.
  "always_use_3arcsecond": false
}
```

The data source should be configured with a single band set containing a single band.
The band name can be set arbitrarily, but "dem" or "srtm" is suggested. The data type
should be int16 to match the source data.

Items from this data source do not come with a time range.

### rslearn.data_sources.google_earth_engine.GEE

This data source is for ingesting images from Google Earth Engine (GEE).

It must be configured with the name of an ee.ImageCollection on GEE. Each ee.Image in
the ee.ImageCollection is treated as a different data source item. A Cloud Storage
bucket is also required to store the intermediate outputs from GEE export jobs.

During the prepare stage, it will first export the metadata (geometry and time range)
of all ee.Image objects in the ee.ImageCollection. Then it will use this to build an
rtree from which prepare requests can be satisfied.

During the ingest stage, it will start export jobs to export images to the bucket. Each
worker will start one job and poll until it finishes before proceeding onto the next
ee.Image to export. After the export finishes, the resulting GeoTIFF(s) are read and
processed into the tile store. Note that export jobs can take several minutes to
complete depending on the size of the image.

This data source does support direct materialization, which can greatly speed up
materialization for sparse windows. Whereas exporting a 10Kx10K image make take 5000
EECU-seconds (and potentially several minutes), exporting a 256x256 image should take
only a few seconds.

```jsonc
{
  // Required name of the ee.ImageCollection, e.g. "COPERNICUS/S1_GRD".
  "collection_name": "COPERNICUS/S1_GRD",
  // Required name of the GCS bucket to use to store intermediate outputs from export
  // jobs. You could set up lifecycle rules on this bucket to delete outputs after 1
  // day.
  "gcs_bucket_name": "...",
  // Required service account name.
  "service_account_name": "...",
  // Required path to a local file containing the service account credentials.
  "service_account_credentials": "/etc/credentials/gee_credentials.json",
  // Required directory to store rtree index over the exported ee.Image metadata.
  "index_cache_dir": "cache/gee",
  // Optional filters to aply on the ee.ImageCollection. See Sentinel-1 example below.
  // Currently only equality filters are supported.
  "filters": null
}
```

The available bands depends on the chosen ee.ImageCollection. Here is an example layer
configuration for Sentinel-1. The filters match only ee.Image objects where the
"transmitterReceiverPolarisation" attribute is ["VV", "VH"] and the "instrumentMode"
attribute is "IW".

```json
{
  "sentinel1": {
    "band_sets": [
      {
        "bands": [
          "VV",
          "VH"
        ],
        "dtype": "uint16",
        "format": "geotiff"
      }
    ],
    "data_source": {
      "class_path": "rslearn.data_sources.google_earth_engine.GEE",
      "init_args": {
        "collection_name": "COPERNICUS/S1_GRD",
        "dtype": "float32",
        "filters": [
          [
            "transmitterReceiverPolarisation",
            [
              "VV",
              "VH"
            ]
          ],
          [
            "instrumentMode",
            "IW"
          ]
        ],
        "gcs_bucket_name": "YOUR_BUCKET_NAME",
        "index_fname": "cache/sentinel1_index",
        "service_account_credentials": "/etc/credentials/gee_credentials.json",
        "service_account_name": "YOUR_SERVICE_ACCOUNT_NAME"
      },
      "query_config": {
        "max_matches": 1
      }
    },
    "type": "raster"
  }
}
```

### rslearn.data_sources.google_earth_engine.GoogleSatelliteEmbeddings

This data source is for Google Satellite Embeddings (AlphaEarth Embeddings) from Google
Earth Engine. The embedding values are stored as unsigned 16-bit integers from 0 to
16383, computed by multiplying the original [-1, 1] floating point values by 8192 and
adding 8192.

```jsonc
{
  // See rslearn.data_sources.google_earth_engine.GEE for details about these
  // required configuration options.
  "gcs_bucket_name": "...",
  "service_account_name": "...",
  "service_account_credentials": "/etc/credentials/gee_credentials.json",
  "index_cache_dir": "cache/gee"
}
```

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

### rslearn.data_sources.planetary_computer.PlanetaryComputer

This data source is for raster data from Microsoft Planetary Computer. See their
[Data Catalog](https://planetarycomputer.microsoft.com/catalog).

This data source supports direct materialization: if the "ingest" flag is set false,
then ingestion will be skipped and windows will be directly populated from windowed
reads of the underlying cloud-optimized GeoTIFFs on Azure Blob Storage.

```jsonc
{
  // Required collection name, e.g. "landsat-c2-l2" or "modis-17A2HGF-061".
  "collection_name": null,
  // Required map from asset name to list of bands in the asset to download.
  // You may need to perform a STAC search to see what the asset names are.
  // Example: {"B8A": ["B8A"], "visual": ["R", "G", "B"]}
  "asset_bands": null,
  // Include this query argument for STAC searches.
  // Example: {"sar:instrument_mode": {"eq": "IW"}}
  "query": null,
  // Sort by this property in the STAC items.
  // Example: "eo:cloud_cover"
  "sort_by": null,
  // Whether to sort ascending or descending (default ascending).
  "sort_ascending": true,
  // Timeout for requests.
  "timeout_seconds": 10,
}
```

### rslearn.data_sources.planetary_computer.Sentinel1

Sentinel-1 radiometrically-terrain-corrected data on Microsoft Planetary Computer.
Direct materialization is supported.

It automatically determines the bands to download from the band sets, so all parameters
are optional. The band names are "hh", "hv", "vv", and "vh" depending on the scene.

```jsonc
{
  // See rslearn.data_sources.planetary_computer.PlanetaryComputer.
  "query": null,
  "sort_by": null,
  "sort_ascending": true,
  "timeout_seconds": 10,
}
```

### rslearn.data_sources.planetary_computer.Sentinel2

Sentinel-2 L2A data on Microsoft Planetary Computer. Direct materialization is
supported.

The bands to download are determined from the band sets.

```jsonc
{
  // Flag (default false) to harmonize pixel values across different processing
  // baselines (recommended), see
  // https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
  "harmonize": false
  // See rslearn.data_sources.planetary_computer.PlanetaryComputer.
  "query": null,
  "sort_by": null,
  "sort_ascending": true,
  "timeout_seconds": 10,
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
- visual

Note that B10 is not present in L2A.

### rslearn.data_sources.planetary_computer.LandsatC2L2

Landsat 8/9 Collection 2 Level-2 data on Microsoft Planetary Computer. Direct
materialization is supported.

See the dataset page: https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2

This data source uses Landsat-style band identifiers (for compatibility with
`rslearn.data_sources.aws_landsat.LandsatOliTirs`). It defaults to:
- B1
- B2
- B3
- B4
- B5
- B6
- B7
- B10

Band mapping (rslearn band → STAC `common_name` / STAC `eo:bands[].name`):
- B1 → coastal / OLI_B1
- B2 → blue / OLI_B2
- B3 → green / OLI_B3
- B4 → red / OLI_B4
- B5 → nir08 / OLI_B5
- B6 → swir16 / OLI_B6
- B7 → swir22 / OLI_B7
- B10 → lwir11 / TIRS_B10

Note: this is Level-2 data rather than Level-1, so it does not provide Level-1 bands
like B8 (panchromatic), B9 (cirrus / OLI_B9), and B11 (thermal / TIRS_B11). If you need those, use the slower
`rslearn.data_sources.aws_landsat.LandsatOliTirs` data source.

```jsonc
{
  // Optional list of bands to expose. Values can be Landsat band identifiers
  // ("B1", "B2", ..., "B10") or the STAC common names / STAC `eo:bands[].name`
  // aliases listed above (e.g. "red" or "OLI_B4").
  "band_names": null,
  // The optional STAC query filter to use. Defaults to selecting Landsat 8 and 9. For example, set
  // to {"platform": ["landsat-8"]} to use Landsat 8 only.
  "query": {"platform": ["landsat-8", "landsat-9"]},
  // See rslearn.data_sources.planetary_computer.PlanetaryComputer.
  "sort_by": null,
  "sort_ascending": true,
  "timeout_seconds": 10
}
```

### rslearn.data_sources.planetary_computer.Naip

NAIP imagery on Microsoft Planetary Computer. Direct materialization is supported.

This data source uses the Planetary Computer `naip` collection, and reads the `image`
asset which contains four bands: `R`, `G`, `B`, `NIR`.

Note: NAIP provides a single 4-band GeoTIFF asset (`image`). Internally, rslearn will
still ingest/read this full 4-band asset, but you can configure your raster layer band
set to materialize any subset of `["R", "G", "B", "NIR"]` (for example `["NIR"]`).
If you need a different asset/band mapping, use
`rslearn.data_sources.planetary_computer.PlanetaryComputer` directly with a custom
`asset_bands` mapping.

```jsonc
{
  // See rslearn.data_sources.planetary_computer.PlanetaryComputer.
  "query": null,
  "sort_by": null,
  "sort_ascending": true,
  "timeout_seconds": 10
}
```

### rslearn.data_sources.planetary_computer.CopDemGlo30

Copernicus DEM GLO-30 (30m) data on Microsoft Planetary Computer. Direct materialization
is supported.

This is a "static" dataset (no meaningful temporal coverage), so it ignores window time
ranges when searching and matching STAC items.

The Copernicus DEM items expose the DEM GeoTIFF as the `data` asset, and this data
source maps it to a single band.

```jsonc
{
  // Optional band name to use if the layer config is missing from context (default "DEM").
  "band_name": "DEM",
  // See rslearn.data_sources.planetary_computer.PlanetaryComputer.
  "timeout_seconds": 10
}
```

### rslearn.data_sources.usda_cdl.CDL

This data source is for the USDA Cropland Data Layer.

The GeoTIFF data will be downloaded from the USDA website. See
https://www.nass.usda.gov/Research_and_Science/Cropland/SARS1a.php for details about
the data.

There is one GeoTIFF item per year from 2008. Each GeoTIFF spans the entire continental
US, and has a single band.

```jsonc
{
  // Optional timeout for HTTP requests.
  "timeout_seconds": 10
}
```

The data source yields one band, and the name will match whatever is configured in the
band set. It should be uint8.

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

### rslearn.data_sources.soilgrids.SoilGrids

This data source provides access to [ISRIC SoilGrids](https://www.isric.org/explore/soilgrids)
via the public WCS endpoints (e.g. `https://maps.isric.org/mapserv?map=/map/clay.map`).

This source is intended for **direct materialization** (set `"ingest": false` in the
layer's `data_source` config), since data is fetched on-demand per window.

Example (clay, user-provided WCS subset parameters):

```jsonc
{
  "service_id": "clay",
  "coverage_id": "clay_0-5cm_mean",
  // Optional request CRS, defaults to EPSG:3857. You can specify either "EPSG:3857"
  // or the URN form "urn:ogc:def:crs:EPSG::3857".
  // "crs": "EPSG:3857"
}
```

If `"width"`/`"height"` and `"resx"`/`"resy"` are omitted, rslearn will default to
requesting at ~250 m resolution in the request CRS and then reprojecting to the window
grid. For EPSG:4326 requests, SoilGrids requires `"width"`/`"height"` so rslearn will
default those to the window pixel size.

Available bands:
- B1 (float32 recommended; scale/offset applied; set `nodata_vals` to `-32768`)

### rslearn.data_sources.worldcereal.WorldCereal

This data source is for the ESA WorldCereal 2021 agricultural land cover map. For
details about the land cover map, see https://esa-worldcereal.org/en.

This data source will download and extract all of the WorldCereal GeoTIFFs to a local
directory. Since different regions are covered with different bands, the data source is
designed to only be configured with one band per layer; to materialize multiple bands,
repeat the data source across multiple layers (with different bands).

```jsonc
{
  // Required local path to extract the WorldCereal GeoTIFF files. For high performance,
  // this should be a local directory; if the dataset is remote, prefix with a protocol
  // ("file://") to use a local directory.
  "worldcereal_dir": "cache/worldcereal"
}
```

Available bands (specify one per layer, with a single-band band set):
- tc-annual_temporarycrops_confidence
- tc-annual_temporarycrops_classification
- tc-maize-main_irrigation_confidence
- tc-maize-main_irrigation_classification
- tc-maize-main_maize_confidence
- tc-maize-main_maize_classification
- tc-maize-second_irrigation_confidence
- tc-maize-second_irrigation_classification
- tc-maize-second_maize_confidence
- tc-maize-second_maize_classification
- tc-springcereals_springcereals_confidence
- tc-springcereals_springcereals_classification
- tc-wintercereals_irrigation_confidence
- tc-wintercereals_irrigation_classification
- tc-wintercereals_wintercereals_confidence
- tc-wintercereals_wintercereals_classification

### rslearn.data_sources.worldcover.WorldCover

This data source is for the ESA WorldCover 2021 land cover map.

For details about the land cover map, see https://worldcover2021.esa.int/.

This data source downloads the 18 zip files that contain the map. They are then
extracted, yielding 2,651 GeoTIFF files. These are then used with
`rslearn.data_sources.local_files.LocalFiles` to implement the data source.

```jsonc
{
  // Required local path to store the downloaded zip files and extracted GeoTIFFs.
  "worldcover_dir": "cache/worldcover"
}
```

Available bands:
- B1 (uint8)

### rslearn.data_sources.worldpop.WorldPop

This data source is for world population data from worldpop.org.

Currently, this only supports the WorldPop Constrained 2020 100 m Resolution dataset.
See https://hub.worldpop.org/project/categories?id=3 for details.

The data is split by country. We implement with LocalFiles data source for simplicity,
but it means that all of the data must be downloaded first.

```jsonc
{
  // Required local path to store the downoladed WorldPop data.
  "worldpop_dir": "cache/worldpop"
}
```

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
