## rslearn Dataset Format

This document details the storage format for rslearn datasets. For a higher-level
introduction, see [CoreConcepts](CoreConcepts.md).

rslearn datasets are stored under a directory tree, which can be on the local
system or a remote filesystem compatible with the fsspec library. The structure is like
this:

```
config.json
windows/
  group1/
    window_name1/
      metadata.json
      items.json
      layers/
        unitemporal_raster_layer1/
          completed
          B01_B02_B03_B04/
            geotiff.tif
        unitemporal_raster_layer2/
          ...
        vector_layer1/
          completed
          data.geojson
        multitemporal_raster_layer1/
          ...
        multitemporal_raster_layer1.1/
          ...
        multitemporal_raster_layer1.2/
          ...
    window_name2/
      ...
  group2/
    ...
```

The dataset configuration file is stored in `config.json` directly under the dataset
directory tree. This file specifies the layers in the dataset, and for layers that are
populated automatically from data sources, it also specifies configuration options for
the data source. See [DatasetConfig](DatasetConfig.md) for more details.

The data is organized by window. The first level under `windows/` consists of one
subfolder for each window group that appears in the dataset. Window groups can be used
arbitrarily by the user to group together different subsets of the dataset, and many
datasets contain a single group. The second level (under the group subfolders) consists
of the per-window folders.

### `metadata.json`

In the per-window folders, `metadata.json` specifies the spatial bounds and time range
of the window. Here is an example:

```json
{
  "projection": {
    "crs": "EPSG:32612",
    "x_resolution": 10,
    "y_resolution": -10
  },
  "bounds": [
    35855,
    -383001,
    35887,
    -382969
  ],
  "time_range": [
    "2020-09-08T00:00:00+00:00",
    "2020-10-08T00:00:00+00:00"
  ],
  "options": {
    "split": "train"
  }
}
```

The `projection` key specifies the coordinate reference system and resolution; here,
the resolution is 10 m/pixel. It is typical for the `y_resolution` to be negative so
that north is up in the image and south is down. The `time_range` key specifies the
time range of the window.

The `bounds` key specifies the bounds of the window in pixel coordinates. This can be
multiplied by the resolution to get the bounds in projection units. Here, the bounds in
EPSG:32612 projection units (meters) are `(358550, 3830010, 358870, 3829690)`.

The `options` key stores arbitrary user-specified key-value pairs. Oftentimes, windows
are assigned to training and validation splits via an option here, although this can
also be achieved by using different window groups for train/val.

### `items.json`

In the per-window folders, `items.json` is populated when the dataset is prepared, and
specifies which items in the data source matched with the window. Here is an example:

```json
[
  {
    "layer_name": "sentinel2",
    "serialized_item_groups": [
      [
        {
          "name": "S2B_MSIL2A_20200904T180919_R084_T12SUD_20201027T145542",
          "geometry": {
            "projection": {
              "crs": "EPSG:4326",
              "x_resolution": 1,
              "y_resolution": 1
            },
            "shp": "POLYGON ((-113.19754165 35.22312494, -111.99134995 35.23901758, -111.97959195 34.2490077, -113.1714975 34.23368737, -113.19754165 35.22312494))",
            "time_range": [
              "2020-09-04T18:09:19.024000+00:00",
              "2020-09-04T18:09:19.024000+00:00"
            ]
          }
        }
      ],
    ]
  }
]
```

The file contains a JSON list with one dict for each layer that has been prepared (this
only includes layers that are populated from data sources). The dict specifies the
layer name, along with the item groups that matched. `item_groups` is a serialized
`list[list[Item]]`, where each sub-list is one group of data source items that should
be merged/mosaicked together to form one raster or vector file for the window. If there
are multiple sub-lists, it typically corresponds to multi-temporal data, and each one
will result in a different raster or vector file after the data is materialized.

Materialization will use the first item group in `item_groups` to populate
`layers/LAYER_NAME`, the second to populate `layers/LAYER_NAME.1`, and so on.

For example, consider this query configuration for a data source
(see [DatasetConfig](DatasetConfig.md) for details):

```json
  "query_config": {
    "space_mode": "INTERSECTS",
    "max_matches": 3
  }
```

Since the space mode is `INTERSECTS`, the window is matched with individual items that
intersect the window spatiotemporally, so the sub-lists in `serialized_item_groups`
will consist of exactly one item each. There will be up to 3 sub-lists depending on
whether there are enough matching items.

Each serialized item includes its name and geometry. The geometry consists of the
projection, time range, and WKT shape in pixel coordinates. Here, the projection
resolution is 1 projection unit / pixel, and the CRS is EPSG:4326 (WGS84), so the pixel
coordinates are just longitudes and latitudes. Data sources may serialize other
information with the item as well.

### layers folder

`layers/` is populated during data materialization, when data previously ingested into
the tile store is cropped and aligned with the window (or some data sources support
directly materializing data from the data source without ingestion).

The layer directories are named `[LAYER_NAME].[GROUP_INDEX]`, where `GROUP_INDEX` is
the index of the item group in the list of item groups corresponding to `items.json`.
For the first group (GROUP_INDEX=0), the subfolder is instead named just with the layer
name.

Materialization produces a sentinel `completed` file once all item groups for the layer
are populated. This is checked during training to determine which windows have the
required layers available.

The other contents within the layer directory depend on the raster and vector format
specified in the dataset configuration. For raster layers, there will be one raster per
band set, and so there are subfolders called raster directories named based on the
bands; the default raster format will produce a GeoTIFF within the raster directory.
The default vector format will save the vector features in a `data.geojson` file.
