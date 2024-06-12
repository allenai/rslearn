rslearn will be a tool for developing remote sensing datasets and models.

rslearn helps with:

1. Developing remote sensing datasets, starting with identifying spatiotemporal
   windows (roughly equivalent to training examples) that should be annotated.
2. Importing raster and vector data from various online data sources, as well
   as from local files, into the dataset.
3. Annotating new categories of vector data (like points, polygons, and
   classification labels) using integrated web-based labeling apps.
2. Training models on these datasets.
3. Applying models on new inputs, including both inputs formatted into rslearn
   datasets, and externally obtained raster and vector input files.

For tasks that do not require customized workflows, users will primarily
interact with rslearn as a command-line tool, using its CLI to create datasets,
manage annotation, train models, and apply models.

However, for some use cases, only pieces of rslearn's functionality will be
relevant. Other use cases may have custom requirements that require
incorporating task-specific code to address. These use cases will still be able
to use the pieces of rslearn's functionality that are relevant to them by
accessing rslearn as a Python library.

Quick links:
- [FeatureList](FeatureList.md) enumerates many features that rslearn should
  support.
- [Architecture](Architecture.md) documents the structure of the Python library.
- [CLI](CLI.md) documents the command-line interface.
- [Workflows](workflows/): detailed examples of how rslearn will be able to be
  used for specific tasks.

Datasets
========

The first phase of development will focus on developing remote sensing
datasets. We detail how dataset development will work in rslearn in this
section.


Core Concepts
-------------

An rslearn *dataset* consists of a set of raster and vector layers, along with
spatiotemporal windows where data in those layers is available.

A *window* roughly corresponds to a training or test example, and is the unit
for both data annotation and model training. Essentially, a window is a
geographic area, coupled with a time range, over which we want a model to make
some prediction.

Each *layer* stores a certain kind of data. As a simple example, a dataset may
include one raster layer storing Sentinel-2 images and one vector layer where
solar farm polygons are labeled.

Layers may be:

1. Retrieved layers, where data is retrieved from an online data source, like
   Sentinel-2 images from the ESA Copernicus API, or OpenStreetMap building
   data from a global PBF URL.
2. Non-retrieved layers, where data may be imported from local geospatial files
   or labeled via a labeling app.

Some layers may exclusively be used as reference images for labeling, while
other layers may serve as model inputs or targets.

A *data source* provides an interface to an online API that can be used to
populate retrieved layers. A data source consists of items that can be
individually downloaded, like a Sentinel-2 scene or global OpenStreetMap PBF
file.


Workflow
--------

Dataset development with the rslearn CLI follows this workflow:

1. `rslearn dataset create`: create a new, empty dataset.
2. `rslearn dataset add_windows`: add spatiotemporal windows to the dataset.
   This step defines the locations and time ranges where we will retrieve,
   import, or label data.
3. `rslearn dataset prepare`: for each configured retrieved layer, identify
   items from the data source that match with windows in the dataset.
4. `rslearn dataset materialize`: for each configured retrieved layer, download
   the relevant items and incorporate them into the dataset.
5. `rslearn annotate toolname`: launch an integrated annotation tool and label
   data.


Creating a Dataset
------------------

An rslearn dataset is simply a folder on disk that has a specific format.
(Alternative storage interfaces can support storing the dataset on, say, an S3
bucket instead of on the local filesystem.)

The format is like this:

```
dataset_root/
   config.json
   layers/
    layer1_name.json
    layer2_name.json
    ...
   windows/
      group1_name/
         window1_name/
            metadata.json
            items.json
            layers/
               layer1_name/
                  0_0_tci.png
               layer2_name/
                  0_0_polygons.json
               ...
         ...
      ...
```

All rslearn datasets share the same format, and rslearn is not designed to
directly work with pre-existing datasets. Instead, pre-existing datasets must
either be reformatted into the rslearn dataset format, or retrieved through the
data source API.

`rslearn dataset create --ds_root /path/to/dataset_root/` simply creates a new
folder with a `config.json` file. The user must modify this file to configure
their new dataset. An example configuration file is:

```json
{
  "layers": {
    "sentinel2": {
      "type": "raster",
      "band_sets": [{
        "bands": ["R", "G", "B"],
        "dtype": "uint8",
        "format": {"name": "single_image", "format": "png"},
        "remap": {
          "name": "linear",
          "src": [0, 5000],
          "dst": [0, 255]
        }
      }, {
        "bands": ["B05"],
        "dtype": "uint16",
        "format": {"name": "geotiff"}
      }],
      "data_source": {
        "name": "rslearn.data_sources.aws_open_data.Sentinel2",
        "modality": "L1C",
        "metadata_cache_dir": "cache/sentinel2_metadata/",
        "max_time_delta": 90
      },
      "query_config": {
        "max_matches": 3,
        "space_mode": "mosaic",
        "time_mode": "within"
      }
    },
    "polygons": {
      "type": "polygon",
      "categories": [
        "Solar Farm",
      ],
    }
  },
  "tile_store": {
     "name": "file",
	 "root_dir": "tiles"
  }
}
```

This file specifies two layers:

1. A retrieved layer consisting of Sentinel-2 images retrieved from an AWS S3
   bucket.
2. A non-retrieved layer consisting of polygons with one category.

More details on the dataset configuration file format are provided at
[DatasetConfig](DatasetConfig.md).


Adding Windows
--------------

Each spatiotemporal window specifies a coordinate reference system (CRS),
resolution, bounding box, and time range.

A window is a folder in the dataset like `windows/group1/window1/`. The
`metadata.json` file specifies information about the window:

```json
{
  "name": "window1",
  "group": "group1",
  "crs": "EPSG:3857",
  "resolution": 10,
  "bounds": [631808, 2907136, 632320, 2907648],
  "time_range": ["2019-01-01T00:00:00+00:00", "2022-01-01T00:00:00+00:00"],
}
```

The time range can be null but generally would be set.

Windows can be added to the dataset in several ways:

1. Using `rslearn dataset add_windows --fname blah.geojson ...`, windows can
   be created based on features in a feature file readable by fiona. Users can
   opt to have the windows correspond to the feature bounds, or to create at
   cells intersecting the feature along a grid of cell size specified by
   `--size X`.
2. Using `rslearn dataset add_windows --box lon1,lat1,lon2,lat2`. As above, the
   box can directly specify the window bounds or windows can be created along a
   grid.
3. Users can also programatically create windows, or manually create the window
   folders and `metadata.json` file.


Projections
-----------

A projection in rslearn consists of a CRS and a resolution.

The CRS specifies how coordinates map to locations on the planet. Examples are
WGS-84 (EPSG:4326), WebMercator (EPSG:3857), and the various UTM zones.

Coordinates in a CRS are in projection units, which are often measured in
degrees or meters. The resolution specifies the number of projection units per
pixel.

For example, in an appropriate UTM zone, Sentinel-2 10 m/pixel bands can be
stored at 10 projection units / pixel (since UTM uses meter coordinates). But
users could define spatiotemporal windows with different resolutions, and the
image data will be stored accordingly for each window.

Data sources will re-project data from the native projections of items in the
data source to the projections specified in windows in the dataset.

Within a dataset, though, all data is stored with windows in the pixel
coordinate system of the window. Thus, labeling apps and models don't need to
consider the projection since they can just operate on these pixel coordinates.


Data Sources
------------

A data source is a set of raster or vector items that can be automatically
retrieved to populate the retrieved layers in a dataset.

Data sources can consist of:

1. Many rasters/vectors that each span a small spatial extent and are captured
   at a point in time or a short time range (e.g. Sentinel-2 images).
2. Multiple mosaic rasters/vectors that span the world and have a longer time
   range. (e.g. monthly Satlas Sentinel-2 mosaics).
3. A single raster/vector that spans the world with undefined time range (e.g.
   a global aerial image mosaic or OpenStreetMap).

These are all represented in the same way: items consist of a time range (which
can be null), a projection, and a spatial extent. Rasters additionally specify
bands along with a resolution for each band.

Retrieval of data happens in three phases:

1. Prepare: lookup items in the data source that match with windows in the
   dataset.
2. Ingest: retrieve relevant parts of matched items, warp them to the
   projection of matching windows, split up the result into grid cells (tiles)
   based on the global pixel coordinates, and save the tiles into a tile store.
3. Materialize: bring data from the tile store into the dataset. In the tile
   store, the data is stored in global pixel units. In the dataset, the data is
   stored with respect to each window, and stored in pixel coordinates relative
   to the top-left of the window.

So, during ingestion, items are essentially re-projected and then saved as-is
in the tile store (although some data sources may only retrieve parts of items,
e.g. if the item is a global aerial image mosaic).

During materialization, besides slicing up the data into pieces that correspond
to windows, several additional things can happen, like creating mosaics out of
multiple items associated with that layer, and storing different combinations
of bands together.


Prepare
-------

`rslearn dataset prepare` looks up items in retrieved layers configured in the
dataset that match with dataset windows.

Data sources provide a `get_items(geometries, query_config)` function to
perform the lookup. The arguments specify the spatiotemporal windows, along
with a query configuration, which contains common configuration options shared
across all retrieved layers. Specifically, the query configuration includes:

- `time_mode`: how to match items to windows temporally. Valid values are
  `before`, `within`, `nearest`, and `after`. `before`, `within`, and `after`
  indicate that items should be strictly before, within, or after the window
  time range, while `nearest` looks for items that are closest in time.
- `space_mode`: how to match items to windows spatially. Valid values are
  `contains`, `intersects`, and `mosaic`. `contains` means the item should
  contain the entire window, `intersects` means the item just needs to
  intersect some part of the window, and `mosaic` means to collect groups of
  items that together contain the entire window.
- `max_matches`: how many items to display from this dataset, default 1. For
  mosaic, this many mosaics (groups of items) will be created.

The items identified are stored in `window_root/items.json`.


Ingest
------

`rslearn dataset ingest` retrieves items, re-projects them, and stores tiles of
the items into the tile store.

The tile store can be shared across multiple datasets, and local path, S3
bucket, and other implementations are available.

Ingestion (re-project item and split up into tiles) and materialization (slice
up re-projected item into pieces that correspond to each window in the dataset)
are separated mainly because mosaic and band remixing, if enabled, are easier
to do in a second phase after data has been split up into tiles.

Data sources can opt to directly expose an interface to materialize portions of
items, making ingestion unnecessary. This is useful if the data source has
cloud-optimized GeoTIFFs or XYZ tiles that support random access. In this case
the data source can be marked with `"ingest": false` and it will be directly
materialized from the data source rather than materializing from the tile
store.


Materialize
-----------

`rslearn dataset materialize` materializes ingested tiles into the dataset.

For a retrieved layer `layer1`, in each window directory `window_root/`, the
data will be saved under `window_root/layers/layer1/`. If the layer has
multiple sub-layers due to setting `max_matches > 1`, then after the first
sub-layer it is `window_root/layers/layer1.1/`, `layer1.2`, and so on.

By default, vector data is stored as GeoJSON and raster data as GeoTIFF. The
GeoTIFFs have a small block size to support random access in case the user is
training on patches that are much smaller than the window sizes.


Annotation
----------

TODO


Models
======

TODO

6. `rslearn model create` (TODO): create a new model. The model configuration
   file must be edited to specify options like which dataset(s) to use, which
   layers in each dataset to input, which layers to serve as targets, the model
   architecture, etc.
7. `rslearn model fit`: train the model.
8. `rslearn model test/predict`: apply the model on new spatiotemporal windows.
