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

Most projects that use rslearn for end-to-end dataset and model development follow a
workflow like this:

1. Create a dataset and populate windows.
2. Materialize data for layers that reference a data source.
3. Import or annotate data in other layers.
4. Train a model that inputs one or more layers and outputs one or more other layers.
5. Use the model to make predictions on new windows.


Data Sources
------------

A core feature of rslearn is its ability to materialize aligned raster and vector
data across many built-in data sources.

Data sources share a unified API. A data source is a collection of items that generally
can be individually downloaded. For example, in the Sentinel-2 data source, an item
corresponds to a Sentinel-2 scene.

When a dataset is configured with one or more layers that have data sources defined
(see [DatasetConfig](DatasetConfig.md)), rslearn can automatically populate those
layers with information from items in the data source that intersect spatially and
temporally with each window in the dataset. This data materialization process takes
place in three steps:

1. Prepare: identify items in the data source that correspond to windows in the
   dataset.
2. Ingest: download those items to the dataset's tile store.
3. Materialize: crop and re-project the items relevant to each window from the tile
   store as needed to align them with the window.

### Prepare

The first step is to match items in the data source with each window in the dataset.
The output of matching is a list of *item groups* for each window, where each group
specifies a different list of items that should be composited to form a different
vector or raster file for that window.

There are a number of options in the [DataSourceConfig](dataset_config/DataSourceConfig.md) that can
control how this matching is performed. The default is to create one mosaic that covers
the window's spatial extent. In this case, we iterate over items that intersect the
window spatiotemporally, and continue adding items until the window is spatially
covered, skipping items that add zero additional spatial coverage.

### Ingest

The second step is to download the items (across all windows in the dataset) into the
configured tile store for the dataset. The items are converted to formats (e.g. GeoTIFF
for raster data or tiled set of GeoJSONs for vector data) that support random access to
enable fast materialization in the next step.

Note that this operation is parallelized over data source items, whereas prepare and
materialize are parallelized over dataset windows.

### Materialize

The third step is to crop, re-project, and mosaic the items to extract portions aligned
with the windows. For raster data, this means the source GeoTIFFs are merged and
cropped to correspond to the projection and bounds of the window. For vector data,
features are concatenated across items in the same group, and then features that do not
intersect the window bounds are filtered.
