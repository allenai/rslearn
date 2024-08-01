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
2. Ingest data for layers that reference a data source.
3. Import or annotate data in other layers.
4. Train a model that inputs one or more layers and outputs one or more other layers.
5. Use the model to make predictions on new windows.
