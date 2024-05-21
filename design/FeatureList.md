This enumerates many features that we would like to ideally support. These may
not be supported in the first version and we may decide to drop many features
entirely in favor of simplicity.


Adding Windows
--------------

- Creating windows along a grid of a certain size corresponding to each feature
  in a shapefile, with a time range specified by `--start_time` and `--end_time`.
- Creating windows without a time range.
- Creating windows of a certain size centered at the center of each feature.
- Creating windows of variable size corresponding to the bounds of each
  feature.
- Specifying the spatial extent via the `--box` option instead of shapefile.
- Creating windows in WebMercator projection where the resolution (projection
  units per pixel) is specified with a `--zoom` option.
- Creating windows in UTM/UPS projection where the resolution is specified but
  the projection is automatically determined based on the position of each
  feature.
- Creating windows where the time range is specified in the shapefile via
  property names `--start_time_property` and `--end_time_property`.
- Only creating windows that are disjoint from existing windows in the dataset.
- Providing features which may have start/end time properties, and then
  creating windows that correspond to items in a data source. So user can say
  I want a NAIP image of Seattle in 2019-2021, but then the window ends up
  being exactly the date of the NAIP image. This is probably something to
  partially support via the Python library and not through CLI. This is useful
  for then finding other images that match in time with the NAIP image. I
  suppose it could be supported in a different way too.


Preparing Datasets
------------------

Reminder: the prepare function looks up items in retrieved layers that are
relevant to each spatiotemporal window in the dataset.

- If the data source supports it, only matching windows with items that are in
  the same projection. For example if we want to make sure we're not warping
  Sentinel-2 images at all.
- Finding up to N items that are all within the time range of the window.
- Finding up to N items that are closest to the time range of the window.
- Finding individual items that fully cover the spatial extent of the window,
  or finding any item that intersects the extent, or finding groups of items
  (mosaics) that fully cover the extent.
- When finding a limited number of items within the window time range, users
  should be able to order the items by attributes like cloud cover.
- Finding at least M items and discarding windows where that minimum is not
  satisfied.
- Finding items that are some offset from the window time range (e.g.
  historical images for context about how location has changed).


Ingesting Datasets
------------------

- Choosing the format of image: at least GeoTIFF, PNG, JPEG.
- Storing 16-bit, 32-bit, and floating point values without converting dtype
  (only for GeoTIFF I think).
- Normalizing large pixel values into 8-bit and potentially other dtypes via a
  configurable linear mapping.
- Should be easy to parallelize ingestion across many machines (e.g. ingest
  different subset of windows on each machine and rsync the datasets together
  to get final result).


Vector Data Types
-----------------

- Points
- Bounding boxes
- Polygons (can be used for instance segmentation, semantic segmentation, or
  per-pixel regression)
- Window-level classification
- Window-level regression


Raster Data Sources
-------------------

- Sentinel-1, Sentinel-2, and other images from ESA Copernicus API. Users
  should be able to filter scenes by options supported by the API such as cloud
  cover.
- Sentinel-2 images from AWS (https://aws.amazon.com/marketplace/pp/prodview-2ostsvrguftb2).
- Sentinel-2 images from GCP (https://cloud.google.com/storage/docs/public-datasets/sentinel-2).
- NAIP images from AWS (https://aws.amazon.com/marketplace/pp/prodview-cedhkcjocmfs4).
- Terrain-corrected Sentinel-1 and other images from Google Earth Engine.
- Landsat, NAIP, and other images from USGS API (https://m2m.cr.usgs.gov/).


Vector Data Sources
-------------------

- OpenStreetMap. Users should specify a data type (polygon, point, etc), along
  with filters on tags or property functions that compute things like class IDs
  from tags.


Annotation
----------

- For WebMercator windows, viewing images in supported data sources like tile
  URL without ingesting the images.
- Viewing both raster and vector layers in the dataset.
- Configuring which layers show up in the annotation tool.
- Configuring which sets of bands in each layer to display.
- For retrieved layers where we retrieved multiple items or mosaics, toggling
  between those different items.


Model Training
--------------

- Train over union of multiple datasets.
- Input individual images, image time series, one image from each of many
  modalities, and multi-modal time series.
- Train on subsets of bands from a raster layer.
- Train on subsets of categories or modalities in a vector layer.
- Train with multiple targets where the model may need multiple heads.
- Easily drop in different model architectures.
- Specify various Lightning modules to use in training?
- Easily use various pre-trained model weights (both built-in and custom).


Model Inference
---------------

- Apply on subsets of the world in Kubernetes or Beaker jobs.
- Set it up so that it's easy to integrate into InferD or Vertex AI.
- Apply on custom GeoTIFF files.
