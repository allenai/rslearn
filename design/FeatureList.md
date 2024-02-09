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