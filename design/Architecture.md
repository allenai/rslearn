As of 2024-06-11, rslearn consists of the following modules:

- config: objects to store the configuration of datasets.
- dataset: datasets and windows, along with the management functions to
  prepare, ingest, and materialize datasets.
- data_sources: data source implementations that support retrieving data into
  the dataset.
- tile_stores: tile store implementations. During ingestion, items from the
  dataset are stored in the tile store. During materialization, these items are
  read from the tile store and written to the window directory.
- utils: various utilities.
- models: model architectures available for training.
- train: support for model training and inference.
- main.py: command-line interface.

These modules are detailed below.


config
------

The classes here represent a dataset configuration file. The configuration file
is currently a JSON file, and these classes just parse different parts of the
JSON file into a more easy-to-use representation.

Eventually we should try to migrate this to jsonargparse so the config objects
can be set directly.


dataset
-------

- `dataset.py` and `window.py` provide the basics of accessing a dataset and
  the windows that it consists of.
- `add_windows.py` provides functions for adding new windows to a dataset in
  various supported ways.
- `manage.py` provides high-level prepare, ingest, and materialize functions.
- `materialize.py` defines materializers that use different methods to
  materialize ingested data (sitting in the tile store) into the dataset.


data_sources
------------

- `raster_source.py` provides shared utility functions for raster data sources,
  like ingesting a rasterio dataset after the raster data has been downloaded.


tile_stores
-----------

A tile store supports reading and writing raster and vector data. Currently
there is only a local filesystem implementation which can be configured with a
raster format and a vector format to use to save the data (e.g. GeoTIFF or PNG
for rasters). The default is to store rasters with GeoTIFF and vector data with
GeoJSON.

The tile store of a dataset is configured through
`rslearn.config.dataset.TileStoreConfig`.


utils
-----

- feature: a vector feature, to be used across rslearn for interacting with
  vector data.
- geometry: defines the Projection and STGeometry classes that are used across
  rslearn for representing spatiotemporal geometries (has shape in space and a
  time range) and reprojecting them to different projections.
- raster_format, vector_format: different formats for storing raster and vector
  data.


train
-----

This module is built on top of pytorch Lightning and supports training and
applying models in rslearn.

Most tasks are expected to use a common LightningModule (in
`rslearn.train.lightning_module`) and LightningDataModule (in
`rslearn.train.data_module`).

The RslearnDataModule reads configurable layers from an rslearn dataset while
providing options to customize the transforms, whether to read patches of
windows versus entire windows, which groups to read for the train/val/test
splits, etc.

A Task (`rslearn.train.tasks`) specifies how raw raster and vector data should
be processed into targets for training. For example, `RegressionTask` exposes
options to specify how a regression target value should be read from vector
features, and returns that value. The Task also provides metrics.

The RslearnLightningModule initializes the model and loads the metrics from the
specified Task.

The model is expected to input two dictionaries (input_dicts, target_dicts)
containing the inputs and targets respectively (as returned by the Task). It
must output a tuple (outputs, loss_dict), where outputs is compatible with the
metrics and visualization functions in the Task, while loss_dict is a map from
a label to a scalar Tensor.


models
------

This module contains components for deep learning models.

See this example YAML config for an example of configuring the models:
https://github.com/allenai/rslearn_projects/blob/master/sentinel2_postprocess_revamp/train_config.yaml
