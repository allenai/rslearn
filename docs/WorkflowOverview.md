Usage of rslearn typically follows this workflow:

1. Specify the dataset configuration file.
2. Create windows.
3. Import raster and vector data from the configured data sources that align with the
   windows.
4. If needed, programmatically add additional raster and vector data to the dataset
   (e.g., labels from an external annotation tool).
5. Specify the model configuration file and train a model.
6. Get model predictions in new regions.

Steps 1-3 are sufficient if you are using rslearn to obtain remote sensing data, but
not to train remote sensing models.

## Specify the Dataset Configuration File

The dataset configuration file specifies the raster and vector layers in the rslearn
dataset. Each layer can be programmatically populated with data, or can specify a data
source from which the layer can be automatically populated.

- [DatasetConfig](./DatasetConfig.md) provides a reference for the dataset
  configuration file.
- [DataSources](./DataSources.md) details the data sources that are built-in to
  rslearn. For each data source, example usage is provided, including an example
  dataset configuration file.
- Each example in [Examples](./Examples.md) includes a dataset configuration file.

## Create Windows

rslearn datasets consist of windows. Each window is a spatiotemporal box.

- [DatasetAddWindows](./DatasetAddWindows.md) details how to create windows from the
  command-line interface.
- [ProgrammaticWindows](examples/ProgrammaticWindows.md) shows how to create windows
  programmatically.

## Import from Data Sources

Once windows are created, data can be imported from configured data sources by running
the prepare (match data source items to windows), ingest (download items), and
materialize (re-project and crop items to align with windows) stages:

```
rslearn dataset prepare --root /path/to/dataset
rslearn dataset ingest --root /path/to/dataset
rslearn dataset materialize --root /path/to/dataset
```

## Add Additional Raster and Vector Data

If you have vector annotations that you want to use as targets during training, or you
already have your own remote sensing images, then you can add these to your rslearn
dataset instead of importing it from a data source.

[ProgrammaticWindows](examples/ProgrammaticWindows.md) shows how to programmatically
add additional raster and vector data to a dataset. Alternatively, if you have a
local collection of GeoTIFF or GeoJSON files, you can use the
[LocalFiles data source](./data_sources/LocalFiles.md) to import them.

## Specify the Model Configuration File

The model configuration file specifies the model architecture, the machine learning
task (e.g. segmentation or object detection), the dataset layers to use as inputs and
targets, and training hyperparameters.

- [ModelConfig](./ModelConfig.md) provides a reference for the model configuration
  file.
- [TasksAndModels](./TasksAndModels.md) details the built-in tasks and model
  components.
- The examples in [Examples](./Examples.md) that involve training a model each include
  a model configuration file.
- See [OlmoEarth.md](./foundation_models/OlmoEarth.md) for details on fine-tuning
  OlmoEarth in particular.

## Get Model Predictions

Getting model predictions in a new region involves creating windows corresponding to
that region, and running the `rslearn model predict` command. The examples in the
[README](../README.md) and [Examples](./Examples.md) show how to do that.
