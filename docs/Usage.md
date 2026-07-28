# Usage

An rslearn workflow typically follows these steps:

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

## Specify the Dataset Configuration

The dataset configuration file specifies the raster and vector layers in the rslearn
dataset. Each layer can be programmatically populated with data, or can specify a data
source from which the layer can be automatically populated.

- [DatasetConfig](./DatasetConfig.md) provides a reference for the dataset
  configuration file.
- [DataSources](./DataSources.md) details the data sources that are built-in to
  rslearn. For each data source, example usage is provided, including an example
  dataset configuration file.
- [Compositors](./Compositors.md) details built-in and custom raster compositors.
- Each example in [Examples](./Examples.md) includes a dataset configuration file.

## Create Windows

rslearn datasets consist of windows. Each window is a spatiotemporal box.

- [Create Windows via CLI](DatasetAddWindows.md) covers bounding boxes, vector files,
  projections, fixed-size windows, and grids.
- [Create Windows Programmatically](CreateWindowsProgrammatically.md) covers custom
  logic such as per-feature time ranges and train/validation metadata.

## Import from Data Sources

Once windows are created, data can be imported from configured data sources by running
the prepare (match data source items to windows), ingest (download items), and
materialize (re-project and crop items to align with windows) stages:

```bash
rslearn dataset prepare --root /path/to/dataset
rslearn dataset ingest --root /path/to/dataset
rslearn dataset materialize --root /path/to/dataset
```

## Add Additional Raster and Vector Data

If you already have imagery or annotations, add them through the `LocalFiles` data
source or write data directly to windows:

- [Add Raster Data](AddRasterData.md) shows how to import GeoTIFFs and how to convert
  and write a task-ready label raster directly.
- [Add Vector Data](AddVectorData.md) shows how to import GeoJSON/Shapefile data and
  how to create windows and point labels from a CSV.

The [Find Stadiums](examples/FindStadiums.md) tutorial demonstrates a related pattern:
it starts with vector point labels and writes raster labels for `SegmentationTask`.

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
that region, and running the `rslearn model predict` command. The
[Quickstart](examples/Quickstart.md) and other [Examples](Examples.md) show complete
prediction workflows.
