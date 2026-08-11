# Configuration Reference

rslearn is driven by two configuration files:

- **[Dataset configuration](DatasetConfig.md)** (`config.json`) defines your dataset:
  the layers, the data sources they are imported from, how rasters are composited, and
  how windows are stored. This is what you set up when building a dataset and importing
  raster and vector data into it.
- **[Model configuration](ModelConfig.md)** defines model training: the model
  architecture and components, the training task, data transforms, and the training
  loop. This is what you set up when fine-tuning a model on a dataset.
