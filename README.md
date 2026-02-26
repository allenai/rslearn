rslearn is a tool for developing remote sensing datasets and models.

rslearn helps with:

1. Developing remote sensing datasets, starting with defining spatiotemporal windows
   (3D boxes in height/width/time) that are roughly equivalent to training examples.
2. Importing raster and vector data from various online or local data sources into the
   dataset.
3. Fine-tuning remote sensing foundation models on these datasets.
4. Applying models on new locations and times.


Quickstart
----------

If you are new to rslearn, we suggest starting here:

1. First, read [CoreConcepts](docs/CoreConcepts.md), which summarizes key concepts in
   rslearn, including datasets, windows, layers, and data sources.
2. Second, read [WorkflowOverview](docs/WorkflowOverview.md), which provides an overview
   of the typical workflow in rslearn, from defining windows to training models.
3. Finally, walk through the [IntroExample](docs/examples/IntroExample.md), or find
   another example in [Examples.md](docs/Examples.md) that can most readily be adapted
   for your project.

Other links:
- [DatasetConfig](docs/DatasetConfig.md) documents the dataset configuration file.
- [DataSources](docs/DataSources.md) details the built-in data sources in rslearn, from
  which raster and vector data can be imported into rslearn dataset layers.
- [ModelConfig](docs/ModelConfig.md) documents the model configuration file.
- [TasksAndModels](docs/TasksAndModels.md) details the training tasks and model
  components available in rslearn.


Setup
-----

rslearn requires Python 3.11+ (Python 3.12 is recommended).

```
git clone https://github.com/allenai/rslearn.git
cd rslearn
pip install .[extra]
```

For linting and tests:

```
pip install .[dev]
pre-commit install
pre-commit run --all-files
pytest tests/unit tests/integration
# For online data source tests, you can store credentials in .env and they will be
# loaded by pytest-dotenv.
pytest tests/online
```


Contact
-------

For questions and suggestions, please open an issue on GitHub.
