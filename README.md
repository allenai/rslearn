Overview
--------

rslearn is a library and tool for developing remote sensing datasets and models.

rslearn helps with:

1. Developing remote sensing datasets, starting with defining spatiotemporal windows
   (roughly equivalent to training examples) that should be annotated.
2. Importing raster and vector data from various online or local data sources into the
   dataset.
3. Annotating new categories of vector data (like points, polygons, and classification
   labels) using integrated web-based labeling apps.
4. Fine-tuning remote sensing foundation models on these datasets.
5. Applying models on new locations and times.


Quick links:
- [CoreConcepts](docs/CoreConcepts.md) summarizes key concepts in rslearn, including
  datasets, windows, layers, and data sources.
- [Examples](docs/Examples.md) contains more examples, including customizing different
  stages of rslearn with additional code.
- [DatasetConfig](docs/DatasetConfig.md) documents the dataset configuration file.
- [DataSources](docs/DataSources.md) details the built-in data sources in rslearn, from
  which raster and vector data can be imported into rslearn dataset layers.
- [ModelConfig](docs/ModelConfig.md) documents the model configuration file.
- [TasksAndModels](docs/TasksAndModels.md) details the training tasks and model
  components available in rslearn.


Setup
-----

rslearn requires Python 3.10+ (Python 3.12 is recommended).

```
git clone https://github.com/allenai/rslearn.git
cd rslearn
pip install .[extra]
```

Experiment tracking
-------------------

rslearn supports experiment logging via the Weights & Biases API. If you prefer a local,
wandb-compatible backend, you can install Trackio and have rslearn use it instead:

```
pip install trackio
export RSLEARN_WANDB_BACKEND=trackio
```

Trackio is API-compatible with `wandb.init`, `wandb.log`, and `wandb.finish` (rslearn uses
this subset). By default, rslearn uses `wandb` if it is installed, otherwise it falls
back to `trackio` if available.


Supported Data Sources
----------------------

rslearn supports ingesting raster and vector data from the following data sources. Even
if you don't plan to train models within rslearn, you can still use it to easily
download, crop, and re-project data based on spatiotemporal rectangles (windows) that
you define. See [Examples](docs/Examples.md) and [DatasetConfig](docs/DatasetConfig.md)
for how to setup these data sources.

- Sentinel-1
- Sentinel-2 L1C and L2A
- Landsat 8/9 OLI-TIRS
- National Agriculture Imagery Program
- OpenStreetMap
- Xyz (Slippy) Tiles (e.g., Mapbox tiles)
- Planet Labs (PlanetScope, SkySat)
- ESA WorldCover 2021
- ISRIC SoilGrids (WCS)

rslearn can also be used to easily mosaic, crop, and re-project any sets of local
raster and vector files you may have.


Example Usage
-------------

This is an example of building a remote sensing dataset, and then training a model
on that dataset, using rslearn. Specifically, we will train a model that inputs
Sentinel-2 images and predicts land cover through a semantic segmentation task.

Let's start by defining a region of interest and obtaining Sentinel-2 images. Create a
directory `/path/to/dataset` and corresponding configuration file at
`/path/to/dataset/config.json` as follows:

```json
{
    "layers": {
        "sentinel2": {
            "type": "raster",
            "band_sets": [{
                "dtype": "uint8",
                "bands": ["R", "G", "B"]
            }],
            "data_source": {
                "class_path": "rslearn.data_sources.gcp_public_data.Sentinel2",
                "init_args": {
                    "index_cache_dir": "cache/sentinel2/",
                    "sort_by": "cloud_cover",
                    "use_rtree_index": false
                }
            }
        }
    }
}
```

Here, we have initialized an empty dataset and defined a raster layer called
`sentinel2`. Because it specifies a data source, it will be populated automatically. In
particular, the data will be sourced from a
[public Google Cloud Storage bucket containing Sentinel-2 imagery](https://cloud.google.com/storage/docs/public-datasets/sentinel-2).
The `sort_by` option sorts scenes in ascending order by cloud cover, so we will end up
choosing the scenes with minimal cloud cover.

Next, let's create our spatiotemporal windows. These will correspond to training
examples.

```
export DATASET_PATH=/path/to/dataset
rslearn dataset add_windows --root $DATASET_PATH --group default --utm --resolution 10 --grid_size 128 --src_crs EPSG:4326 --box=-122.6901,47.2079,-121.4955,47.9403 --start 2024-06-01T00:00:00+00:00 --end 2024-08-01T00:00:00+00:00 --name seattle
```

This creates windows along a 128x128 grid in the specified projection (i.e.,
appropriate UTM zone for the location with 10 m/pixel resolution) covering the
specified bounding box, which is centered at Seattle.

We can now obtain the Sentinel-2 images by running prepare, ingest, and materialize.

* Prepare: lookup items (in this case, Sentinel-2 scenes) in the data source that match with the spatiotemporal windows we created.
* Ingest: retrieve those items. This step populates the `tiles` directory within the dataset.
* Materialize: crop/mosaic the items to align with the windows. This populates the `layers` folder in each window directory.

```
rslearn dataset prepare --root $DATASET_PATH --workers 32 --batch-size 8
rslearn dataset ingest --root $DATASET_PATH --workers 32 --no-use-initial-job --jobs-per-process 1
rslearn dataset materialize --root $DATASET_PATH --workers 32 --no-use-initial-job
```

For ingestion, you may need to reduce the number of workers depending on the available
memory on your system.

You should now be able to open the GeoTIFF images. Let's find the window that
corresponds to downtown Seattle:

```python
import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset
from rslearn.utils import Projection, STGeometry
from upath import UPath

# Define longitude and latitude for downtown Seattle.
downtown_seattle = shapely.Point(-122.333, 47.606)

# Iterate over the windows and find the closest one.
dataset = Dataset(path=UPath("/path/to/dataset"))
best_window_name = None
best_distance = None
for window in dataset.load_windows(workers=32):
    shp = window.get_geometry().to_projection(WGS84_PROJECTION).shp
    distance = shp.distance(downtown_seattle)
    if best_distance is None or distance < best_distance:
        best_window_name = window.name
        best_distance = distance

print(best_window_name)
```

It should be `seattle_54912_-527360`, so let's open it in qgis (or your favorite GIS
software):

```
qgis $DATASET_PATH/windows/default/seattle_54912_-527360/layers/sentinel2/R_G_B/geotiff.tif
```


### Adding Land Cover Labels

Before we can train a land cover prediction model, we need labels. Here, we will use
the ESA WorldCover land cover map as labels.

Start by downloading the WorldCover data from https://worldcover2021.esa.int

```
wget https://worldcover2021.esa.int/data/archive/ESA_WorldCover_10m_2021_v200_60deg_macrotile_N30W180.zip
mkdir world_cover_tifs
unzip ESA_WorldCover_10m_2021_v200_60deg_macrotile_N30W180.zip -d world_cover_tifs/
```

It would require some work to write a script to re-project and crop these GeoTIFFs so
that they align with the windows we have previously defined (and the Sentinel-2 images
we have already ingested). We can use the LocalFiles data source to have rslearn
automate this process. Update the dataset `config.json` with a new layer:

```jsonc
"layers": {
    "sentinel2": {
        # ...
    },
    "worldcover": {
        "type": "raster",
        "band_sets": [{
            "dtype": "uint8",
            "bands": ["B1"]
        }],
        "resampling_method": "nearest",
        "data_source": {
            "class_path": "rslearn.data_sources.local_files.LocalFiles",
            "init_args": {
                "src_dir": "file:///path/to/world_cover_tifs/"
            }
        }
    }
},
# ...
```

Repeat the materialize process so we populate the data for this new layer:

```
rslearn dataset prepare --root $DATASET_PATH --workers 32 --batch-size 8
rslearn dataset ingest --root $DATASET_PATH --workers 32 --no-use-initial-job --jobs-per-process 1
rslearn dataset materialize --root $DATASET_PATH --workers 32 --no-use-initial-job
```

We can visualize both the GeoTIFFs together in qgis:

```
qgis $DATASET_PATH/windows/default/seattle_54912_-527360/layers/*/*/geotiff.tif
```

We can also visualize samples using the visualization module:
```
python -m rslearn.vis.vis_server \
    $DATASET_PATH \
    --layers sentinel2 \  # image modality layers
    --label_layers label_raster \  # layer layers
    --bands '{"sentinel2": ["B04", "B03", "B02"]}' \  # specify bands wanted for each image modality
    --normalization '{"sentinel2": "sentinel2_rgb"}'  \  # specify normalization wanted for each image modality
    --task_type segmentation \  # segmentation, detection, or classification
    --max_samples 100 \  # number of datapoints to randomly sample and visualize
    --port 8000
```

### Training a Model

Create a model configuration file `land_cover_model.yaml`:

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    # This part defines the model architecture.
    # Essentially we apply the SatlasPretrain Sentinel-2 backbone with a UNet decoder
    # that terminates at a segmentation prediction head.
    # The backbone outputs four feature maps at different scales, and the UNet uses
    # these to compute a feature map at the input scale.
    # Finally the segmentation head applies per-pixel softmax to compute the land
    # cover class.
    model:
      class_path: rslearn.models.singletask.SingleTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.satlaspretrain.SatlasPretrain
            init_args:
              model_identifier: "Sentinel2_SwinB_SI_RGB"
        decoder:
          - class_path: rslearn.models.unet.UNetDecoder
            init_args:
              in_channels: [[4, 128], [8, 256], [16, 512], [32, 1024]]
              # We use 101 classes because the WorldCover classes are 10, 20, 30, 40
              # 50, 60, 70, 80, 90, 95, 100.
              # We could process the GeoTIFFs to collapse them to 0-10 (the 11 actual
              # classes) but the model will quickly learn that the intermediate
              # values are never used.
              out_channels: 101
              conv_layers_per_resolution: 2
          - class_path: rslearn.train.tasks.segmentation.SegmentationHead
    # Remaining parameters in RslearnLightningModule define different aspects of the
    # training process like initial learning rate.
    lr: 0.0001
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    path: ${DATASET_PATH}
    # This defines the layers that should be read for each window.
    # The key ("image" / "targets") is what the data will be called in the model,
    # while the layers option specifies which layers will be read.
    inputs:
      image:
        data_type: "raster"
        layers: ["sentinel2"]
        bands: ["R", "G", "B"]
        passthrough: true
      targets:
        data_type: "raster"
        layers: ["worldcover"]
        bands: ["B1"]
        is_target: true
    task:
      # Train for semantic segmentation.
      # The remap option is only used when visualizing outputs during testing.
      class_path: rslearn.train.tasks.segmentation.SegmentationTask
      init_args:
        num_classes: 101
        remap_values: [[0, 1], [0, 255]]
    batch_size: 8
    num_workers: 32
    # These define different options for different phases/splits, like training,
    # validation, and testing.
    # Here we use the same transform across splits except training where we add a
    # flipping augmentation.
    # For now we are using the same windows for training and validation.
    default_config:
      transforms:
        - class_path: rslearn.train.transforms.normalize.Normalize
          init_args:
            mean: 0
            std: 255
    train_config:
      transforms:
        - class_path: rslearn.train.transforms.normalize.Normalize
          init_args:
            mean: 0
            std: 255
        - class_path: rslearn.train.transforms.flip.Flip
          init_args:
            image_selectors: ["image", "target/classes", "target/valid"]
      groups: ["default"]
    val_config:
      groups: ["default"]
    test_config:
      groups: ["default"]
    predict_config:
      groups: ["predict"]
      load_all_crops: true
      skip_targets: true
      patch_size: 512
trainer:
  max_epochs: 10
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        save_top_k: 1
        save_last: true
        monitor: val_accuracy
        mode: max
        dirpath: ./land_cover_model_checkpoints/
```

Now we can train the model:

```
rslearn model fit --config land_cover_model.yaml
```


### Apply the Model

Let's apply the model on Portland, OR (you can change it to Portland, ME if you like).
We start by defining a new window around Portland. This time, instead of creating
windows along a grid, we just create one big window. This is because we are just going
to run the prediction over the whole window rather than use different windows as
different training examples.

```
rslearn dataset add_windows --root $DATASET_PATH --group predict --utm --resolution 10 --src_crs EPSG:4326 --box=-122.712,45.477,-122.621,45.549 --start 2024-06-01T00:00:00+00:00 --end 2024-08-01T00:00:00+00:00 --name portland
rslearn dataset prepare --root $DATASET_PATH --workers 32 --batch-size 8
rslearn dataset ingest --root $DATASET_PATH --workers 32 --no-use-initial-job --jobs-per-process 1
rslearn dataset materialize --root $DATASET_PATH --workers 32 --no-use-initial-job
```

We also need to add an RslearnPredictionWriter to the trainer callbacks in the model
configuration file, as it will handle writing the outputs from the model to a GeoTIFF.

```yaml
trainer:
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      ...
    - class_path: rslearn.train.prediction_writer.RslearnWriter
      init_args:
        # We need to include this argument, but it will be overridden with the dataset
        # path from data.init_args.path.
        path: placeholder
        output_layer: output
```

Because of our `predict_config`, when we run `model predict` it will apply the model on
windows in the "predict" group, which is where we added the Portland window.

And it will be written in a new output_layer called "output". But we have to update the
dataset configuration so it specifies the layer:

```jsonc
"layers": {
    "sentinel2": {
        # ...
    },
    "worldcover": {
        # ...
    },
    "output": {
        "type": "raster",
        "band_sets": [{
            "dtype": "uint8",
            "bands": ["output"]
        }]
    }
},
```

Now we can apply the model:

```
# Find model checkpoint in lightning_logs dir.
ls lightning_logs/*/checkpoints/last.ckpt
rslearn model predict --config land_cover_model.yaml --ckpt_path land_cover_model_checkpoints/last.ckpt
```

And visualize the Sentinel-2 image and output in qgis:

```
qgis $DATASET_PATH/windows/predict/portland/layers/*/*/geotiff.tif
```


### Defining Train and Validation Splits

We can visualize the logged metrics using Tensorboard:

```
tensorboard --logdir=lightning_logs/
```

However, because our training and validation data are identical, the validation metrics
are not meaningful.

There are two suggested ways to split windows into different subsets:

1. Assign windows to different groups.
2. Use different key-value pairs in the windows' options dicts for different splits.

We will use the second approach. The script below sets a "split" key in the options
dict (which is stored in each window's `metadata.json` file) to "train" or "val"
based on the SHA-256 hash of the window name.

```python
import hashlib
import tqdm
from rslearn.dataset import Dataset, Window
from upath import UPath

ds_path = UPath("/path/to/dataset/")
dataset = Dataset(ds_path)
windows = dataset.load_windows(show_progress=True, workers=32)
for window in tqdm.tqdm(windows):
    if hashlib.sha256(window.name.encode()).hexdigest()[0] in ["0", "1"]:
        split = "val"
    else:
        split = "train"
    if "split" in window.options and window.options["split"] == split:
        continue
    window.options["split"] = split
    window.save()
```

Now we can update the model configuration file to use these splits:

```yaml
default_config:
  transforms:
    - class_path: rslearn.train.transforms.normalize.Normalize
      init_args:
        mean: 0
        std: 255
train_config:
  transforms:
    - class_path: rslearn.train.transforms.normalize.Normalize
      init_args:
        mean: 0
        std: 255
    - class_path: rslearn.train.transforms.flip.Flip
      init_args:
        image_selectors: ["image", "target/classes", "target/valid"]
  groups: ["default"]
  tags:
    split: train
val_config:
  groups: ["default"]
  tags:
    split: val
test_config:
  groups: ["default"]
  tags:
    split: val
predict_config:
  groups: ["predict"]
  load_all_crops: true
  skip_targets: true
  patch_size: 512
```

The `tags` option that we are adding here tells rslearn to only load windows with a
matching key and value in the window options.

Previously when we run `model fit`, it should show the same number of windows for
training and validation:

```
got 4752 examples in split train
got 4752 examples in split val
```

With the updates, it should show different numbers like this:

```
got 4167 examples in split train
got 585 examples in split val
```


### Visualizing with `model test`

We can visualize the ground truth labels and model predictions in the test set using
the `model test` command:

```
mkdir ./vis
rslearn model test --config land_cover_model.yaml --ckpt_path land_cover_model_checkpoints/last.ckpt --model.init_args.visualize_dir=./vis/
```

This will produce PNGs in the vis directory. The visualizations are produced by the
`Task.visualize` function, so we could customize the visualization by subclassing
SegmentationTask and overriding the visualize function.


### Checkpoint and Logging Management

Above, we needed to configure the checkpoint directory in the model config (the
`dirpath` option under `lightning.pytorch.callbacks.ModelCheckpoint`), and explicitly
specify the checkpoint path when applying the model. Additionally, metrics are logged
to the local filesystem and not well organized.

We can instead let rslearn automatically manage checkpoints, along with logging to
Weights & Biases. To do so, we add project_name, run_name, and management_dir options
to the model config. The project_name corresponds to the W&B project, and the run name
corresponds to the W&B name. The management_dir is a directory to store project data;
rslearn determines a per-project directory at `{management_dir}/{project_name}/{run_name}/`
and uses it to store checkpoints.

```yaml
model:
  # ...
data:
  # ...
trainer:
  # ...
project_name: land_cover_model
run_name: version_00
# This sets the option via the MANAGEMENT_DIR environment variable.
management_dir: ${MANAGEMENT_DIR}
```

Now, set the `MANAGEMENT_DIR` environment variable and run `model fit`:

```
export MANAGEMENT_DIR=./project_data
rslearn model fit --config land_cover_model.yaml
```

The training and validation loss and accuracy metric should now be logged to W&B. The
accuracy metric is provided by SegmentationTask, and additional metrics can be enabled
by passing the relevant init_args to the task, e.g. mean IoU and F1:

```yaml
      class_path: rslearn.train.tasks.segmentation.SegmentationTask
      init_args:
        num_classes: 101
        remap_values: [[0, 1], [0, 255]]
        enable_miou_metric: true
        enable_f1_metric: true
```

When calling `model test` and `model predict` with management_dir set, rslearn will
automatically load the best checkpoint from the project directory, or raise an error if
no existing checkpoint exists. This behavior can be overridden with the
`--load_checkpoint_mode` and `--load_checkpoint_required` options (see `--help` for
details). Logging will be enabled during fit but not test/predict, and this can also
be overridden, using `--log_mode`.


### Inputting Multiple Sentinel-2 Images

Currently our model inputs a single Sentinel-2 image. However, for most tasks where
labels are not expected to change from week to week, we find that accuracy can be
significantly improved by inputting multiple images, regardless of the pre-trained
model used. Multiple images makes the model more resilient to clouds and image
artifacts, and allows the model to synthesize information across different views that
may come from different seasons or weather conditions.

We first update our dataset configuration to obtain three images, by customizing the
query_config section. This can replace the sentinel2 layer:

```jsonc
"layers": {
    "sentinel2_multi": {
        "type": "raster",
        "band_sets": [{
            "dtype": "uint8",
            "bands": ["R", "G", "B"]
        }],
        "data_source": {
            "class_path": "rslearn.data_sources.gcp_public_data.Sentinel2",
            "init_args": {
              "index_cache_dir": "cache/sentinel2/",
              "sort_by": "cloud_cover",
              "use_rtree_index": false
            },
            "query_config": {
                "max_matches": 3
            }
        }
    },
    "worldcover": {
        # ...
    },
    "output": {
        # ...
    }
}
```

Repeat the steps from earlier to prepare, ingest, and materialize the dataset.

Now we update our model configuration file. First, we modify the model architecture to
be able to input an image time series. We use the SimpleTimeSeries model, which takes
an encoder that expects a single-image input, and applies that encoder on each image in
the time series. It then applies max temporal pooling to combine the per-image feature
maps extracted by the encoder.

Image time series in rslearn are currently stored as [T*C, H, W] tensors. So we pass
the `image_channels` to SimpleTimeSeries so it knows how to slice up the tensor to
recover the per-timestep images.

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.singletask.SingleTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.simple_time_series.SimpleTimeSeries
            init_args:
              encoder:
                class_path: rslearn.models.satlaspretrain.SatlasPretrain
                init_args:
                  model_identifier: "Sentinel2_SwinB_SI_RGB"
              image_channels: 3
        decoder:
          # ...
```

Next, we update the data module section so that the dataset loads the image time series
rather than a single image. The `load_all_layers` option tells the dataset to stack the
rasters from all of the layers specified, and also to ignore windows where any of those
layers are missing.

```yaml
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    path: # ...
    inputs:
      image:
        data_type: "raster"
        layers: ["sentinel2_multi", "sentinel2_multi.1", "sentinel2_multi.2"]
        bands: ["R", "G", "B"]
        passthrough: true
        load_all_layers: true
      targets:
        # ...
```

Now we can train an updated model:

```
rslearn model fit --config land_cover_model.yaml
```


Contact
-------

For questions and suggestions, please open an issue on GitHub.
