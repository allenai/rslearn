## Quickstart

This is an example of building a remote sensing dataset, and then training a model
on that dataset, using rslearn. Specifically, we will fine-tune the OlmoEarth
foundation model to predict land cover from Sentinel-2 images through a semantic
segmentation task.

## Creating Windows and Obtaining Satellite Imagery

Let's start by defining a region of interest and obtaining Sentinel-2 images. Create a
directory `/path/to/dataset` and corresponding configuration file at
`/path/to/dataset/config.json` as follows:

```json
{
  "layers": {
    "sentinel2": {
      "type": "raster",
      "band_sets": [
        {
          "dtype": "uint8",
          "bands": ["R", "G", "B"]
        },
        {
          "dtype": "uint16",
          "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
        }
      ],
      "data_source": {
        "class_path": "rslearn.data_sources.planetary_computer.Sentinel2",
        "init_args": {
          "cache_dir": "cache/planetary_computer",
          "harmonize": true,
          "sort_by": "eo:cloud_cover"
        },
        "ingest": false,
        "query_config": {
          "max_matches": 1,
          "space_mode": "MOSAIC"
        }
      }
    }
  }
}
```

Here, we have initialized an empty dataset and defined a raster layer called
`sentinel2`. Because it specifies a data source, it will be populated automatically. The
imagery comes from Sentinel-2 L2A scenes on
[Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a),
which are stored as Cloud-Optimized GeoTIFFs (COGs). Because COGs can be read directly,
rslearn can crop the imagery straight from the scenes during materialization, so we set
`"ingest": false` and skip the ingest step entirely (which would otherwise download entire
Sentinel-2 scenes first). The `sort_by` option selects the least cloudy scene, and
`space_mode: MOSAIC` mosaics multiple scenes together when a single one does not cover the whole window.

We define two band sets: an 8-bit `R`, `G`, `B` band set that is convenient for
visualization, and a 16-bit band set with the 12 spectral bands that we will feed to
the model.

Next, let's create our spatiotemporal windows. These will correspond to training
examples.

```
export DATASET_PATH=/path/to/dataset
rslearn dataset add_windows --root $DATASET_PATH --group default --utm --resolution 10 --grid_size 128 --src_crs EPSG:4326 --box=-122.45,47.55,-122.30,47.67 --start 2024-06-01T00:00:00+00:00 --end 2024-08-01T00:00:00+00:00 --name seattle
```

This creates windows along a 128x128 grid in the specified projection (i.e.,
appropriate UTM zone for the location with 10 m/pixel resolution) covering the
specified bounding box over Seattle. We use a small bounding box here so that we only
create 110 windows, which keeps this quickstart fast to run; for better results, use a
larger bounding box to create more training examples, e.g.
`--box=-122.7,47.2,-121.5,48.0`.

We can now obtain the Sentinel-2 images. Because the Planetary Computer scenes are
Cloud-Optimized GeoTIFFs, we can materialize the imagery directly and skip the ingest
step.

* Prepare: lookup items (in this case, Sentinel-2 scenes) in the data source that match with the spatiotemporal windows we created.
* Ingest (skipped since the source images are already COGs): download those items locally and convert to formats that support random access.
* Materialize: crop/mosaic the items to align with the windows. This populates the `layers` folder in each window directory.

```
rslearn dataset prepare --root $DATASET_PATH
rslearn dataset materialize --root $DATASET_PATH
```

You should now be able to open the GeoTIFF images. Let's find the window that
corresponds to downtown Seattle:

```python
import os

import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset
from rslearn.utils import Projection, STGeometry
from upath import UPath

# Define longitude and latitude for downtown Seattle.
downtown_seattle = shapely.Point(-122.333, 47.606)

# Iterate over the windows and find the closest one.
dataset = Dataset(path=UPath(os.environ["DATASET_PATH"]))
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


## Adding Land Cover Labels

Before we can train a land cover prediction model, we need labels. Here, we will use
the ESA WorldCover land cover map as labels.

rslearn has a built-in WorldCover data source that reads directly from the public
Cloud-Optimized GeoTIFFs, so there is nothing to download or reproject manually. Update
the dataset `config.json` with a new layer:

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
      "class_path": "rslearn.data_sources.worldcover.WorldCover",
      "init_args": {
        "metadata_cache_dir": "cache/worldcover"
      },
      "ingest": false
    }
  }
},
# ...
```

Like the Sentinel-2 layer, WorldCover supports direct materialization, so we just
prepare and materialize to populate the new layer:

```
rslearn dataset prepare --root $DATASET_PATH
rslearn dataset materialize --root $DATASET_PATH
```

We can visualize both the GeoTIFFs together in qgis:

```
qgis $DATASET_PATH/windows/default/seattle_54912_-527360/layers/*/*/geotiff.tif
```

<p float="left">
<img src="seattle_sentinel2_rgb.png" alt="Sentinel-2 RGB image of downtown Seattle" width="45%" />
<img src="seattle_worldcover.png" alt="ESA WorldCover land cover labels for downtown Seattle" width="45%" />
</p>

*Sentinel-2 RGB image (left) and corresponding ESA WorldCover labels (right).*

We can also visualize samples using the visualization module:
```
python -m rslearn.vis.vis_server \
    $DATASET_PATH \
    --raster_groups sentinel2 worldcover \
    --bands '{"sentinel2": ["B04", "B03", "B02"], "worldcover": ["B1"]}' \
    --raster_render '{"sentinel2": {"name": "sentinel2_rgb"}, "worldcover": {"name": "classes"}}' \
    --max_samples 100 \
    --port 8000
```

## Training a Model

Create a model configuration file `land_cover_model.yaml`:

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    # This part defines the model architecture.
    # We apply the OlmoEarth encoder, followed by a UNet decoder that upsamples the
    # features back to the input resolution, and a segmentation head that predicts a
    # land cover class for each pixel.
    model:
      class_path: rslearn.models.singletask.SingleTaskModel
      init_args:
        encoder:
          # This applies the OlmoEarth encoder on the inputs. We use the v1.2 Nano
          # model here since it is fast; larger models generally give better accuracy.
          # Switch to TINY, SMALL, or BASE for better results.
          - class_path: rslearn.models.olmoearth_pretrain.model.OlmoEarth
            init_args:
              model_id: OLMOEARTH_V1_2_NANO
              # The patch size can be set between 1 and 8. Lower patch sizes are slower
              # but produce finer feature maps.
              patch_size: 8
              # Disable autocasting (which defaults to bfloat16) so the model can run
              # on CPU.
              autocast_dtype: null
        decoder:
          # The UNetDecoder upsamples the OlmoEarth features (which are at 1/patch_size
          # resolution) back to the input resolution. The OlmoEarth v1.2 Nano embedding
          # size is 128. Increase to 192 for TINY, 384 for SMALL, or 768 for BASE.
          - class_path: rslearn.models.unet.UNetDecoder
            init_args:
              in_channels: [[8, 128]]
              out_channels: 101
              conv_layers_per_resolution: 1
              num_channels: {8: 512, 4: 512, 2: 256, 1: 128}
          # The SegmentationHead computes softmax and cross entropy loss.
          # We use 101 classes because the WorldCover class IDs go up to 100.
          - class_path: rslearn.train.tasks.segmentation.SegmentationHead
    # When fine-tuning OlmoEarth we recommend LayerDecayAdamW, which applies a lower
    # learning rate to earlier encoder layers to preserve pre-trained features.
    optimizer:
      class_path: rslearn.models.olmoearth_pretrain.optimizer.LayerDecayAdamW
      init_args:
        lr: 0.0001
        # num_layers must match the encoder depth: 4 for Nano, 12 for Tiny/Small/Base.
        num_layers: 4
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    path: ${DATASET_PATH}
    # This defines the layers that should be read for each window.
    # The key ("sentinel2_l2a" / "targets") is what the data will be called in the
    # model, while the layers option specifies which dataset layers will be read.
    inputs:
      # OlmoEarth expects the Sentinel-2 input to be called "sentinel2_l2a".
      sentinel2_l2a:
        data_type: "raster"
        layers: ["sentinel2"]
        # This is the 12-band order expected by OlmoEarth.
        bands: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
        passthrough: true
        dtype: FLOAT32
      # We also load the uint8 RGB images for visualization.
      image:
        data_type: "raster"
        layers: ["sentinel2"]
        bands: ["R", "G", "B"]
        passthrough: true
        dtype: FLOAT32
      targets:
        data_type: "raster"
        layers: ["worldcover"]
        bands: ["B1"]
        dtype: FLOAT32
        is_target: true
    task:
      # Train for semantic segmentation.
      class_path: rslearn.train.tasks.segmentation.SegmentationTask
      init_args:
        num_classes: 101
    batch_size: 8
    num_workers: 32
    # These define different options for different phases/splits, like training,
    # validation, and testing.
    # OlmoEarthNormalize converts the raw Sentinel-2 reflectance values into the
    # normalized inputs expected by OlmoEarth.
    # For now we are using the same windows for training and validation.
    default_config:
      groups: ["default"]
      transforms:
        - class_path: rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize
          init_args:
            band_names:
              sentinel2_l2a: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
    train_config:
      transforms:
        - class_path: rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize
          init_args:
            band_names:
              sentinel2_l2a: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
        - class_path: rslearn.train.transforms.flip.Flip
          init_args:
            image_selectors: ["sentinel2_l2a", "target/classes", "target/valid"]
      groups: ["default"]
    val_config:
      groups: ["default"]
    test_config:
      groups: ["default"]
    predict_config:
      groups: ["predict"]
      load_all_crops: true
      skip_targets: true
      # crop_size matches our 128x128 training windows; overlap reduces border effects
      # during sliding-window inference on the larger prediction window.
      crop_size: 128
      overlap_pixels: 12
trainer:
  max_epochs: 100
  callbacks:
    - class_path: rslearn.train.callbacks.checkpointing.BestLastCheckpoint
      init_args:
        dirpath: ./land_cover_model_checkpoints/
        monitor: val_accuracy
        mode: max
```

Now we can train the model:

```
rslearn model fit --config land_cover_model.yaml
```


## Apply the Model

Let's apply the model on Portland, Oregon.
We start by defining a new window around Portland. This time, instead of creating
windows along a grid, we just create one big window. This is because we are just going
to run the prediction over the whole window rather than use different windows as
different training examples.

```
rslearn dataset add_windows --root $DATASET_PATH --group predict --utm --resolution 10 --src_crs EPSG:4326 --box=-122.712,45.477,-122.621,45.549 --start 2024-06-01T00:00:00+00:00 --end 2024-08-01T00:00:00+00:00 --name portland
rslearn dataset prepare --root $DATASET_PATH
rslearn dataset materialize --root $DATASET_PATH
```

We also need to add an RslearnWriter to the trainer callbacks in the model
configuration file, as it will handle writing the outputs from the model to a GeoTIFF.

```yaml
trainer:
  callbacks:
    - class_path: rslearn.train.callbacks.checkpointing.BestLastCheckpoint
      ...
    - class_path: rslearn.train.prediction_writer.RslearnWriter
      init_args:
        output_layer: output
        merger:
          class_path: rslearn.train.prediction_writer.RasterMerger
          init_args:
            # Discards the overlapping border so it matches the overlap_pixels we set
            # in predict_config, keeping the center 116x116 of each 128x128 output.
            overlap_pixels: 12
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
rslearn model predict --config land_cover_model.yaml --ckpt_path land_cover_model_checkpoints/best.ckpt
```

And visualize the Sentinel-2 image and output in qgis:

```
qgis $DATASET_PATH/windows/predict/portland/layers/*/*/geotiff.tif
```


## Defining Train and Validation Splits

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
import os

import tqdm
from rslearn.dataset import Dataset, Window
from upath import UPath

ds_path = UPath(os.environ["DATASET_PATH"])
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
  groups: ["default"]
  transforms:
    - class_path: rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize
      init_args:
        band_names:
          sentinel2_l2a: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
train_config:
  transforms:
    - class_path: rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize
      init_args:
        band_names:
          sentinel2_l2a: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
    - class_path: rslearn.train.transforms.flip.Flip
      init_args:
        image_selectors: ["sentinel2_l2a", "target/classes", "target/valid"]
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
  crop_size: 128
  overlap_pixels: 12
```

The `tags` option that we are adding here tells rslearn to only load windows with a
matching key and value in the window options.

Previously when we run `model fit`, it should show the same number of windows for
training and validation:

```
got 110 examples in split train
got 110 examples in split val
```

With the updates, it should show different numbers like this:

```
got 92 examples in split train
got 18 examples in split val
```


## Visualizing with `model test`

We can visualize the ground truth labels and model predictions in the test set using
the `model test` command:

```
mkdir ./vis
rslearn model test --config land_cover_model.yaml --ckpt_path land_cover_model_checkpoints/best.ckpt --model.init_args.visualize_dir=./vis/
```

This will produce PNGs in the vis directory. The visualizations are produced by the
`Task.visualize` function, so we could customize the visualization by subclassing
SegmentationTask and overriding the visualize function.


## Checkpoint and Logging Management

Above, we needed to configure the checkpoint directory in the model config (the
`dirpath` option under `rslearn.train.callbacks.checkpointing.BestLastCheckpoint`), and explicitly
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
  callbacks:
    # ...
    - class_path: rslearn.train.callbacks.checkpointing.ManagedBestLastCheckpoint
      init_args:
        monitor: val_accuracy
        mode: max
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
        enable_miou_metric: true
        enable_f1_metric: true
```

When calling `model test` and `model predict` with management_dir set, rslearn will
automatically load the best checkpoint from the project directory, or raise an error if
no existing checkpoint exists. This behavior can be overridden with the
`--load_checkpoint_mode` and `--load_checkpoint_required` options (see `--help` for
details). Logging will be enabled during fit but not test/predict, and this can also
be overridden, using `--log_mode`.


## Inputting Multiple Sentinel-2 Images

Currently our model inputs a single Sentinel-2 image. However, for most tasks where
labels are not expected to change from week to week, we find that accuracy can be
significantly improved by inputting multiple images. Multiple images make the model
more resilient to clouds and image artifacts, and allow the model to synthesize
information across different views that may come from different seasons or weather
conditions. OlmoEarth accepts an image time series directly, so we do not need to change
the model architecture.

We first update our dataset configuration to obtain three images per window, by
customizing the `query_config` section of the `sentinel2` layer:

```jsonc
"sentinel2": {
  "type": "raster",
  "band_sets": [
    {"dtype": "uint8", "bands": ["R", "G", "B"]},
    {"dtype": "uint16", "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]}
  ],
  "data_source": {
    "class_path": "rslearn.data_sources.planetary_computer.Sentinel2",
    "init_args": {
      "cache_dir": "cache/planetary_computer",
      "harmonize": true,
      "sort_by": "eo:cloud_cover"
    },
    "ingest": false,
    "query_config": {
      "max_matches": 3,
      "period_duration": "30d",
      "space_mode": "MOSAIC"
    }
  }
}
```

With `max_matches: 3`, rslearn stores the additional images in layers named
`sentinel2.1` and `sentinel2.2` (alongside the original `sentinel2` layer). Repeat the
prepare and materialize steps to populate them.

Next, we update the data module so that the dataset stacks the three images into a time
series. The `load_all_layers` option stacks the rasters from all of the listed layers,
and ignores windows where any of them are missing.

```yaml
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    path: # ...
    inputs:
      sentinel2_l2a:
        data_type: "raster"
        layers: ["sentinel2", "sentinel2.1", "sentinel2.2"]
        bands: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
        passthrough: true
        dtype: FLOAT32
        load_all_layers: true
      targets:
        # ...
```

Now we can train an updated model:

```
rslearn model fit --config land_cover_model.yaml
```
