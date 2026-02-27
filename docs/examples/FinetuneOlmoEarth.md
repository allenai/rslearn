## Fine-tune OlmoEarth

In this example, we use rslearn to fine-tune [OlmoEarth](https://github.com/allenai/olmoearth_pretrain)
for segmenting land cover and crop type categories. We use labels from the USDA
Cropland Data Layer. For the inputs, we use four Sentinel-2 images (one per month).

If you are new to rslearn, you may want to read [the main README](../../README.md) or
[CoreConcepts](../CoreConcepts.md) first.

## Create the Dataset

Create a folder like `./dataset` to store the rslearn dataset, and save this dataset
configuration file to `./dataset/config.json`:

```json
{
  "layers": {
    "cdl": {
      "band_sets": [{
          "bands": ["cdl"],
          "dtype": "uint8"
      }],
      "data_source": {
        "class_path": "rslearn.data_sources.usda_cdl.CDL"
      },
      "resampling_method": "nearest",
      "type": "raster"
    },
    "sentinel2_l2a": {
      "band_sets": [{
          "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
          "dtype": "uint16"
      }],
      "data_source": {
        "class_path": "rslearn.data_sources.planetary_computer.Sentinel2",
        "init_args": {
          "cache_dir": "cache/planetary_computer",
          "harmonize": true,
          "sort_by": "eo:cloud_cover"
        },
        "ingest": false,
        "query_config": {
          "max_matches": 4,
          "period_duration": "30d",
          "space_mode": "PER_PERIOD_MOSAIC"
        }
      },
      "type": "raster"
    },
    "output": {
      "band_sets": [{
        "bands": ["output"],
        "dtype": "uint8"
      }],
      "type": "raster"
    }
  }
}
```

This tells rslearn to obtain two types of data: CDL data from USDA, and Sentinel-2 L2A
images captured by ESA and released on Microsoft Planetary Computer. The "output" layer
will be used later to store prediction outputs.

Next, we create windows to serve as training and validation examples. Here, we create
windows at a few locations around Seattle. A more diverse dataset would likely improve
performance.

```
rslearn dataset add_windows --root ./dataset --group default --utm --resolution 10 --src_crs EPSG:4326 --box=-122.4,47.6,-122.3,47.7 --start 2024-05-01T00:00:00+00:00 --end 2024-09-01T00:00:00+00:00 --grid_size 128
rslearn dataset add_windows --root ./dataset --group default --utm --resolution 10 --src_crs EPSG:4326 --box=-122.0,47.8,-121.9,47.9 --start 2024-05-01T00:00:00+00:00 --end 2024-09-01T00:00:00+00:00 --grid_size 128
rslearn dataset add_windows --root ./dataset --group default --utm --resolution 10 --src_crs EPSG:4326 --box=-122.1,47.6,-122.0,47.7 --start 2024-05-01T00:00:00+00:00 --end 2024-09-01T00:00:00+00:00 --grid_size 128
rslearn dataset add_windows --root ./dataset --group default --utm --resolution 10 --src_crs EPSG:4326 --box=-122.8,47.7,-122.7,47.8 --start 2024-05-01T00:00:00+00:00 --end 2024-09-01T00:00:00+00:00 --grid_size 128
rslearn dataset add_windows --root ./dataset --group default --utm --resolution 10 --src_crs EPSG:4326 --box=-122.6,47.6,-122.5,47.7 --start 2024-05-01T00:00:00+00:00 --end 2024-09-01T00:00:00+00:00 --grid_size 128
rslearn dataset add_windows --root ./dataset --group default --utm --resolution 10 --src_crs EPSG:4326 --box=-123.2,47.2,-123.1,47.3 --start 2024-05-01T00:00:00+00:00 --end 2024-09-01T00:00:00+00:00 --grid_size 128
rslearn dataset add_windows --root ./dataset --group default --utm --resolution 10 --src_crs EPSG:4326 --box=-122.1,47.2,-122.0,47.3 --start 2024-05-01T00:00:00+00:00 --end 2024-09-01T00:00:00+00:00 --grid_size 128
```

Use rslearn to materialize the data:

```
rslearn dataset prepare --root ./dataset --workers 32 --retry-max-attempts 5 --retry-backoff-seconds 5
rslearn dataset ingest --root ./dataset --workers 32
rslearn dataset materialize --root ./dataset --workers 32 --retry-max-attempts 5 --retry-backoff-seconds 5 --ignore-errors
```

Split the data into train and val:

```python
import hashlib
import tqdm
from rslearn.dataset import Dataset, Window
from upath import UPath

ds_path = UPath("./dataset")
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

## Fine-tune the Model

We will use this model configuration file for fine-tuning OlmoEarth. See the comments
below for details on how each part functions. [The OlmoEarth rslearn reference](../foundation_models/OlmoEarth.md)
also has more information.

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.singletask.SingleTaskModel
      init_args:
        encoder:
          # This applies the OlmoEarth encoder on the inputs.
          - class_path: rslearn.models.olmoearth_pretrain.model.OlmoEarth
            init_args:
              # This can be one of:
              # - OLMOEARTH_V1_NANO
              # - OLMOEARTH_V1_TINY
              # - OLMOEARTH_V1_BASE
              # - OLMOEARTH_V1_LARGE
              model_id: OLMOEARTH_V1_BASE
              # The patch size should be set between 1 and 8, depending on the size of
              # the features being predicted, and the available compute (lower patch
              # sizes are slower).
              patch_size: 8
        decoder:
          # For the decoder, we apply UNetDecoder to upsample features to the input
          # resolution. It isn't really a UNet since we only have features at one
          # resolution.
          - class_path: rslearn.models.unet.UNetDecoder
            init_args:
              in_channels: [[8, 768]]
              out_channels: 256
              conv_layers_per_resolution: 2
              num_channels: {8: 512, 4: 512, 2: 256, 1: 128}
          # The SegmentationHead computes softmax and cross entropy loss.
          - class_path: rslearn.train.tasks.segmentation.SegmentationHead
    optimizer:
      class_path: rslearn.train.optimizer.AdamW
      init_args:
        lr: 0.0001
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    path: ./dataset
    inputs:
      # We read the four Sentinel-2 images materialized earlier.
      # OlmoEarth expects the input to be called "sentinel2_l2a".
      sentinel2_l2a:
        data_type: "raster"
        layers: ["sentinel2_l2a", "sentinel2_l2a.1", "sentinel2_l2a.2", "sentinel2_l2a.3"]
        # This is the band order expected by  OlmoEarth.
        bands: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
        passthrough: true
        dtype: FLOAT32
        load_all_layers: true
      # We also read the CDL data here. SegmentationTask expects the target to be
      # called "targets".
      targets:
        data_type: "raster"
        layers: ["cdl"]
        bands: ["cdl"]
        dtype: FLOAT32
        is_target: true
    task:
      class_path: rslearn.train.tasks.segmentation.SegmentationTask
      init_args:
        num_classes: 256
        enable_miou_metric: true
    batch_size: 8
    num_workers: 32
    default_config:
      groups: ["default"]
      transforms:
        - class_path: rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize
          init_args:
            band_names:
              sentinel2_l2a: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
    train_config:
      tags:
        split: "train"
    val_config:
      tags:
        split: "val"
    test_config:
      tags:
        split: "val"
    predict_config:
      groups: ["predict"]
      load_all_crops: true
      # We set crop_size=128 here to support the option of using larger windows during
      # prediction. Note that this controls the sliding window inference crop size,
      # and we want that to match the size of our training windows.
      crop_size: 128
      # We use some overlap when we need to apply sliding window inference on large
      # windows to reduce border effects.
      overlap_pixels: 12
      skip_targets: true
trainer:
  max_epochs: 100
  callbacks:
    # We find that freezing the model for the first few epochs helps to improve the
    # performance of the fine-tuned models.
    - class_path: rslearn.train.callbacks.freeze_unfreeze.FreezeUnfreeze
      init_args:
        module_selector: ["model", "encoder", 0]
        unfreeze_at_epoch: 10
    # The RslearnWriter is used during `model predict` to save the predicted outputs to
    # the rslearn dataset.
    - class_path: rslearn.train.prediction_writer.RslearnWriter
      init_args:
        output_layer: output
        merger:
          class_path: rslearn.train.prediction_writer.RasterMerger
          init_args:
            # This removes the boundary that is redundant because of the overlap_pixels.
            # So we keep the center 116x116 of each 128x128 output, since there are
            # 12 pixels of overlap between adjacent inference crops.
            overlap_pixels: 12
    # Save best checkpoint based on accuracy metric.
    - class_path: rslearn.train.callbacks.checkpointing.ManagedBestLastCheckpoint
      init_args:
        monitor: val_accuracy
        mode: max
# Here we enable automatic checkpoint management and logging to W&B.
# Set WANDB_MODE=offline to disable online logging.
project_name: ${PROJECT_NAME}
run_name: ${RUN_NAME}
management_dir: ${MANAGEMENT_DIR}
```

Save this as `model.yaml` and then run `model fit`:

```
export PROJECT_NAME=olmoearth_cdl
export RUN_NAME=run_00
export MANAGEMENT_DIR=./project_data
rslearn model fit --config model.yaml
```

You should see loss and metric curves logged to W&B. The checkpoints will be stored in
the `MANAGEMENT_DIR`, and rerunning `model fit` should automatically resume from the
last saved checkpoint.

## Apply the Model

Similar to in the [main README](../../README.md), we can now apply the model on a new
window.

We create a big window northeast of Bellingham, WA:

```
rslearn dataset add_windows --root ./dataset --group predict --utm --resolution 10 --src_crs EPSG:4326 --box=-122.5,48.8,-122.3,49.0 --start 2024-05-01T00:00:00+00:00 --end 2024-09-01T00:00:00+00:00 --name bellingham
rslearn dataset prepare --root ./dataset --group predict
rslearn dataset materialize --root ./dataset --group predict
```

Then we apply the model (it will automatically restore the best checkpoint):

```
rslearn model predict --config model.yaml
```

We can then open up one of the input Sentinel-2 images, the model prediction, and the
actual CDL in qgis:

```
qgis dataset/windows/predict/bellingham2/layers/{cdl,output,sentinel2_l2a}/*/geotiff.tif
```
