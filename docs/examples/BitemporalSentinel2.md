## Bi-temporal Sentinel-2 Example

In this example, we will acquire Sentinel-2 images from 2016 and 2024, and train a
model to predict which is earlier. For the model, we will apply OlmoEarth independently
on the two images, and concatenate the feature maps and pass them to a small decoder to
make the final prediction. We will use a custom transform to randomize the order of the
images.

There will be three steps:

1. Create the dataset.
2. Define the model architecture.
3. Implement the transform to randomize the image order.

## Create the Dataset

First, create a folder like `./bitemporal_dataset` and initialize it by saving this
dataset configuration file as `./bitemporal_dataset/config.json`:

```json
{
  "layers": {
    "old": {
      "band_sets": [{
          "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"],
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
        "time_offset": "-2920d"
      },
      "type": "raster"
    },
    "new": {
      "band_sets": [{
          "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"],
          "dtype": "uint16"
      }],
      "data_source": {
        "class_path": "rslearn.data_sources.planetary_computer.Sentinel2",
        "init_args": {
          "cache_dir": "cache/planetary_computer",
          "harmonize": true,
          "sort_by": "eo:cloud_cover"
        },
        "ingest": false
      },
      "type": "raster"
    },
    "label": {
      "type": "vector"
    }
  }
}
```

We will create windows with a time range from January to December 2024. Then, the
second layer will create a Sentinel-2 mosaic using the least cloudy images captured
within that time range. In the first layer, we set the `time_offset` option so that it
will create a similar mosaic, but for the time range shifted eight years earlier, i.e.
January to December 2016.

We also include a placeholder "label" layer that we will use to store classification
labels.

We focus on images of city centers so that there are more signals that the model can
use to make an accurate prediction. To this end, we have leveraged the
[SimpleMaps Prominent Cities Dataset](https://simplemaps.com/data/world-cities)
to derive GeoJSONs containing the 2,000 most populous cities (we randomly split them
between train and val sets). The data is released under Creative Commons Attribution
4.0 (see `examples/BitemporalSentinel2/simplemaps_license.txt`).

Then, we can use the GeoJSON to create windows in the dataset. We create a 128x128
pixel window centered at each city, at a resolution of 10 m/pixel (since that is the
resolution of Sentinel-2 images).

```
rslearn dataset add_windows --root ./bitemporal_dataset --group train --utm --resolution 10 --fname docs/examples/BitemporalSentinel2/train_cities.geojson --start 2024-01-01T00:00:00+00:00 --end 2025-01-01T00:00:00+00:00 --window_size 128
rslearn dataset add_windows --root ./bitemporal_dataset --group val --utm --resolution 10 --fname docs/examples/BitemporalSentinel2/val_cities.geojson --start 2024-01-01T00:00:00+00:00 --end 2025-01-01T00:00:00+00:00 --window_size 128
```

Now we can materialize the Sentinel-2 images:

```
rslearn dataset prepare --root ./bitemporal_dataset --workers 128 --retry-max-attempts 5 --retry-backoff-seconds 5
rslearn dataset materialize --root ./bitemporal_dataset --workers 128 --retry-max-attempts 5 --retry-backoff-seconds 5 --ignore-errors
```

You can visualize the 2016 and 2024 images for one of the cities in qgis:

```
qgis bitemporal_dataset/windows/train/56479_-234996_56607_-234868_2024-01-01T00:00:00+00:00_2025-01-01T00:00:00+00:00/layers/{old,new}/B01_B02_B03_B04_B05_B06_B07_B08_B09_B11_B12_B8A/geotiff.tif
```

### Using the Label Layer

We will train the model using [ClassificationTask](../TasksAndModels.md#classificationtask),
which expects to read a category name from the property of a GeoJSON feature. To make
our dataset compatible, we will set up the label layer with GeoJSONs that all have the
category set to "old_then_new", indicating that the old image appears first and the new
image second. Later, in our model configuration file, we will read the images in that
order, but we will implement a transform to randomly reverse the order, and we will set
it up so that the transform also reverses the category to "new_then_old" if it reverses
the image order.

Then, we can populate the label layer programmatically:

```python
import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.feature import Feature
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

# Iterate over all windows in the dataset.
ds_path = UPath("./bitemporal_dataset")
windows = Dataset(ds_path).load_windows()
for window in tqdm.tqdm(windows):
    # Create a GeoJSON feature with the category property.
    # The geometry doesn't matter for ClassificationTask, so we just use the window
    # geometry.
    feat = Feature(
        window.get_geometry(),
        {"category": "old_then_new"},
    )
    # Then write it to the label layer.
    layer_dir = window.get_layer_dir("label")
    GeojsonVectorFormat().encode_vector(layer_dir, [feat])
    window.mark_layer_completed("label")
```

## Define the Model Architecture

We develop a model configuration file that applies OlmoEarth on each image, and makes a
prediction using the concatenated features across the images:

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.singletask.SingleTaskModel
      init_args:
        encoder:
          # We wrap the OlmoEarth model in SimpleTimeSeries, which will apply a model
          # independently on each image in a time series.
          - class_path: rslearn.models.simple_time_series.SimpleTimeSeries
            init_args:
              encoder:
                # OlmoEarth-v1-Base will compute a feature map at 1/8 the input
                # resolution, with an embedding size of 768.
                class_path: rslearn.models.olmoearth_pretrain.model.OlmoEarth
                init_args:
                  model_id: "OLMOEARTH_V1_BASE"
                  patch_size: 8
              image_channels: 12
              # SimpleTimeSeries will apply max temporal pooling across images in the
              # same feature group, but concatenate across feature groups. Here, we
              # only want to concatenate the features across the two images, so we put
              # each image index in its own
              groups: [[0], [1]]
              image_key: sentinel2_l2a
        decoder:
          # PoolingDecoder will take the temporally concatenated feature map, and apply
          # a sequence of convolutional layers, spatial max pooling, and fully
          # connected layers to compute classification logits.
          - class_path: rslearn.models.pooling_decoder.PoolingDecoder
            init_args:
              # It inputs 1536 channels since we have 768 from each image.
              in_channels: 1536
              # We apply two conv layers on the concatenated features before spatial
              # pooling.
              num_conv_layers: 2
              conv_channels: 256
              # We also apply two fully connected layers after spatial pooling.
              num_fc_layers: 2
              fc_channels: 128
              # Then there is one final fully connected layer from 128 -> 2 classes.
              out_channels: 2
          # The ClassificationHead computes softmax cross entropy loss against the
          # ground truth category.
          - class_path: rslearn.train.tasks.classification.ClassificationHead
    optimizer:
      class_path: rslearn.train.optimizer.AdamW
      init_args:
        lr: 0.0001
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    path: ./bitemporal_dataset
    inputs:
      # "sentinel2_l2a" is the key for Sentinel-2 images expected by the OlmoEarth model.
      sentinel2_l2a:
        data_type: "raster"
        # As discussed above, we read the old image first, then the new image.
        # Later, we will implement a transform that can reverse the order.
        layers: ["old", "new"]
        load_all_layers: true
        # This is the order of bands expected by the OlmoEarth model.
        bands: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
        passthrough: true
      # ClassificationTask expects the labels to be called "target" in the input dict.
      targets:
        data_type: "vector"
        layers: ["label"]
        is_target: true
    task:
      class_path: rslearn.train.tasks.classification.ClassificationTask
      init_args:
        property_name: "category"
        classes: ["old_then_new", "new_then_old"]
        metric_kwargs:
          average: "micro"
        # image_bands and remap_values specify how images should be visualized during
        # `rslearn model test`.
        image_bands: [2, 1, 0]
        remap_values: [[-0.77, 0.67], [0, 255]]
    batch_size: 16
    num_workers: 32
    default_config:
      transforms:
        - class_path: rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize
          init_args:
            band_names:
              sentinel2_l2a: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
        # We will implement the ReversImageOrder class later!
        - class_path: ReverseImageOrder
    train_config:
      groups: ["train"]
    val_config:
      groups: ["val"]
    test_config:
      groups: ["val"]
trainer:
  max_epochs: 100
  callbacks:
    # Save both the latest checkpoint (last.ckpt) and the best one (epoch=....ckpt).
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        save_top_k: 1
        save_last: true
        monitor: val_accuracy
        mode: max
    # It is recommended to freeze the OlmoEarth encoder for the first few epochs.
    - class_path: rslearn.train.callbacks.freeze_unfreeze.FreezeUnfreeze
      init_args:
        module_selector: ["model", "encoder", 0]
        unfreeze_at_epoch: 10
# Here we enable automatic checkpoint management and logging to W&B.
# Set WANDB_MODE=offline to disable online logging.
project_name: ${PROJECT_NAME}
run_name: ${RUN_NAME}
management_dir: ${MANAGEMENT_DIR}
```

Save this as `model.yaml`.

See [TasksAndModels](../TasksAndModels.md) for more details about the SimpleTimeSeries
and OlmoEarth model components.

## Implement the ReverseImageOrder Transform

The only remaining step is to implement a transform that will reverse the order of the
images randomly. You can see the `rslearn.train.transforms` module for the Transform
API and examples of various built-in transforms.

Here is the `ReverseImageOrder` transform:

```python
import random

import torch
from rslearn.train.transforms.transform import Transform

class ReverseImageOrder(Transform):
    def forward(
        self, input_dict: dict, target_dict: dict
    ) -> tuple[dict, dict]:
        # Randomly decide whether reverse the order.
        if random.random() < 0.5:
            # Do nothing.
            return input_dict, target_dict

        # input_dict["sentinel2_l2a"] will contain the old and new images stacked on
        # the channel axis. So we just need to reverse them.
        assert input_dict["sentinel2_l2a"].shape[0] == 24
        input_dict["sentinel2_l2a"] = torch.cat([
            input_dict["sentinel2_l2a"][12:24],
            input_dict["sentinel2_l2a"][0:12],
        ])

        # We also reverse the classification label.
        target_dict["class"] = torch.tensor(1, dtype=torch.int64)
        return input_dict, target_dict
```

The input dict contains the passthrough inputs, while the target dict is computed by
the task based on the provided labels. [Transforms.md](../Transforms.md) provides more
details about this data loading process.

Above, we access known keys where the image is located in the input, and where the
target class ID has been stored by ClassificationTask. Note that we randomly reverse
the order both for training and for validation, so each validation epoch will see a
different order.

Save this as `bitemporal_train.py` with an entrypoint to run rslearn:

```python
import random

import torch
from rslearn.main import main
from rslearn.train.transforms.transform import Transform

class ReverseImageOrder(Transform):
    # ...

if __name__ == "__main__":
    main()
```

Finally, we can train the model:

```
export PROJECT_NAME=bitemporal_sentinel2
export RUN_NAME=model_00
export MANAGEMENT_DIR=./project_data/
python bitemporal_train.py model fit --config model.yaml
```

The model achieves unrealistically accuracy (98%) which suggests there may be a shift
in the satellite images that the model is using to "cheat".
