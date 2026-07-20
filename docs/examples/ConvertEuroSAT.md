# Convert EuroSAT to an rslearn Dataset

This tutorial converts the
[EuroSAT dataset](https://zenodo.org/records/7711810#.ZAm3k-zMKEA) into an
rslearn dataset and fine-tunes OlmoEarth for land-cover classification. EuroSAT is a
Sentinel-2 image dataset with one class label per image.

The tutorial is intended to provide examples of programmatically creating windows, and
then writing raster and vector data to each window.

## Download EuroSAT

Download and extract the multispectral dataset. The resulting `EuroSAT_MS` directory
contains one subdirectory per category and one GeoTIFF per example.

```bash
wget https://zenodo.org/records/7711810/files/EuroSAT_MS.zip
unzip EuroSAT_MS.zip
```

## Configure the Dataset

Create `dataset/config.json`:

```json
{
  "layers": {
    "label": {
      "type": "vector"
    },
    "sentinel2": {
      "type": "raster",
      "band_sets": [
        {
          "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"],
          "dtype": "uint16"
        }
      ]
    }
  }
}
```

We do not specify a data source for either layer because we will programmatically write
both of them.

## Convert the Data

The script below creates one window per EuroSAT GeoTIFF. The source raster determines
the window projection and bounds. A stable hash assigns each example to the train or
validation split, and the source category becomes a vector property.

```python
import hashlib
from datetime import UTC, datetime

import rasterio
import tqdm
from upath import UPath

from rslearn.dataset import Dataset, Window
from rslearn.utils.feature import Feature
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import (
    GeotiffRasterFormat,
    get_raster_projection_and_bounds,
)
from rslearn.utils.vector_format import GeojsonVectorFormat

source_path = UPath("./EuroSAT_MS")
dataset = Dataset(UPath("./dataset"))
examples = [
    (tif_path, category_path.name)
    for category_path in source_path.iterdir()
    for tif_path in category_path.iterdir()
    if tif_path.suffix.lower() in {".tif", ".tiff"}
]

sentinel2_bands = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B09", "B10", "B11", "B12", "B8A",
]
for tif_path, category in tqdm.tqdm(examples):
    with rasterio.open(tif_path) as source:
        projection, bounds = get_raster_projection_and_bounds(source)
        array = source.read()

    window_name = tif_path.stem
    digest = hashlib.sha256(window_name.encode()).hexdigest()
    split = "val" if digest[0] in {"0", "1", "2"} else "train"
    window = Window(
        storage=dataset.storage,
        group="default",
        name=window_name,
        projection=projection,
        bounds=bounds,
        time_range=(
            datetime(2018, 1, 1, tzinfo=UTC),
            datetime(2019, 1, 1, tzinfo=UTC),
        ),
        options={"split": split},
        data_factory=dataset.window_data_storage_factory,
    )
    window.save()

    with window.data.open_layer_writer("sentinel2") as writer:
        writer.write_raster(
            sentinel2_bands,
            GeotiffRasterFormat(),
            projection,
            bounds,
            RasterArray(chw_array=array),
        )
    window.mark_layer_completed("sentinel2")

    feature = Feature(window.get_geometry(), {"category": category})
    with window.data.open_layer_writer("label") as writer:
        writer.write_vector(GeojsonVectorFormat(), [feature])
    window.mark_layer_completed("label")
```

## Fine-tune OlmoEarth

Save the model configuration below as `model.yaml`. It adds a decoder after the
OlmoEarth-v1.2-Base model, and configures it to input the Sentinel-2 image and train
against the classification labels.

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.singletask.SingleTaskModel
      init_args:
        # This section specifies the model architecture. We pair the
        # OlmoEarth-v1.2-Base encoder with a decoder that applies max pooling
        # to get down to a feature vector, and then applies two fully connected
        # layers to obtain classification logits.
        encoder:
          - class_path: rslearn.models.olmoearth_pretrain.model.OlmoEarth
            init_args:
              model_id: OLMOEARTH_V1_2_BASE
              patch_size: 8
        decoder:
          - class_path: rslearn.models.pooling_decoder.PoolingDecoder
            init_args:
              in_channels: 768
              num_fc_layers: 1
              fc_channels: 128
              out_channels: 10
          # The ClassificationHead will apply a softmax and compute cross
          # entropy loss.
          - class_path: rslearn.train.tasks.classification.ClassificationHead
    optimizer:
      class_path: rslearn.models.olmoearth_pretrain.optimizer.LayerDecayAdamW
      init_args:
        lr: 0.0001
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    path: ./dataset
    inputs:
      # We input both the Sentinel-2 images and the vector class IDs from the
      # dataset. The "sentinel2_l2a" name and band order matches what is
      # expected by the OlmoEarthNormalize transform and OlmoEarth model.
      sentinel2_l2a:
        data_type: raster
        layers: [sentinel2]
        bands: [B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09]
        passthrough: true
      targets:
        data_type: vector
        layers: [label]
        is_target: true
    task:
      # We train with ClassificationTask since it is window level
      # classification.
      class_path: rslearn.train.tasks.classification.ClassificationTask
      init_args:
        property_name: category
        classes: [AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake]
        metric_kwargs:
          # Use micro accuracy so that the computed accuracy is just the
          # fraction of correctly classified windows.
          average: micro
    batch_size: 16
    num_workers: 32
    default_config:
      transforms:
        - class_path: rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize
          init_args:
            band_names:
              sentinel2_l2a: [B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09]
    train_config:
      tags:
        split: train
    val_config:
      tags:
        split: val
trainer:
  max_epochs: 100
  callbacks:
    - class_path: rslearn.train.callbacks.checkpointing.ManagedBestLastCheckpoint
      init_args:
        monitor: val_accuracy
        mode: max
project_name: ${PROJECT_NAME}
run_name: ${RUN_NAME}
management_dir: ${MANAGEMENT_DIR}
```

Run training:

```bash
export PROJECT_NAME=eurosat
export RUN_NAME=eurosat_00
export MANAGEMENT_DIR=./project_data
rslearn model fit --config model.yaml
```

## Add a Sentinel-2 Time Series

EuroSAT provides one image per example. To compare it with a multi-image input, add a
retrieved layer to `dataset/config.json`:

```json
{
  "layers": {
    "sentinel2_ts": {
      "type": "raster",
      "band_sets": [
        {
          "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"],
          "dtype": "uint16"
        }
      ],
      "data_source": {
        "class_path": "rslearn.data_sources.planetary_computer.Sentinel2",
        "init_args": {
          "harmonize": true,
          "sort_by": "eo:cloud_cover"
        },
        "ingest": false,
        "query_config": {
          "max_matches": 4,
          "space_mode": "MOSAIC"
        }
      }
    }
  }
}
```

Run `prepare` and `materialize` for the new layer:

```bash
rslearn dataset prepare --root ./dataset --enabled-layers sentinel2_ts
rslearn dataset materialize --root ./dataset --enabled-layers sentinel2_ts
```

Then replace the `sentinel2_l2a` input in `model.yaml` with:

```yaml
sentinel2_l2a:
  data_type: raster
  layers: [sentinel2_ts, sentinel2_ts.1, sentinel2_ts.2, sentinel2_ts.3]
  bands: [B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09]
  passthrough: true
  load_all_layers: true
```

Train the time-series variant under a new run name:

```bash
export RUN_NAME=eurosat_ts_00
rslearn model fit --config model.yaml
```
