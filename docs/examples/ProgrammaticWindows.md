## Programmatically Creating Windows

In this example, we show how to create windows programmatically. This is useful when the
`rslearn dataset add_windows` command does not offer sufficient flexibility.

### Table of Contents

1. [Quickstart: Create One Window](#quickstart-create-one-window)
2. [EuroSAT Conversion Example](#eurosat-conversion-example)
   1. [Convert EuroSAT to an rslearn Dataset](#convert-eurosat-to-an-rslearn-dataset)
   2. [Fine-tune OlmoEarth on EuroSAT](#fine-tune-olmoearth-on-eurosat)
   3. [Test with More Sentinel-2 Images](#test-with-more-sentinel-2-images)

## Quickstart: Create One Window

Here is a quick example of creating a window of a certain size centered at a longitude
and latitude, to show what the API looks like.

Create a folder `./dataset` to store the rslearn dataset, and populate
`./dataset/config.json` with this dataset configuration file:

```json
{
  "layers": {
    "sentinel2": {
      "band_sets": [{
          "bands": ["R", "G", "B"],
          "dtype": "uint8"
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
    }
  }
}
```

This will obtain one Sentinel-2 L2A image from Microsoft Planetary Computer. It will
only get the 8-bit R, G, and B bands from the true-color image product.

Here is how to create a window from Python:

```python
from datetime import datetime, UTC

import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from upath import UPath

# We start with a point in Seattle.
lon, lat = -122.33, 47.62

# Get an appropriate UTM projection for this location, with 10 m/pixel resolution.
# Note that a rasterio CRS specifies the coordinate reference system, while the rslearn
# Projection includes the projection-units-per-pixel resolution as well.
utm_projection = get_utm_ups_projection(lon, lat, 10, -10)

# Convert the longitude and latitude to UTM.
# WGS84_PROJECTION is equivalent to Projection(CRS.from_epsg(4326), 1, 1).
src_geom = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None)
dst_geom = src_geom.to_projection(utm_projection)

# Now compute integer pixel bounds from the resulting geometry, for a 512x512 window.
bounds = (
    int(dst_geom.shp.x) - 256,
    int(dst_geom.shp.y) - 256,
    int(dst_geom.shp.x) + 256,
    int(dst_geom.shp.y) + 256,
)

# And we can create the window.
ds_path = UPath("./dataset")
window = Window(
    # The window path is determined based on the dataset path, window group, and
    # window name.
    path=Window.get_window_root(ds_path, "default", "my_window"),
    group="default",
    name="my_window",
    projection=utm_projection,
    bounds=bounds,
    # We set the window's time range to summer so the image we get won't be cloudy.
    time_range=(
        datetime(2024, 6, 1, tzinfo=UTC),
        datetime(2024, 9, 1, tzinfo=UTC),
    ),
)
window.save()
```

Now we can materialize the image and visualize it in qgis:

```
rslearn dataset prepare --root ./dataset
rslearn dataset materialize --root ./dataset
qgis ./dataset/windows/default/my_window/layers/sentinel2/R_G_B/geotiff.tif
```

## EuroSAT Conversion Example

We now walk through a more advanced example where we convert the
[EuroSAT dataset](https://zenodo.org/records/7711810#.ZAm3k-zMKEA)
into an rslearn dataset. EuroSAT is a land use and land cover classification dataset
based on Sentinel-2 satellite imagery.

### Convert EuroSAT to an rslearn Dataset

We will assume the multispectral version of the dataset has been downloaded and
extracted, so `./EuroSAT_MS` contains one subfolder per category, and each of those
subfolders contain GeoTIFFs of Sentinel-2 images.

```
wget https://zenodo.org/records/7711810/files/EuroSAT_MS.zip
unzip EuroSAT_MS.zip
```

First, create a folder for the rslearn dataset (e.g., `./dataset`) and create the
dataset configuration file:

```python
{
  "layers": {
    "label": {
      "type": "vector"
    },
    "sentinel2": {
      "band_sets": [
        {
          "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"],
          "dtype": "uint16"
        }
      ],
      "type": "raster"
    }
  }
}
```

Note that we do not specify a data source for the sentinel2 layer above since we will
be populating it programmatically from the EuroSAT dataset rather than materializing
the data using rslearn. The band order above corresponds to the band order in EuroSAT.

Now we can convert the data. We start by enumerating the examples:

```python
from upath import UPath
eurosat_path = UPath("./EuroSAT_MS")
examples = []
for category_dir in eurosat_path.iterdir():
    for tif_fname in category_dir.iterdir():
        examples.append((tif_fname, category_dir.name))
```

We convert each example to an rslearn window. See the comments below for details on the
steps that we take.

```python
import hashlib
from datetime import datetime, timezone

import rasterio
import tqdm
from rslearn.dataset import Window
from rslearn.utils.feature import Feature
from rslearn.utils.raster_format import get_raster_projection_and_bounds, GeotiffRasterFormat
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.vector_format import GeojsonVectorFormat

# This is the path to the output rslearn dataset.
ds_path = UPath("./dataset")

# Iterate over the training examples.
for tif_fname, category in tqdm.tqdm(examples):
    with rasterio.open(tif_fname) as raster:
        # Get the projection and bounds based on the GeoTIFF.
        projection, bounds = get_raster_projection_and_bounds(raster)
        # We also read the satellite image data here.
        array = raster.read()

    # We name the window based on the GeoTIFF filename. We assign a train or val split
    # based on the last digit in the filename.
    window_name = tif_fname.name.split(".")[0]
    split = "val" if window_name[-1] in ["0", "1"] else "train"

    # Now we can create the window.
    window = Window(
        # The window path is determined based on the dataset path, window group, and
        # window name.
        path=Window.get_window_root(ds_path, "default", window_name),
        group="default",
        name=window_name,
        # The projection (which specifies CRS and resolution) and pixel bounds are set
        # based on the values extracted from the GeoTIFF.
        projection=projection,
        bounds=bounds,
        # This time range corresponds to the year that EuroSAT was released.
        time_range=(
            datetime(2018, 1, 1, tzinfo=timezone.utc),
            datetime(2019, 1, 1, tzinfo=timezone.utc),
        ),
        options={
            "split": split,
        }
    )
    window.save()

    # We manually populate the sentinel2 layer with the satellite image content.
    raster_dir = window.get_raster_dir("sentinel2", ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"])
    # The projection and bounds here are used to set the georeference metadata in the
    # GeoTIFF.
    GeotiffRasterFormat().encode_raster(raster_dir, projection, bounds, RasterArray(chw_array=array))
    window.mark_layer_completed("sentinel2")

    # We manually populate the label layer with a single GeoJSON feature corresponding
    # to the window geometry, which has a property specifying the category.
    feature = Feature(window.get_geometry(), {
        "category": category,
    })
    layer_dir = window.get_layer_dir("label")
    GeojsonVectorFormat().encode_vector(layer_dir, [feature])
    window.mark_layer_completed("label")
```

### Fine-tune OlmoEarth on EuroSAT

Now that we have the rslearn dataset, we can easily fine-tune remote sensing foundation
models like OlmoEarth model on it.

Here is a model configuration for OlmoEarth:

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.singletask.SingleTaskModel
      init_args:
        encoder:
          # For the encoder, we apply OlmoEarth. It computes one feature map at 1/8 the
          # input resolution, with embedding size 768.
          - class_path: rslearn.models.olmoearth_pretrain.model.OlmoEarth
            init_args:
              model_id: "OLMOEARTH_V1_BASE"
              patch_size: 8
        decoder:
          # For the decoder, we first apply PoolingDecoder which applies spatial
          # max pooling to get a flat 768 embedding, and then applies one fully
          # connected layer with ReLU activation (outputting 128 features) followed by
          # an output layer (outputting logits for the 10 EuroSAT classes).
          - class_path: rslearn.models.pooling_decoder.PoolingDecoder
            init_args:
              in_channels: 768
              num_fc_layers: 1
              fc_channels: 128
              out_channels: 10
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
    # This should be set to the path of the rslearn dataset.
    path: ./dataset
    inputs:
      # We load both layers from the dataset.
      # OlmoEarth expects the Sentinel-2 image to be called "sentinel2_l2a" in the
      # input dict.
      sentinel2_l2a:
        data_type: "raster"
        layers: ["sentinel2"]
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
        classes: ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
        metric_kwargs:
          average: "micro"
    batch_size: 16
    num_workers: 32
    default_config:
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
trainer:
  max_epochs: 100
  callbacks:
    # It is recommended to freeze the OlmoEarth encoder for the first few epochs.
    - class_path: rslearn.train.callbacks.freeze_unfreeze.FreezeUnfreeze
      init_args:
        module_selector: ["model", "encoder", 0]
        unfreeze_at_epoch: 10
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

Save this as `model.yaml`.

Now execute training with `model fit`:

```
# These environment variables are only needed if including the WandbLogger in the model
# config file.
export PROJECT_NAME=eurosat
export RUN_NAME=eurosat_00
export MANAGEMENT_DIR=./project_data
rslearn model fit --config model.yaml
```

### Test with More Sentinel-2 Images

Using a satellite image time series often improves performance. To this end, we can
experiment with applying OlmoEarth to predict EuroSAT categories with four input
images. EuroSAT only provides one image, so we need to materialize the image time
series using rslearn.

Update the dataset configuration file with a new layer. We will call it "sentinel2_ts",
and it downloads up to four Sentinel-2 L2A mosaics from Microsoft Planetary Computer.
rslearn will create the mosaics by stitching together individual Sentinel-2 scenes
until together they cover the window bounds, and the `sort_by` option ensures that
rslearn will add scenes starting with the least cloudy ones. Only scenes captured
within the time range of our windows (which we specified as January to December 2018
when converting the dataset) will be used.

```json
{
  "layers": {
    // ...
    "sentinel2_ts": {
      "band_sets": [
        {
          "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"],
          "dtype": "uint16"
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
          "max_matches": 4,
          "space_mode": "MOSAIC"
        }
      },
      "type": "raster"
    }
  }
}
```

We then make an update to `model.yaml` to input the time series: in the inputs section,
we use sentinel2_ts instead of sentinel2, and load all four timesteps:

```yaml
data:
  # ...
  init_args:
    # ...
    inputs:
      sentinel2_l2a:
        data_type: "raster"
        layers: ["sentinel2_ts", "sentinel2_ts.1", "sentinel2_ts.2", "sentinel2_ts.3"]
        bands: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
        passthrough: true
        load_all_layers: true
      targets:
        data_type: "vector"
        layers: ["label"]
        is_target: true
    # ...
```

We can train the updated model:

```
export WANDB_NAME=eurosat_ts_00
rslearn model fit --config model.yaml
```

In our experiments, the single-image model (using the image provided by EuroSAT)
achieves 98.1% accuracy, while our four-image model achieves 98.6% accuracy.
