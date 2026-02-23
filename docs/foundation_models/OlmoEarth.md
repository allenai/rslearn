## OlmoEarth

This component wraps the [OlmoEarth model](https://github.com/allenai/olmoearth_pretrain)
for fine-tuning in rslearn.

### Usage

OlmoEarth inputs time series of any subset of Sentinel-2 L2A, Sentinel-1 IW GRD vv+vh,
and Landsat 8/9 OLI-TIRS satellite images. It is recommended to use a number of
timesteps between 1 and 12, and an input size between 1 (single pixel) and 128. The
resolution should be 10 m/pixel.

The input should include at least one of these keys:

- "sentinel2_l2a": a Sentinel-2 L2A image (or time series stacked on channel axis). The
  pixel values should be 16-bit integers, and the band order is B02, B03, B04, B08,
  B05, B06, B07, B8A, B11, B12, B01, B09.
- "sentinel1": a Sentinel-1 IW GRD vv+vh image (or time series stacked on channel
  axis). The values should be converted to decibels, and the band order is vv, vh. We
  recommend using radiometrically terrain corrected images since that matches
  pre-training.
- "landsat": a Landsat 8/9 OLI-TIRS
  [Collection 2 Level-1](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-1-data)
  image (or time series stacked on channel axis). The pixel values should be 16-bit
  integers, and the band order is B8, B1, B2, B3, B4, B5, B6, B7, B9, B10, B11.

Here is a summary of the model arguments:

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.singletask.SingleTaskModel
      init_args:
        encoder:
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
              patch_size: 4
```

The output is a single feature map, with a resolution equal to `1/patch_size` of the
input resolution. The embedding size depends on the `model_id`:

- OLMOEARTH_V1_NANO: 128
- OLMOEARTH_V1_TINY: 192
- OLMOEARTH_V1_BASE: 768
- OLMOEARTH_V1_LARGE: 1024

## Examples

Here are some tutorials and examples of applying OlmoEarth.

- [FinetuneOlmoEarth](../examples/FinetuneOlmoEarth.md) shows how to fine-tune
  OlmoEarth on segmentation labels from the USDA Cropland Data Layer.
- [ProgrammaticWindows](../examples/ProgrammaticWindows.md) shows how to apply
  OlmoEarth on EuroSAT.
- In olmoearth_projects, see [Fine-tuning OlmoEarth for Classification](https://github.com/allenai/olmoearth_projects/blob/main/docs/tutorials/FinetuneOlmoEarthClassification.md).
  Note that this example uses olmoearth_run in addition to rslearn.
- You may also find the [model configuration files in olmoearth_projects](https://github.com/allenai/olmoearth_projects/tree/main/olmoearth_run_data)
  helpful. They pair with rslearn datasets that can be downloaded from Hugging Face.

## Data Inputs

Here is an example dataset configuration to obtain Sentinel-2 and Sentinel-1 images
from Microsoft Planetary Computer and Landsat images from AWS. Note that the AWS bucket
is requester pays so it would require credentials. We recommend starting with
Sentinel-2 or Sentinel-2 + Sentinel-1 since that seems to work well for most tasks. You
may need to adjust the `query_config` depending on your window time range.

```json
{
  "layers": {
    "landsat": {
      "band_sets": [{
          "bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"],
          "dtype": "uint16"
      }],
      "data_source": {
        "class_path": "rslearn.data_sources.aws_landsat.LandsatOliTirs",
        "init_args": {
          "metadata_cache_dir": "cache/landsat",
          "sort_by": "cloud_cover"
        },
        "ingest": false,
        "query_config": {
          "max_matches": 12,
          "period_duration": "30d",
          "space_mode": "PER_PERIOD_MOSAIC"
        }
      },
      "type": "raster"
    },
    "sentinel1": {
      "band_sets": [{
          "bands": ["vv", "vh"],
          "dtype": "float32",
          "nodata_vals": [-32768, -32768]
      }],
      "data_source": {
        "class_path": "rslearn.data_sources.planetary_computer.Sentinel1",
        "init_args": {
          "cache_dir": "cache/planetary_computer",
          "query": {
            "sar:instrument_mode": {"eq": "IW"},
            "sar:polarizations": {"eq": ["VV", "VH"]}
          }
        },
        "ingest": false,
        "query_config": {
          "max_matches": 12,
          "period_duration": "30d",
          "space_mode": "PER_PERIOD_MOSAIC"
        }
      },
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
          "max_matches": 12,
          "period_duration": "30d",
          "space_mode": "PER_PERIOD_MOSAIC"
        }
      },
      "type": "raster"
    }
  }
}
```

To best match pre-training, the window projection should be UTM and the resolution
should be 10 m/pixel.

With data materialized in this way, the data inputs and normalization transforms in the
model configuration file could be configured as follows. The number of item groups
under `layers` would depend on your window time range and `query_config`.

```yaml
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    inputs:
      # The keys here (corresponding to keys in the input dict) must exactly match with
      # those expected by the pre-trained model ("sentinel1", "sentinel2_l2a", and
      # "landsat").
      # See "class Modality" in https://github.com/allenai/olmoearth_pretrain/blob/main/olmoearth_pretrain/data/constants.py
      sentinel1:
        data_type: "raster"
        layers: ["sentinel1", "sentinel1.1", "sentinel1.2", "sentinel1.3", "sentinel1.4", "sentinel1.5", "sentinel1.6", "sentinel1.7", "sentinel1.8", "sentinel1.9", "sentinel1.10", "sentinel1.11"]
        bands: ["vv", "vh"]
        passthrough: true
        dtype: FLOAT32
        load_all_layers: true
      sentinel2_l2a:
        data_type: "raster"
        layers: ["sentinel2_l2a", "sentinel2_l2a.1", "sentinel2_l2a.2", "sentinel2_l2a.3"]
        bands: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
        passthrough: true
        dtype: FLOAT32
        load_all_layers: true
      landsat:
        data_type: "raster"
        layers: ["landsat", "landsat.1", "landsat.2", "landsat.3"]
        bands: ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"]
        passthrough: true
        dtype: FLOAT32
        load_all_layers: true
    default_config:
      transforms:
        # If you are using Sentinel-1 RTC from Planetary Computer, or other sources
        # where the pixel values are not already decibels, then make sure to convert to
        # decibels.
        - class_path: rslearn.train.transforms.sentinel1.Sentinel1ToDecibels
          init_args:
            selectors: ["sentinel1"]
        # Then apply OlmoEarth normalization.
        - class_path: rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize
          init_args:
            band_names:
              # Only include the modalities that you are using here, otherwise it will
              # raise an error. The input dict keys and band names must match exactly
              # with those in the normalization config dict in olmoearth_pretrain:
              # https://github.com/allenai/olmoearth_pretrain/blob/main/olmoearth_pretrain/data/norm_configs/computed.json
              sentinel1: ["vv", "vh"]
              sentinel2_l2a: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
              landsat: ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"]
```

## Model Architecture

Here is an example for classification:

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.singletask.SingleTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.olmoearth_pretrain.model.OlmoEarth
            init_args:
              model_id: OLMOEARTH_V1_BASE
              patch_size: 4
        decoder:
          - class_path: rslearn.models.pooling_decoder.PoolingDecoder
            init_args:
              in_channels: 768
              # Replace with the number of classes.
              out_channels: 10
              num_conv_layers: 2
              num_fc_layers: 2
          - class_path: rslearn.train.tasks.classification.ClassificationHead
```

Here is an example for object detection:

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.singletask.SingleTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.olmoearth_pretrain.model.OlmoEarth
            init_args:
              model_id: OLMOEARTH_V1_BASE
              patch_size: 4
        decoder:
          - class_path: rslearn.models.faster_rcnn.FasterRCNN
            init_args:
              downsample_factors: [4]
              num_channels: 768
              # Replace with the number of classes.
              num_classes: 10
              anchor_sizes: [[32]]
```

Here is an example for segmentation:

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.singletask.SingleTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.olmoearth_pretrain.model.OlmoEarth
            init_args:
              model_id: OLMOEARTH_V1_BASE
              patch_size: 4
        decoder:
          - class_path: rslearn.models.unet.UNetDecoder
            init_args:
              in_channels: [[4, 768]]
              # Replace with the number of classes.
              out_channels: 2
              conv_layers_per_resolution: 2
              # This limits the number of channels when the UNet is up-sampling to the
              # input resolution. Otherwise it will stick with 768 which may be too
              # large at the highest resolutions.
              num_channels: {4: 512, 2: 256, 1: 128}
          - class_path: rslearn.train.tasks.segmentation.SegmentationHead
```
