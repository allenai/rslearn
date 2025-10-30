## SatlasPretrain

This component wraps the [SatlasPretrain model](https://github.com/allenai/satlaspretrain_models)
for fine-tuning in rslearn.

The input should include a key "image" containing a single image. The number of
channels and expected bands and normalization depends on the model ID being used.

The Sentinel2_X_RGB models expect the R, G, and B bands (from the 8-bit true-color image
product) normalized to 0-1 by dividing by 255 and clipping. Here is an example:

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.multitask.SingleTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.satlaspretrain.SatlasPretrain
            init_args:
              model_identifier: Sentinel2_SwinB_SI_RGB
        decoder:
          # ...
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    inputs:
      image:
        data_type: "raster"
        layers: ["sentinel2"]
        bands: ["R", "G", "B"]
        passthrough: true
        dtype: FLOAT32
    default_config:
      transforms:
        - class_path: rslearn.train.transforms.normalize.Normalize
          init_args:
            mean: 0
            std: 255
            # This ensures the value after dividing by 255 is clipped to 0-1.
            valid_range: [0, 1]
```

The Sentinel2_X_MS expect the bands to be order and normalized as follows. This is also
an example of applying the model on an image time series using SimpleTimeSeries with
temporal max pooling, which is the recommended way:

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.multitask.SingleTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.simple_time_series.SimpleTimeSeries
            encoder:
              - class_path: rslearn.models.satlaspretrain.SatlasPretrain
                init_args:
                  model_identifier: Sentinel2_SwinB_SI_MS
              image_channels: 9
              op: "max"
        decoder:
          # ...
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    inputs:
      image:
        data_type: "raster"
        layers: ["sentinel2", "sentinel2.1", "sentinel2.2", "sentinel2.3"]
        bands: ["B04", "B03", "B02", "B05", "B06", "B07", "B08", "B11", "B12"]
        passthrough: true
        dtype: FLOAT32
        load_all_layers: true
    default_config:
      transforms:
        - class_path: rslearn.train.transforms.normalize.Normalize
          init_args:
            mean: 0
            std: 3000
            valid_range: [0, 1]
            bands: [0, 1, 2]
            num_bands: 9
        - class_path: rslearn.train.transforms.normalize.Normalize
          init_args:
            mean: 0
            std: 8160
            valid_range: [0, 1]
            bands: [3, 4, 5, 6, 7, 8]
            num_bands: 9
```

The Base models output feature maps like this, given batch size B, input height H, and
input width W:

```
[
  B x 128 x (H/4) x (W/4) tensor,
  B x 256 x (H/8) x (W/8) tensor,
  B x 512 x (H/16) x (W/16) tensor,
  B x 1024 x (H/32) x (W/32) tensor,
]
```

Here is an example for classification:

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.multitask.SingleTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.satlaspretrain.SatlasPretrain
            init_args:
              model_identifier: Sentinel2_SwinB_SI_RGB
        decoder:
          # This will use the last (lowest resolution) feature map.
          - class_path: rslearn.models.pooling_decoder.PoolingDecoder
            init_args:
              in_channels: 1024
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
      class_path: rslearn.models.multitask.SingleTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.satlaspretrain.SatlasPretrain
            init_args:
              model_identifier: Sentinel2_SwinB_SI_RGB
              fpn: true
        decoder:
          - class_path: rslearn.models.faster_rcnn.FasterRCNN
            init_args:
              downsample_factors: [4, 8, 16, 32]
              num_channels: 128
              num_classes: 10
              anchor_sizes: [[32], [64], [128], [256]]
```

Here is an example for segmentation:

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.multitask.SingleTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.satlaspretrain.SatlasPretrain
            init_args:
              model_identifier: Sentinel2_SwinB_SI_RGB
        decoder:
          - class_path: rslearn.models.unet.UNetDecoder
            init_args:
              in_channels: [[4, 128], [8, 256], [16, 512], [32, 1024]]
              out_channels: 2
              conv_layers_per_resolution: 2
          - class_path: rslearn.train.tasks.segmentation.SegmentationHead
```

### Previous Checkpoint Versions

The previous version of the SatlasPretrain checkpoints may offer higher performance
(https://github.com/allenai/satlas/issues/47). It
is recommended to use the MI checkpoints even for single-image tasks.

The MI_MS model can be loaded like this:

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.multitask.MultiTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.simple_time_series.SimpleTimeSeries
            init_args:
              encoder:
                class_path: rslearn.models.swin.Swin
                init_args:
                  pretrained: true
                  input_channels: 9
                  output_layers: [1, 3, 5, 7]
              image_channels: 9
    restore_config:
      restore_path: https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-model-v1-lowres-band-multi.pth
      remap_prefixes:
        - ["backbone.backbone.backbone.", "encoder.0.encoder.model."]
```

Here is an example of configuring the MI_RGB model without SimpleTimeSeries. The
`remap_prefixes` needs an update to reflect the new position of the Swin component in
the architecture.

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.multitask.MultiTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.swin.Swin
            init_args:
              pretrained: true
              input_channels: 3
              output_layers: [1, 3, 5, 7]
    restore_config:
      restore_path: https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-model-v1-lowres-multi.pth
      remap_prefixes:
        - ["backbone.backbone.backbone.", "encoder.0.model."]
```
