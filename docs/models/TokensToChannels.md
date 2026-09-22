## rslearn.models.tokens_to_channels.TokensToChannels

TokensToChannels applies a shared linear projection to each token of a
TokenFeatureMaps and lays the projected tokens out as channels.

It inputs a TokenFeatureMaps, where each feature map is a `B x C x H x W x N` tensor
with N tokens per spatial location. A linear layer `C -> out_dim` is applied to every
token independently, and the results are concatenated along the channel dimension in
token order. The output is a FeatureMaps where each feature map is
`B x (N * out_dim) x H x W`; the channels `n * out_dim` to `(n + 1) * out_dim - 1`
correspond to token `n`.

With `out_dim: 1` (the default), this turns per-timestep tokens into one logit per
timestep at each location. This is useful for predicting the timestep at which an
event occurred as an N-way classification at each pixel: the output feature map can be
passed directly to [SegmentationHead](SegmentationHead.md) with a
[SegmentationTask](../TasksAndModels.md#segmentationtask) that has `num_classes` equal
to the number of timesteps.

If the input TokenFeatureMaps has masks, the output shape is unchanged, but the
`out_dim` channels corresponding to each invalid token are overwritten with
`mask_fill_value` (0 by default). With `out_dim: 1` and a SegmentationHead, you may
want to set `mask_fill_value` to a large negative number so that padded timesteps are
never predicted.

### Configuration

```yaml
        decoder:
          - class_path: rslearn.models.tokens_to_channels.TokensToChannels
            init_args:
              # The token embedding dimension C.
              in_dim: 768
              # The number of output values per token. The default is 1.
              out_dim: 1
              # The value written to the output channels of tokens that are
              # marked invalid by the input mask (if any). The default is 0.
              mask_fill_value: 0.0
```

### Example

This example predicts, at each pixel, the month in which a change occurred, given a
year of monthly Sentinel-2 mosaics. The label raster contains the index of the month
(0 to 11) at pixels where a change occurred, and 255 elsewhere.

The [OlmoEarth](../foundation_models/OlmoEarth.md) encoder is configured with
`token_pooling: false`, so instead of pooling over timesteps it outputs a
TokenFeatureMaps with one token per timestep. We use an OlmoEarth v1.2 model, which
has a single band set per modality, and pass only one modality (Sentinel-2), so the N
tokens are exactly the T timesteps in chronological order. With OlmoEarth v1 models
(which tokenize Sentinel-2 into three band sets) or when passing multiple modalities,
N would instead be the number of timesteps times the number of band sets summed over
modalities, so the tokens would no longer correspond one-to-one with timesteps.

Every window should contain the same number of Sentinel-2 layers so that N is fixed
across batches (the number of output channels must match `num_classes`). Within a
batch, if a window has fewer images, OlmoEarth pads the token dimension and marks the
padded tokens invalid in the TokenFeatureMaps mask; TokensToChannels then writes
`mask_fill_value` into the corresponding output channels.

TokensToChannels produces a `B x 12 x (H/4) x (W/4)` feature map (12 monthly logits at
each patch), which we upsample to the input resolution before SegmentationHead
computes the cross entropy loss against the month index.

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
              model_id: OLMOEARTH_V1_2_BASE
              patch_size: 4
              # Output a TokenFeatureMaps (B x 768 x H/4 x W/4 x 12) instead of
              # pooling over the 12 timesteps.
              token_pooling: false
              # Use the actual image timestamps rather than dummy monthly ones.
              use_legacy_timestamps: false
        decoder:
          # Project each of the 12 per-timestep tokens to one logit, giving a
          # B x 12 x H/4 x W/4 feature map.
          - class_path: rslearn.models.tokens_to_channels.TokensToChannels
            init_args:
              in_dim: 768
              out_dim: 1
          # Upsample by the patch size to get logits at the input resolution.
          - class_path: rslearn.models.upsample.Upsample
            init_args:
              scale_factor: 4
              mode: "bilinear"
          - class_path: rslearn.train.tasks.segmentation.SegmentationHead
    optimizer:
      class_path: rslearn.models.olmoearth_pretrain.optimizer.LayerDecayAdamW
      init_args:
        lr: 0.0001
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    inputs:
      sentinel2_l2a:
        data_type: "raster"
        # 12 monthly mosaics, in chronological order.
        layers: ["sentinel2_l2a", "sentinel2_l2a.1", "sentinel2_l2a.2", "sentinel2_l2a.3", "sentinel2_l2a.4", "sentinel2_l2a.5", "sentinel2_l2a.6", "sentinel2_l2a.7", "sentinel2_l2a.8", "sentinel2_l2a.9", "sentinel2_l2a.10", "sentinel2_l2a.11"]
        bands: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
        passthrough: true
        dtype: FLOAT32
        load_all_layers: true
      targets:
        data_type: "raster"
        layers: ["label"]
        bands: ["change_month"]
        dtype: INT32
        is_target: true
    task:
      class_path: rslearn.train.tasks.segmentation.SegmentationTask
      init_args:
        # One class per timestep.
        num_classes: 12
        # Pixels where no change occurred are excluded from the loss.
        nodata_value: 255
        enable_miou_metric: true
    default_config:
      transforms:
        - class_path: rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize
          init_args:
            band_names:
              sentinel2_l2a: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
```
