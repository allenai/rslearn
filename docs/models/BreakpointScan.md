## rslearn.models.breakpoint_scan.BreakpointScan

BreakpointScan applies a learned changepoint scan over the token (time) dimension of a
TokenFeatureMaps.

It inputs a TokenFeatureMaps containing a single `B x C x H x W x T` feature map, with
one token per timestep in chronological order (T must be at least 2). For every
candidate split `t` in `[0, T-2]`, it computes the mean of the tokens up to and
including `t` (the "before" aggregate) and the mean of the tokens after `t` (the
"after" aggregate), and passes `|after - before|` through a shared 1x1 convolutional
scorer with `hidden` channels. The scorer also produces a scalar score per split, which
is used as a softmax attention over the candidate breakpoints.

The `output` option selects which feature of the scan is returned, as a FeatureMaps
with a single `B x C' x H x W` feature map:

| `output` | Shape | Description |
|---|---|---|
| `EVIDENCE` | `B x hidden x H x W` | The per-split scorer features max-pooled over the splits. A decoder on this feature can only express "change" as a before-vs-after dissimilarity at some breakpoint. |
| `BEFORE` | `B x C x H x W` | The before aggregates, weighted by the softmax over the per-split scores. Represents the state before the most likely breakpoint. |
| `AFTER` | `B x C x H x W` | The after aggregates, weighted by the softmax over the per-split scores. Represents the state after the most likely breakpoint. |
| `BEFORE_AFTER` | `B x 2C x H x W` | The channel concatenation of `BEFORE` and `AFTER`. |

We suggest using `EVIDENCE` if predicting whether a change occurred, `BEFORE_AFTER` if
predicting a category of change, and `BEFORE` or `AFTER` individually if predicting a
pre-change or post-change category describing the conditions before and after the
change.

If the input TokenFeatureMaps has masks, invalid tokens are excluded from the before
and after means, and any split with no valid token on one of its sides is excluded
from the `EVIDENCE` max-pooling and from the softmax over breakpoints. This allows
samples in a batch to have different numbers of timesteps: the token dimension T is
the maximum across the batch and the trailing padded tokens are masked. Locations with
fewer than two valid tokens produce a well-defined (but uninformative) output rather
than an error.

### Configuration

```yaml
        decoder:
          - class_path: rslearn.models.breakpoint_scan.BreakpointScan
            init_args:
              # The token embedding dimension C.
              in_dim: 768
              # Which feature of the scan to return: EVIDENCE, BEFORE, AFTER,
              # or BEFORE_AFTER.
              output: EVIDENCE
              # The hidden width of the split scorer, which is also the channel
              # count of the EVIDENCE output. The default is 256.
              hidden: 256
```

### Example

This example trains a binary change segmentation model: given a year of monthly
Sentinel-2 mosaics, predict at each pixel whether a change occurred at any point during
the year. The label raster contains 1 where a change occurred and 0 elsewhere.

The [OlmoEarth](../foundation_models/OlmoEarth.md) encoder is configured with
`token_pooling: false`, so it outputs a TokenFeatureMaps with one token per timestep
rather than pooling over time. We use an OlmoEarth v1.2 model, which has a single band
set per modality, and pass only one modality (Sentinel-2), so the tokens correspond
exactly to the T timesteps in chronological order. With OlmoEarth v1 models (which
tokenize Sentinel-2 into three band sets) or when passing multiple modalities, the
token dimension would interleave band sets and modalities with timesteps, which is not
what BreakpointScan expects.

Since we only need to predict whether a change occurred, we use the `EVIDENCE` output.
BreakpointScan produces a `B x 256 x (H/4) x (W/4)` feature map, which a UNetDecoder
upsamples to the input resolution with two output channels (change / no change) for
[SegmentationHead](SegmentationHead.md).

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
              # Output a TokenFeatureMaps (B x 768 x H/4 x W/4 x T) instead of
              # pooling over the T timesteps.
              token_pooling: false
              # Use the actual image timestamps rather than dummy monthly ones.
              use_legacy_timestamps: false
        decoder:
          # Scan for a breakpoint at each patch and return the change evidence
          # feature, which is B x 256 x H/4 x W/4.
          - class_path: rslearn.models.breakpoint_scan.BreakpointScan
            init_args:
              in_dim: 768
              output: EVIDENCE
              hidden: 256
          # Decode the evidence feature to per-pixel change / no change logits.
          - class_path: rslearn.models.unet.UNetDecoder
            init_args:
              # The evidence feature is at 1/4 resolution with 256 channels.
              in_channels: [[4, 256]]
              out_channels: 2
              conv_layers_per_resolution: 2
              num_channels: {4: 256, 2: 128, 1: 64}
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
        bands: ["change"]
        dtype: INT32
        is_target: true
    task:
      class_path: rslearn.train.tasks.segmentation.SegmentationTask
      init_args:
        # Two classes: 0 for no change, 1 for change.
        num_classes: 2
        enable_miou_metric: true
    default_config:
      transforms:
        - class_path: rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize
          init_args:
            band_names:
              sentinel2_l2a: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
```

To additionally predict the category of the change (for example, the land cover
transition), use `MultiTaskModel` with a second decoder that applies BreakpointScan
with `output: BEFORE_AFTER` (giving a `B x 1536 x H/4 x W/4` feature map) followed by
its own UNetDecoder and SegmentationHead.
