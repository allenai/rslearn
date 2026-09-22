## rslearn.models.simple_time_series.SimpleTimeSeries

SimpleTimeSeries is a feature extractor that wraps a unitemporal feature extractor and
applies it on a time series. It encodes each image in the time series individually
using the unitemporal feature extractor, and then pools the features temporally via max
pooling, mean pooling, a ConvRNN, 3D convolutions, or 1D convolutions.

It inputs the time series as an image where the timesteps are stacked along the
channel axis (as produced by a DataInput with multiple layers), and splits the stack
back into individual images using `image_keys`. It outputs a FeatureMaps with the same
structure as the wrapped feature extractor, but with the features pooled over time.

### Configuration

Here is a summary, see `rslearn.models.simple_time_series` for all of the available
options.

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
                class_path: # ...
                init_args:
                  # ...
              # One of "max" (default), "mean", "convrnn", "conv3d", or
              # "conv1d".
              op: "max"
              # Number of layers for convrnn, conv3d, and conv1d ops.
              num_layers: null
              # A map from input dict keys to the number of bands per image.
              # This is used to split up the time series back into the
              # individual images.
              image_keys:
                sentinel2: 12
                sentinel1: 2
          - ...
```

### Example

The [main README](https://github.com/allenai/rslearn/blob/master/README.md) has an
example of using SimpleTimeSeries with SatlasPretrain, and the
[BitemporalSentinel2](../examples/BitemporalSentinel2.md) example uses it for a
bitemporal classification task.
