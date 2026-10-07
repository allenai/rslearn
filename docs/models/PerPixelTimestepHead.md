## rslearn.train.tasks.per_pixel_timestep.PerPixelTimestepHead

PerPixelTimestepHead is the predictor for
[PerPixelTimestepTask](../TasksAndModels.md#perpixeltimesteptask). It predicts one of
the input timesteps at each pixel, and passes along the input image timestamps so
that the task can write the predicted timestep as a date (days since 1970-01-01).

It inputs a FeatureMaps, which must contain a single feature map with one logit per
timestep, e.g. from [TokensToChannels](TokensToChannels.md). Channel `t` must
correspond to timestep `t` of the input selected by `input_key`. Like
[SegmentationHead](SegmentationHead.md), it applies softmax to the logits and computes
the cross entropy loss against the timestep index targets (loss key "cls").

For each example, the output is a dict with:

- "probs": the `C x H x W` softmax probabilities.
- "timestamps": an int64 tensor with the number of days since 1970-01-01 (UTC) of the
  midpoint of each input timestep's time range. The timestamps must be timezone-aware,
  and the midpoints must fall between 1970-01-01 and 2149-06-06 so that they fit in
  the uint16 output of PerPixelTimestepTask.

The input must be a RasterImage with timestamps, with at most as many timesteps as
logit channels. Additional channels (e.g. from padding tokens) are ignored by
PerPixelTimestepTask.

### Configuration

```yaml
        decoder:
          # ...
          - class_path: rslearn.train.tasks.per_pixel_timestep.PerPixelTimestepHead
            init_args:
              # The input whose timesteps the logit channels correspond to.
              input_key: sentinel2_l2a
```

### Example

See the [TokensToChannels](TokensToChannels.md) example, which pairs
PerPixelTimestepHead with per-timestep tokens from OlmoEarth.
