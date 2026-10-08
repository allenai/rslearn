## rslearn.train.tasks.per_pixel_timestep.PerPixelTimestepHead

PerPixelTimestepHead is the predictor for
[PerPixelTimestepTask](../TasksAndModels.md#perpixeltimesteptask). It predicts one of
the input timesteps at each pixel, and passes along the input image timestamps so
that the task can write the predicted timestep as a date (days since 1970-01-01).

It inputs a FeatureMaps, which must contain a single feature map with one logit per
timestep, e.g. from [TokensToChannels](TokensToChannels.md). Channel `t` must
correspond to timestep `t` of the input selected by `input_key`. Like
[SegmentationHead](SegmentationHead.md), it applies softmax to the logits and computes
the cross entropy loss (loss key "cls").

The targets from PerPixelTimestepTask are labeled dates (days since 1970-01-01), so the
head first maps each labeled date to an input timestep using the timestamps of the
input. This happens in the head rather than the task because transforms may change
which images make up the input. The date of each timestep is the UTC day of the
midpoint of its time range, and dates are compared at day granularity:

- With `mode: BEFORE`, the target is the latest timestep on or before the labeled date.
- With `mode: AFTER`, the target is the earliest timestep on or after the labeled date.

A timestep on the same day as the labeled date matches in both modes. If several
timesteps qualify with the same day, the first one is used. If no timestep qualifies
(e.g. the labeled date is before every image with `mode: BEFORE`), the pixel is
excluded from the loss and metrics, unless `fallback_to_closest_timestep_if_no_match`
is set, in which case the closest timestep is used.

For each example, the output is a dict with:

- "probs": the `C x H x W` softmax probabilities.
- "timestamps": an int64 tensor with the number of days since 1970-01-01 (UTC) of the
  midpoint of each input timestep's time range. The timestamps must be timezone-aware,
  and the midpoints must fall between 1970-01-01 and 2149-06-05 so that they fit in
  the uint16 output of PerPixelTimestepTask (65535 is reserved for nodata).
- "targets" (only when targets are given): a dict with the timestep index "classes"
  and "valid" mask used for the loss, which PerPixelTimestepTask uses to compute
  metrics.

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
              # How to map each labeled date to an input timestep: BEFORE (latest
              # timestep on or before the date) or AFTER (earliest timestep on or
              # after the date). Required, including when only predicting.
              mode: BEFORE
              # Whether to use the closest timestep when no timestep is on the
              # requested side of the labeled date, instead of excluding the pixel
              # from the loss. The default is false.
              fallback_to_closest_timestep_if_no_match: false
```

### Example

See the [TokensToChannels](TokensToChannels.md) example, which pairs
PerPixelTimestepHead with per-timestep tokens from OlmoEarth.
