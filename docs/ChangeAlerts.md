# Change Alerts

`rslearn.change_alerts` contains components for training models that alert on
recent changes, i.e. changes that should be detected within a few weeks of when they
become observable. The model inputs a time series whose most recent images are
frequent (e.g. weekly mosaics), optionally preceded by infrequent images (e.g.
quarterly mosaics) for historical context, and predicts per-pixel:

- The change category (including a "no change" category), e.g. with a
  [SegmentationHead](models/SegmentationHead.md).
- The timestep at which the change appears, with
  [PerPixelTimestepHead](models/PerPixelTimestepHead.md). This makes it possible to
  refresh older alerts when the model is applied repeatedly over time.

These components are not intended for comparing conditions between two distant time
periods (e.g. one year versus another). For that, a model that inputs a regular time
series spanning both periods is sufficient.

## Dataset

Each window is a point in time at the date when the change first becomes observable,
i.e. its time range is `(change date, change date)`. Labels consist of a category
raster and a change day raster, which stores the number of days since 1970-01-01 (UTC)
at which the change first becomes observable at each pixel (0 at pixels without a
change, e.g. negatives). The change day raster can be stored as `uint16`.

During training, the change should appear at varying positions within the input time
series. To support this, the dataset contains several "slots", each a time series that
ends at a fixed offset after the change date. `slot_end_offsets` computes evenly
spaced offsets, from one period after the change (so that the change appears only in
the latest image) to the end of the change detection range:

```python
from datetime import timedelta
from rslearn.change_alerts.slots import slot_end_offsets

# [7, 35, 62, 90] days.
offsets = slot_end_offsets(4, detection_range=timedelta(days=90), period=timedelta(days=7))
```

`make_slot_layer_configs` creates the dataset layers for each slot: a frequent layer
ending at the slot end, and optionally an infrequent layer ending at or before the
start of the frequent images that the model uses. The layers can be longer than the
model input, so that the sampler can fall back to older periods when the most recent
ones have no images.

```python
from rslearn.change_alerts.slots import make_slot_layer_configs

layers = make_slot_layer_configs(
    end_offsets=offsets,
    data_source={"class_path": "rslearn.data_sources.planetary_computer.Sentinel2", "ingest": False},
    band_sets=[{"bands": ["B02", "B03", "B04", "B08"], "dtype": "uint16"}],
    frequent_period=timedelta(days=7),
    frequent_duration=timedelta(days=63),
    infrequent_period=timedelta(days=90),
    infrequent_duration=timedelta(days=900),
    infrequent_end_before=timedelta(days=60),
)
# layers is a dict from layer name ("frequent_0", "infrequent_0", ...) to layer
# config dict, to add to the dataset config.json.
```

## ChangeTimeSeriesSampler

`ChangeTimeSeriesSampler` is a transform that picks one slot per example (randomly
during training, or a fixed one via `option_index` for evaluation and prediction),
builds the time series, and derives the targets:

1. Take the latest `num_frequent` frequent images within `frequent_lookback_days` of
   the latest frequent image.
2. Take the latest `num_infrequent` infrequent images that end before the first
   selected frequent image.
3. Concatenate them chronologically into `input_dict[output_key]`.
4. If the change day raster is present, compute the timestep target as the first image
   whose time range ends after the change day. Pixels whose change is outside the time
   series are ignored for both the timestep and category targets.

Options without enough images are only picked if no option has enough, in which case
the time series has fewer timesteps.

The sampler removes the option inputs and the change day raster from the input dict.
Inputs that are only used by the sampler of another split (e.g. validation slots that
are not used for training) can be removed with `drop_keys`.

Each slot input should be loaded with `load_all_layers` and `load_all_item_groups`, and
the change day raster as a passthrough target input. The timestep task needs some
target raster to process, so map the change day raster to it; the sampler then
overwrites the timestep target.

```yaml
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    inputs:
      freq_0:
        data_type: raster
        layers: [frequent_0]
        bands: [B02, B03, B04, B08]
        passthrough: true
        load_all_layers: true
        load_all_item_groups: true
      infreq_0:
        data_type: raster
        layers: [infrequent_0]
        bands: [B02, B03, B04, B08]
        passthrough: true
        load_all_layers: true
        load_all_item_groups: true
      # ... freq_1, infreq_1, etc.
      label_category:
        data_type: raster
        layers: [label_category]
        bands: [label]
        is_target: true
        dtype: INT32
      change_day:
        data_type: raster
        layers: [label_change_day]
        bands: [label]
        is_target: true
        passthrough: true
        dtype: INT32
    task:
      class_path: rslearn.train.tasks.multi_task.MultiTask
      init_args:
        input_mapping:
          category:
            label_category: targets
          timestep:
            change_day: targets
        tasks:
          category:
            class_path: rslearn.train.tasks.segmentation.SegmentationTask
            init_args:
              num_classes: 3
              nodata_value: 0
          timestep:
            class_path: rslearn.train.tasks.per_pixel_timestep.PerPixelTimestepTask
            init_args:
              num_classes: 12
    train_config:
      transforms:
        - class_path: rslearn.change_alerts.sampler.ChangeTimeSeriesSampler
          init_args:
            options:
              - {frequent: freq_0, infrequent: infreq_0}
              - {frequent: freq_1, infrequent: infreq_1}
            num_frequent: 4
            frequent_lookback_days: 60
            num_infrequent: 8
            output_key: sentinel2_l2a
            category_target: category
            timestep_target: timestep
```

## Metrics

`rslearn.change_alerts.metrics` has metrics that can be added to the tasks via
`other_metrics`:

- `BalancedAccuracy`: the mean per-class recall of the category, over the classes
  present in the targets.
- `ChangeAUROC`: the AUROC of change versus no change, scoring each pixel by one minus
  the probability of the no change class (`none_class`).
- `TimestepToleranceAccuracy`: for the timestep task, the fraction of change pixels
  whose predicted timestep is within `tolerance` of the target timestep. Unlike the
  built-in accuracy, it does not require the logits to have exactly `num_classes`
  channels, which is not the case when every time series in a batch is shorter than
  `num_classes`.

```yaml
          category:
            class_path: rslearn.train.tasks.segmentation.SegmentationTask
            init_args:
              num_classes: 3
              nodata_value: 0
              other_metrics:
                balanced_accuracy:
                  class_path: rslearn.change_alerts.metrics.BalancedAccuracy
                  init_args:
                    num_classes: 3
                change_auroc:
                  class_path: rslearn.change_alerts.metrics.ChangeAUROC
                  init_args:
                    none_class: 1
          timestep:
            class_path: rslearn.train.tasks.per_pixel_timestep.PerPixelTimestepTask
            init_args:
              num_classes: 12
              enable_accuracy_metric: false
              other_metrics:
                within1:
                  class_path: rslearn.change_alerts.metrics.TimestepToleranceAccuracy
                  init_args:
                    tolerance: 1
```
