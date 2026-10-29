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

## Example Configs

These example configs train and apply a change alert model on weekly Sentinel-2
mosaics preceded by quarterly mosaics, using OlmoEarth:

- [config_train.json](ChangeAlerts/config_train.json): the training dataset config,
  with four slots (see [Dataset](#dataset)) and the label layers.
- [config_predict.json](ChangeAlerts/config_predict.json): the prediction dataset
  config, with a single slot that ends at the window time (see
  [Prediction](#prediction)).
- [config.yaml](ChangeAlerts/config.yaml): the model config, which is shared by
  training and prediction.

The two dataset configs go in separate datasets, and the model config reads the
dataset path from `DATASET_PATH`:

```
cp config_train.json dataset_train/config.json
cp config_predict.json dataset_predict/config.json
# After creating and materializing windows in each dataset:
DATASET_PATH=./dataset_train rslearn model fit --config config.yaml
DATASET_PATH=./dataset_predict rslearn model predict --config config.yaml
```

Both dataset configs include the output layers, since `RslearnWriter` loads the output
layer config when it is set up, including during `model fit`.

## Dataset

Each window is a point in time at the date when the change first becomes observable,
i.e. its time range is `(change date, change date)`. Labels consist of a category
raster and a change day raster, which stores the number of days since 1970-01-01 (UTC)
at which the change first becomes observable at each pixel, with 65535 at pixels
without a change (e.g. negatives), matching `PerPixelTimestepTask`. The change day
raster can be stored as `uint16`.

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

`make_slot_layer_configs` creates a frequent layer for each slot, ending at the slot
end, and optionally one infrequent layer shared by all slots. The infrequent layer ends
where the latest slot's frequent layer starts and extends back far enough to provide
`infrequent_duration` of history for every slot; the sampler only uses the infrequent
images that end before the selected frequent images.

Each frequent layer holds the latest `frequent_max_matches` periods within
`frequent_duration` that have images, so that it falls back to older periods when the
most recent ones have no images. `frequent_max_matches` is the number of frequent images
that the model uses. Frequent layers also get `min_matches=1`, so slots without any
frequent images are reported as rejected during prepare.

```python
from rslearn.change_alerts.slots import make_slot_layer_configs

layers = make_slot_layer_configs(
    end_offsets=offsets,
    data_source={"class_path": "rslearn.data_sources.planetary_computer.Sentinel2", "ingest": False},
    band_sets=[{"bands": ["B02", "B03", "B04", "B08"], "dtype": "uint16"}],
    frequent_period=timedelta(days=7),
    frequent_duration=timedelta(days=63),
    frequent_max_matches=4,
    infrequent_period=timedelta(days=90),
    infrequent_duration=timedelta(days=900),
)
# layers is a dict from layer name ("frequent_0", ..., "frequent_3", "infrequent")
# to layer config dict, to add to the dataset config.json.
```

## ChangeTimeSeriesSampler

`ChangeTimeSeriesSampler` is a transform that picks one frequent input per example
from `frequent_keys` and builds the time series. It picks the frequent input with the
most images, randomly among ties, or a fixed one via `frequent_index` for evaluation
and prediction. Then:

1. Take the images of the selected frequent input.
2. Take the latest `num_infrequent` infrequent images that end before the first
   selected frequent image.
3. Concatenate them chronologically into `input_dict[output_key]`.

The timestep at which the change appears is predicted with `PerPixelTimestepTask`, using
the change day raster as its target, and `PerPixelTimestepHead` with `mode: AFTER`, which
maps each change day to the earliest image whose midpoint is on or after it. Pixels
without a change day (e.g. negatives) are ignored by the timestep task, so they only
train the category.

If change dates vary within a window, set `category_target` and `change_day_target` (the
`PerPixelTimestepTask` target) so that the sampler marks pixels whose change is outside
the time series (e.g. after the latest image) invalid for both tasks, since the change
cannot be observed in the input. This is not needed when every pixel in a window shares
the window's change date.

Each frequent and infrequent input should be loaded with `load_all_layers` and
`load_all_item_groups`.

Only `freq_0` is required, so windows without any images in that slot are skipped. The
other frequent inputs are not required: if a slot has no images, its input is missing
and counts as having zero images, so another slot is picked.

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
      freq_1:
        data_type: raster
        layers: [frequent_1]
        bands: [B02, B03, B04, B08]
        passthrough: true
        load_all_layers: true
        load_all_item_groups: true
        required: false
      # ... freq_2, freq_3 like freq_1.
      infreq:
        data_type: raster
        layers: [infrequent]
        bands: [B02, B03, B04, B08]
        passthrough: true
        load_all_layers: true
        load_all_item_groups: true
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
    default_config:
      transforms:
        - class_path: rslearn.change_alerts.sampler.ChangeTimeSeriesSampler
          init_args:
            frequent_keys: [freq_0, freq_1, freq_2, freq_3]
            infrequent_key: infreq
            num_infrequent: 8
            output_key: sentinel2_l2a
            # Only needed if change dates vary within a window.
            category_target: category
            change_day_target: timestep
```

## Prediction

At prediction time, each window is a point in time at the time to alert on (e.g. now),
i.e. its time range is `(t, t)`, and the time series looks back from there. So the
prediction dataset only needs one slot, which ends at the window time. Create it with
the same arguments as the training layers, but with a single end offset of zero:

```python
layers = make_slot_layer_configs(
    end_offsets=[timedelta(0)],
    # ... the same data_source, band_sets, and frequent and infrequent options as for
    # training.
)
```

This produces layers named `frequent_0` and `infrequent`, like the training dataset,
so the same model config can be used for training and prediction:

- The `freq_1` to `freq_3` inputs are not required, so they are skipped since the
  prediction windows do not have those layers, and the sampler always picks `freq_0`.
- Set `skip_targets: true` in `predict_config` so that the label inputs are skipped.

The category and change day outputs can be written to separate output layers with two
`RslearnWriter` callbacks, using `selector: [category]` and `selector: [timestep]`. The
change day output layer should use the `uint16` dtype with `nodata_value: 65535`. See
[config.yaml](ChangeAlerts/config.yaml) for the full model config.

## Alerting

When the model is applied repeatedly, e.g. weekly, each change stays in the input time
series and is predicted again on every run until it is older than the detection range.
To generate vector alerts only for new changes, maintain a raster that stores, at each
pixel, the date of the last alert there (as days since 1970-01-01, with 0 where there
has not been an alert yet). This raster should be on the same grid as the predictions,
so use prediction windows with the same names and bounds on every run, and only update
their time range to the current time.

On each run:

1. A pixel is eligible for a new alert if its predicted category is a change category
   (i.e. not invalid or "no change"), and its predicted change date is more than a
   margin after its last alert date. Since the change date predictions can jitter
   between runs, we recommend a margin of 90 days. This also means that a pixel alerts
   at most once within the margin, so a second change at the same pixel within the
   margin is not alerted.
2. Find the connected components of eligible pixels with the same category, and keep
   the components that meet a minimum area.
3. Vectorize each kept component as an alert, and set the last alert date of its
   pixels to the current date.

Since only eligible pixels form components, when a change grows after it was alerted,
the next alert only covers the newly changed area around it (if it meets the minimum
area). Eligible pixels in components that are too small do not get a last alert date,
so they stay eligible on later runs, and can be alerted once the component grows.

Here is a sketch of one run. The predicted category and change day rasters can be read
with rasterio from the `output_category` and `output_change_day` layers of each window,
which also provides the transform.

```python
import numpy as np
import rasterio.features
import scipy.ndimage

INVALID_CLASS = 0
NONE_CLASS = 1
NODATA_DAY = 65535
MARGIN_DAYS = 90


def make_alerts(category, change_day, last_alert_day, run_day, transform, min_pixels):
    """Make alerts from one run's outputs, and update the last alert date raster.

    Args:
        category: HxW predicted category (output_category).
        change_day: HxW predicted change day (output_change_day).
        last_alert_day: HxW day of the last alert at each pixel, 0 if none.
        run_day: the day of this run (days since 1970-01-01).
        transform: the affine transform of the rasters, e.g. from rasterio.
        min_pixels: the minimum number of pixels in an alert.

    Returns:
        a tuple (alerts, last_alert_day) of the list of GeoJSON-like alert features
        and the updated last alert date raster.
    """
    change_day = change_day.astype(np.int32)
    eligible = (
        (category != INVALID_CLASS)
        & (category != NONE_CLASS)
        & (change_day != NODATA_DAY)
        & (change_day > last_alert_day.astype(np.int32) + MARGIN_DAYS)
    )

    alerts = []
    last_alert_day = last_alert_day.copy()
    for cls in np.unique(category[eligible]):
        components, _ = scipy.ndimage.label(eligible & (category == cls))
        sizes = np.bincount(components.ravel())
        keep = np.isin(components, np.flatnonzero(sizes >= min_pixels)) & (components > 0)
        last_alert_day[keep] = run_day

        for geometry, component_id in rasterio.features.shapes(
            components.astype(np.int32), mask=keep, transform=transform
        ):
            pixels = components == component_id
            alerts.append({
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "category": int(cls),
                    "alert_day": run_day,
                    "change_day": int(np.median(change_day[pixels])),
                    "num_pixels": int(pixels.sum()),
                },
            })
    return alerts, last_alert_day
```

`min_pixels` is the minimum area divided by the pixel area, e.g. 10 for 0.1 hectares at
10 m/pixel.

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
