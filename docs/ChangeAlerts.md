# Change Alerts

`rslearn.change_alerts` contains components for training models that alert on
recent changes, i.e. changes that should be detected within a few weeks of when they
become observable. The model inputs a time series whose most recent images are
frequent (e.g. weekly mosaics), optionally preceded by infrequent images (e.g.
quarterly mosaics) for historical context, and predicts per-pixel:

- The change category (including a "no change" category), e.g. with a
  [SegmentationHead](models/SegmentationHead.md).
- The timestep at which the change appears, with
  [PerPixelTimestepHead](models/PerPixelTimestepHead.md). This makes it easier to
  refresh older alerts when the model is applied repeatedly over time.

These components are not intended for comparing conditions between two distant time
periods (e.g. one year versus another). For that, a model that inputs a regular time
series spanning both periods is sufficient.

## Terminology

These terms are used throughout this page. The examples use weekly frequent images and
quarterly infrequent images, as in the [example configs](#example-configs).

- **Change date**: the date when a change first becomes observable in the imagery. Labels are relative to it.
- **Category raster**: the per-pixel change category label. It usually includes an invalid category for unlabeled pixels (e.g. 0) and a "no change" category (e.g. 1).
- **Change day raster**: the per-pixel change date label, as the number of days since 1970-01-01 (UTC), with 65535 at pixels without a change.
- **Frequent images**: the most recent mosaics, with one per frequent period, materialized in the `frequent` layer.
  An example configuration is to have 7-day periods, with 4 max matches and a 56-day time range.
  Setting the time range longer than `(# frequent periods) * (# frequent images)` is useful since some frequent periods may not have any images; they would be skipped with earlier periods acting as a fallback.
- **Infrequent images**: mosaics with one per infrequent period, materialized in the `infrequent` layer, providing historical context before the frequent images.
- **Detection range**: detect changes up to this duration into the past from the end of the input time series.
  During training, we will only pick inputs such that the change date is within this duration of the end of the latest frequent period.

## Example Configs

These example configs train and apply a change alert model on weekly Sentinel-2
mosaics preceded by quarterly mosaics, using OlmoEarth:

- [config.json](ChangeAlerts/config.json): the dataset config, with the frequent and
  infrequent layers (see [Dataset](#dataset)), the label layers, and the output
  layers.
- [config.yaml](ChangeAlerts/config.yaml): the corresponding model config.

## Dataset

Labels consist of a category raster and a change day raster, matching `PerPixelTimestepTask`.

`ChangeAlertTiming` computes the time ranges that should be used for training and prediction windows
given the frequent period, the number of frequent images, and the detection range.

```python
from datetime import UTC, datetime, timedelta
from rslearn.change_alerts.layers import ChangeAlertTiming

timing = ChangeAlertTiming(
    frequent_period=timedelta(days=7),
    num_frequent=4,
    detection_range=timedelta(days=91),
)
# Training window time range: (change date - 21 days, change date + 91 days).
train_time_range = timing.train_time_range(datetime(2024, 5, 10, tzinfo=UTC))
# Prediction window time range: (now - 28 days, now).
predict_time_range = timing.predict_time_range(datetime.now(UTC))
```

During training, the change should appear at varying positions within the time
series. Since the training window spans `(# frequent images) + (detection range) / (frequent period) - 1`
frequent periods, the sampler can pick any consecutive frequent images within it
to construct the input time series:

- The earliest pick covers `(change date - (# frequent images - 1) * (frequent period), change date + frequent period)`, so the change appears in the
  latest image.
- The latest pick ends at the window end, `change date + detection range`, so the change is the detection
  range before the end of the time series.
- In between, the change can appear at every position, one frequent period apart.

`make_layer_configs` creates the two layers used by both training and prediction windows:

- The `frequent` layer covers the window time range, with one mosaic per frequent
  period. It holds up to `(# frequent images) + (detection range) / (frequent period) - 1`
  frequent images (16 in the example). Prediction windows only span `# frequent images` frequent
  periods, so they would have at most that many frequent images.
- The optional `infrequent` layer ends at the window start and extends back by the
  infrequent history duration, with one mosaic per infrequent period.

This setup allows us to use the same dataset config for both training and prediction. For training,
the windows are longer, and get more frequent images, so that they can span all the "picks" that we
might use for the input time series. For prediction, the windows just cover one "pick".

```python
from rslearn.change_alerts.layers import make_layer_configs

layers = make_layer_configs(
    timing,
    data_source={"class_path": "rslearn.data_sources.planetary_computer.Sentinel2", "ingest": False},
    band_sets=[{"bands": ["B02", "B03", "B04", "B08"], "dtype": "uint16"}],
    infrequent_period=timedelta(days=90),
    infrequent_duration=timedelta(days=720),
)
# layers is a dict from layer name ("frequent", "infrequent") to layer config dict, to
# add to the dataset config.json.
```

## ChangeTimeSeriesSampler

`ChangeTimeSeriesSampler` is a transform that builds the time series for each example
from the frequent input (`frequent_key`) and the infrequent input (`infrequent_key`):

1. Make a pick of `num_frequent` consecutive frequent images. With `select: random`
   (the default, for training and evaluation), the pick starts at a random frequent
   image. With `select: latest` (for prediction), it is the latest frequent images.
2. With `select: random`, keep all of the pick with probability
   `full_frequent_probability` (default 0.5). Otherwise, make a reduced pick that
   keeps only the latest `n` images, with `n` uniform between 1 and
   `num_frequent - 1`.
3. Take the infrequent images that end at or before the start of the pick. If
   `infrequent_period` is set (e.g. 90 days), add virtual history for the gap between
   the first frequent image and the start of the pick (see below). Keep the latest
   `num_infrequent` images. In a prediction window, the pick starts at the first
   frequent image, so there is no gap and no virtual history.
4. Concatenate the history and the kept frequent images chronologically into
   `input_dict[output_key]`.

In a prediction window, the infrequent images end where the pick begins, so the latest
infrequent period can include the change. Virtual history reproduces this in training
windows, whose infrequent images end before the change. The gap is divided into
`gap / infrequent_period` virtual infrequent periods of equal duration, rounded up or
down with equal probability (`ROUND_UP_PROBABILITY`). Each virtual period is represented
by a random frequent image within it, like a mosaic whose timestamp is that of one of its
scenes, and virtual periods without frequent images are skipped. If rounding down gives
no periods (a gap shorter than one infrequent period), the gap is instead merged into the
last infrequent period, which is then represented by the infrequent image or, with
probability `gap / (gap + infrequent_period)`, by a random frequent image in the gap.

The images dropped in a reduced pick are treated as missing: the history is still chosen
relative to the start of the original pick, so the dropped images are never virtual
history. So the reduced pick looks like a prediction window where the earlier frequent
periods have no images.

The gap is measured from the first frequent image rather than from the infrequent
images, because the timestamp of each mosaic is the time of one of its scenes (which can
be anywhere within its period), not the period itself. Since the frequent layer starts
where the infrequent layer ends, the first frequent image is within about one frequent
period of the end of the infrequent images.

The timestep at which the change appears is predicted with `PerPixelTimestepTask`, using
the change day raster as its target, and `PerPixelTimestepHead` with `mode: AFTER`, which
maps each change day to the earliest image whose midpoint is on or after it. Pixels
without a change day (e.g. negatives) are ignored by the timestep task, so they only
train the category. With a reduced pick, a change in one of the dropped images is
mapped to the first kept frequent image.

If change dates vary within a window, set `category_target` and `change_day_target` (the
`PerPixelTimestepTask` target) so that the sampler marks pixels whose change is outside
the time series (e.g. after the latest image) invalid for both tasks, since the change
cannot be observed in the input. This is not needed when every pixel in a window shares
the window's change date.

The frequent and infrequent inputs should be loaded with `load_all_layers` and
`load_all_item_groups`. The frequent input is required, so windows without any
frequent images are skipped.

```yaml
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    inputs:
      freq:
        data_type: raster
        layers: [frequent]
        bands: [B02, B03, B04, B08]
        passthrough: true
        load_all_layers: true
        load_all_item_groups: true
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
              # num_infrequent + num_frequent.
              num_classes: 12
    default_config:
      transforms:
        - class_path: rslearn.change_alerts.sampler.ChangeTimeSeriesSampler
          init_args:
            frequent_key: freq
            num_frequent: 4
            infrequent_key: infreq
            num_infrequent: 8
            output_key: sentinel2_l2a
            infrequent_period: "90 days, 0:00:00"
            # Only needed if change dates vary within a window.
            category_target: category
            change_day_target: timestep
```

## Prediction

At prediction time, each window is a prediction window with time range
`(now - (# frequent images) * (frequent period), now)` (from `ChangeAlertTiming.predict_time_range`), so the time series
looks back from the time to alert on. The prediction dataset uses the same dataset
config as the training dataset, and the same model config, with these differences in
`predict_config`:

- Override `transforms` with the sampler using `select: latest`, so that it uses the
  latest frequent images without reducing them. The split transforms replace
  `default_config.transforms`, so the other transforms must be repeated.
- Set `skip_targets: true` so that the label inputs are skipped.

If some frequent periods in a prediction window have no images, the time series has
fewer frequent images, as in a reduced pick during training. Prediction windows
without any frequent images are rejected during prepare, so they get no outputs.

The category and change day outputs can be written to separate output layers with two
`RslearnWriter` callbacks, using `selector: [category]` and `selector: [timestep]`. The
change day output layer should use the `uint16` dtype with `nodata_value: 65535`. See
[config.yaml](ChangeAlerts/config.yaml) for the full model config.

## Alerting

When the model is applied repeatedly, e.g. weekly, each change stays in the input time
series and is predicted again on every run while it is within the detection range.
To generate vector alerts only for new changes, maintain a raster that stores, at each
pixel, the **last alert date** there (as days since 1970-01-01, with 0 where there
has not been an alert yet). This raster should be on the same grid as the predictions,
so use prediction windows with the same names and bounds on every run, and only update
their time range to the current prediction window time range.

On each run:

1. A pixel is eligible for a new alert if its predicted category is a change category
   (i.e. not invalid or "no change"), and its predicted change date is more than a
   **margin** after its last alert date. Since the change date predictions can jitter
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
  the probability of the no change class (`none_class`). By default it stores the
  scores at all valid pixels to compute the exact AUROC; set `thresholds` (e.g. to
  200) to compute a binned AUROC with constant memory instead.
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
