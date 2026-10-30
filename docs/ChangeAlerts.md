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

## Terminology

These terms are used throughout this page. The examples use weekly frequent images and
quarterly infrequent images, as in the [example configs](#example-configs).

- **Change date** (`c`): the date when a change first becomes observable in the
  imagery. Labels are relative to it.
- **Category raster**: the per-pixel change category label, including an invalid
  category (e.g. 0) and a "no change" category (e.g. 1).
- **Change day raster**: the per-pixel change date label, as the number of days since
  1970-01-01 (UTC), with 65535 at pixels without a change (e.g. negatives).
- **Frequent period** (`p`, e.g. 7 days): the duration of each frequent image.
- **Frequent images**: mosaics with one per frequent period, materialized in the
  `frequent` layer. Periods without any images are skipped.
- **Number of frequent images** (`num_frequent`, `N`, e.g. 4): the maximum number of
  frequent images in the time series.
- **Infrequent period** (`q`, e.g. 90 days): the duration of each infrequent image.
- **Infrequent images**: mosaics with one per infrequent period, materialized in the
  `infrequent` layer, providing historical context before the frequent images.
- **Number of infrequent images** (`num_infrequent`, e.g. 8): the maximum number of
  infrequent images in the time series.
- **Infrequent history duration** (`I`, e.g. 720 days): how far back the infrequent
  layer extends before the window start. It must be a multiple of the infrequent
  period, and is typically `num_infrequent` infrequent periods.
- **Detection range** (`D`, e.g. 91 days): the longest time from the change date to
  the end of the time series at which the model should still detect the change. It
  must be a multiple of the frequent period, so that a frequent period starts exactly
  at the change date.
- **Training window**: a window used for training, with time range
  `(c - (N - 1) * p, c + D)`, e.g. `(c - 21 days, c + 91 days)`. For windows without a
  change (negatives), any date can be used as `c`.
- **Time to alert on** (`t`): the time at which the model is applied, e.g. now.
- **Prediction window**: a window used for prediction, with time range
  `(t - N * p, t)`, e.g. `(t - 28 days, t)`, i.e. the latest `N` frequent periods.
- **Window start** and **window end**: the start and end of the window time range.
- **Pick**: the `N` consecutive frequent images (or all of them if there are fewer)
  chosen by `ChangeTimeSeriesSampler` for one example. Since the training window
  starts only `N - 1` frequent periods before the change date, every pick ends between
  one frequent period and the detection range after the change date.
- **Reduced pick**: during training, the sampler sometimes keeps only the latest
  `n < N` images of the pick, to simulate prediction windows where some frequent
  periods have no images.
- **Stand-in image**: the frequent image just before the pick, used in place of the
  most recent infrequent image when the pick starts at least `stand_in_min_gap` (e.g.
  half an infrequent period) after the first frequent image, i.e. when there would
  otherwise be a large gap between the infrequent images and the pick.
- **Time series**: the model input built by the sampler: the infrequent images (and
  stand-in image, if any) followed by the kept frequent images of the pick.
- **Last alert date** and **margin**: see [Alerting](#alerting).

## Example Configs

These example configs train and apply a change alert model on weekly Sentinel-2
mosaics preceded by quarterly mosaics, using OlmoEarth:

- [config.json](ChangeAlerts/config.json): the dataset config, with the frequent and
  infrequent layers (see [Dataset](#dataset)), the label layers, and the output
  layers. It is used for both the training and the prediction dataset.
- [config.yaml](ChangeAlerts/config.yaml): the model config, which is also shared by
  training and prediction.

The training and prediction windows go in separate datasets (or separate groups of
one dataset), and the model config reads the dataset path from `DATASET_PATH`:

```
cp config.json dataset_train/config.json
cp config.json dataset_predict/config.json
# After creating and materializing windows in each dataset:
DATASET_PATH=./dataset_train rslearn model fit --config config.yaml
DATASET_PATH=./dataset_predict rslearn model predict --config config.yaml
```

The dataset config includes the output layers, since `RslearnWriter` loads the output
layer config when it is set up, including during `model fit`.

## Dataset

Labels consist of a category raster and a change day raster (see
[Terminology](#terminology)), matching `PerPixelTimestepTask`. The change day raster
can be stored as `uint16`.

`ChangeAlertTiming` holds the frequent period, the number of frequent images, and the
detection range, and computes the window time ranges:

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
series. Since the training window spans `N + D / p - 1` frequent periods, the sampler
can pick any `N` consecutive frequent images within it:

- The earliest pick covers `(c - (N - 1) * p, c + p)`, so the change appears in the
  latest image.
- The latest pick ends at the window end, `c + D`, so the change is the detection
  range before the end of the time series.
- In between, the change appears at every position, one frequent period apart.

`make_layer_configs` creates the two layers used by both training and prediction
windows:

- The `frequent` layer covers the window time range, with one mosaic per frequent
  period. It holds up to `N + D / p - 1` frequent images (16 in the example), so
  prediction windows, which only span `N` frequent periods, have at most `N`. It also
  gets `min_matches=1`, so windows without any frequent images are reported as
  rejected during prepare.
- The optional `infrequent` layer ends at the window start and extends back by the
  infrequent history duration, with one mosaic per infrequent period.

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
3. Take the latest `num_infrequent` infrequent images that end at or before the start
   of the pick. If `stand_in_min_gap` is set (e.g. to 45 days, half of an infrequent
   period) and the pick starts at least that long after the first frequent image, the
   frequent image just before the pick is added as a stand-in image, replacing the
   oldest infrequent image. In a training window, this applies to the later picks; in
   a prediction window, the pick always starts at the first frequent image, so it does
   not apply.
4. Concatenate the infrequent images, stand-in image (if any), and kept frequent
   images chronologically into `input_dict[output_key]`.

The images dropped in a reduced pick are treated as missing: the infrequent images and
stand-in image are still chosen relative to the start of the original pick. So the
reduced pick looks like a prediction window where the earlier frequent periods have no
images.

The gap for the stand-in image is measured from the first frequent image rather than
from the infrequent images, because the timestamp of each mosaic is the time of one of
its scenes (which can be anywhere within its period), not the period itself. Since the
frequent layer starts where the infrequent layer ends, the first frequent image is
within about one frequent period of the end of the infrequent images.

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
            stand_in_min_gap: "45 days, 0:00:00"
            # Only needed if change dates vary within a window.
            category_target: category
            change_day_target: timestep
```

## Prediction

At prediction time, each window is a prediction window with time range
`(t - N * p, t)` (from `ChangeAlertTiming.predict_time_range`), so the time series
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
