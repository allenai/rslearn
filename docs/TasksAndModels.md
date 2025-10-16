## Tasks and Models

This document details the tasks and model components available in rslearn.

## Tasks

Currently, all rslearn tasks are for supervised training for different types of
predictions (classification, bounding box detection, segmentation, etc.). All tasks
expect the input dict that they receive to include a key "targets" containing the
labels for that task.

When using SingleTaskModel, the `data.init_args.inputs` section of your model
configuration file must include an input named targets. When using MultiTaskModel, you
would generally define one input per task, name it according to the task, and then
remap those names in the input_mapping setting:

```yaml
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    inputs:
      image:
        layers: ["sentinel2"]
        bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
        dtype: FLOAT32
        passthrough: true
      regress_label:
        data_type: "raster"
        layers: ["regress_label"]
        bands: ["label"]
        is_target: true
        dtype: FLOAT32
      segment_label:
        data_type: "raster"
        layers: ["segment_label"]
        bands: ["label"]
        is_target: true
        dtype: INT32
    task:
      class_path: rslearn.train.tasks.multi_task.MultiTask
      init_args:
        tasks:
          regress:
            # ...
          segment:
            # ...
        input_mapping:
          regress:
            regress_label: "targets"
          segment:
            segment_label: "targets
```

### ClassificationTask

ClassificationTask trains a model to make global window-level classification
predictions. For example, the model may input a satellite image of a vessel at sea, and
predict whether it is a passenger vessel, cargo vessel, tanker, etc.

ClassificationTask requires vector targets. It will scan the vector features for one
with a property name matching a configurable name, and read the classification category
name or ID from there.

The configuration snippet below summarizes the most common options. See
`rslearn.train.tasks.classification` for all of the options.

```yaml
    task:
      class_path: rslearn.train.tasks.classification.ClassificationTask
      init_args:
        # The property name from which to extract the class name. The class is read
        # from the first matching feature.
        property_name: "category"
        # A list of class names.
        classes: ["passenger", "cargo", "tanker"]
        # If you are performing multi-task training, and some windows do not have
        # ground truth for the classification task, then you can enable this: if you
        # ensure the window contains the vector layer but does not contain any features
        # with the property_name, then instead of raising an exception, the task will
        # mark that target invalid so it is excluded from the classfication loss.
        allow_invalid: false
        # ClassificationTask will always compute an accuracy metric. A per-category F1
        # metric can also be enabled.
        enable_f1_metric: true
        # By default, argmax is used to determine the predicted category for computing
        # metrics and for writing predictions (in the predict stage). The pair of
        # options below can override the confidence threshold for binary classification
        # tasks (when there are two classes).
        positive_class: "cls_name" # the name of the positive class, in classes list
        positive_class_threshold: 0.75 # predict as cls_name if corresponding probability exceeds this threshold
```

For each training example, ClassificationTask computes a target dict containing the
"class" (class ID) and "valid" (flag indicating whether it is valid) keys.

Here is an example usage:

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
              model_identifier: "Sentinel2_SwinB_SI_MS"
        decoder:
          - class_path: rslearn.models.pooling_decoder.PoolingDecoder
            init_args:
              in_channels: 1024
              # The number of output channels in the layer preceding ClassificationHead
              # must match the number of classes.
              out_channels: 3
              num_conv_layers: 1
              num_fc_layers: 2
          # ClassificationHead will compute the cross entropy loss between the input
          # logits and the label class ID.
          - class_path: rslearn.train.tasks.classification.ClassificationHead
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    inputs:
      image:
        layers: ["sentinel2"]
        bands: ["B04", "B03", "B02", "B05", "B06", "B07", "B08", "B11", "B12"]
        dtype: FLOAT32
        passthrough: true
      targets:
        data_type: "vector"
        layers: ["label"]
        is_target: true
    task:
      # see example above
```

### DetectionTask

DetectionTask trains a model to predict bounding boxes with categories. For example, a
model can be trained to predict the positions of offshore platforms, wind turbines,
and vessels.

DetectionTask requires vector targets. It will only use vector features containing a
property name matching a configurable name, which is the object category. The bounding
box of the feature shape is used as the bounding box label by default, but `box_size`
can be set to instead use a fixed-size box centered at the centroid of the feature
shape.

The configuration snippet below summarizes the most common options. See
`rslearn.train.tasks.detection` for all of the options.

```yaml
    task:
      class_path: rslearn.train.tasks.detection.DetectionTask
      init_args:
        # The property name from which to extract the class name. Features without this
        # property name are ignored.
        property_name: "category"
        # A list of class names.
        classes: ["platform", "wind_turbine", "vessel"]
        # Force all boxes to be two times this size, centered at the centroid of the
        # geometry. Required for Point geometries.
        box_size: 10
        # Confidence threshold for visualization and prediction.
        score_threshold: 0.5
        # Whether to compute precision, recall, and F1 score.
        enable_precision_recall: false
        enable_f1_metric: false
```

For each training example, DetectionTask computes a target dict containing the
"boxes" (bounding box coordinates), "labels" (class labels), "valid" (flag indicating
whether the example is valid), and "width"/"height" (window width and height) keys.

Here is an example usage:

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
              model_identifier: "Sentinel2_SwinB_SI_MS"
              # The Feature Pyramid Network in SatlasPretrain is recommended for
              # detection tasks.
              fpn: true
        decoder:
          - class_path: rslearn.models.pick_features.PickFeatures
            init_args:
              # With FPN enabled, SatlasPretrain outputs five feature maps, with the
              # first one upsampled to the input resolution.
              # For detection tasks, it is best to skip the upsampled one, so we just
              # use the other four.
              indexes: [1, 2, 3, 4]
          - class_path: rslearn.models.faster_rcnn.FasterRCNN
            init_args:
              # The encoder outputs a list of feature maps at different resolutions.
              # The downsample_factors specifies those resolutions relative to the
              # input resolution, i.e., the feature maps are at 1/4, 1/8, 1/16, and
              # 1/32 of the original input resolution.
              downsample_factors: [4, 8, 16, 32]
              # Although the Swin-Base backbone in SatlasPretrain outputs different
              # embedding depths for each feature map, we have enabled the FPN which
              # produces 128 features for each resolution.
              num_channels: 128
              # Our task has three classes, but there is a quirk in the setup here
              # where we need to reserve class 0 for background.
              num_classes: 4
              anchor_sizes: [[32], [64], [128], [256]]
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    inputs:
      image:
        layers: ["sentinel2"]
        bands: ["B04", "B03", "B02", "B05", "B06", "B07", "B08", "B11", "B12"]
        dtype: FLOAT32
        passthrough: true
      targets:
        data_type: "vector"
        layers: ["label"]
        is_target: true
    task:
      class_path: rslearn.train.tasks.detection.DetectionTask
      init_args:
        property_name: "category"
        # We reserve the first class for Faster R-CNN to use to indicate background.
        classes: ["unknown", "platform", "wind_turbine", "vessel"]
        box_size: 10
```

### PerPixelRegressionTask

PerPixelRegressionTask trains a model to predict a real value at each input pixel. For
example, a model can be trained to predict the live fuel moisture content at each
pixel.

PerPixelRegressionTask requires a raster target with one band containing the ground
truth value at each pixel. If the ground truth is sparse or has missing portions, a
NODATA value can be configured.

The configuration snippet below summarizes the most common options. See
`rslearn.train.tasks.per_pixel_regression` for all of the options.

```yaml
    task:
      class_path: rslearn.train.tasks.per_pixel_regression.PerPixelRegression
      init_args:
        # Multiply ground truth values by this factor before using it for training.
        scale_factor: 0.1
        # What metric to use, either "mse" (default) or "l1".
        metric_mode: "mse"
        # Optional value to treat as invalid. The loss will be masked at pixels where
        # the ground truth value is equal to nodata_value.
        nodata_value: -1
```

For each training example, PerPixelRegressionTask computes a target dict containing the
"values" (scaled ground truth values) and "valid" (mask indicating which pixels are
valid for training) keys.

Here is an example usage:

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.multitask.SingleTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.swin.Swin
            init_args:
              pretrained: true
              output_layers: [1, 3, 5, 7]
        decoder:
          # We apply a UNet-style decoder on the feature maps from the Swin encoder to
          # compute outputs at the input resolution.
          - class_path: rslearn.models.unet.UNetDecoder
            init_args:
              # These indicate the resolution (1/X relative to the input resolution)
              # and embedding sizes of the input feature maps.
              in_channels: [[4, 128], [8, 256], [16, 512], [32, 1024]]
              # Number of output channels, should be 1 for regression.
              out_channels: 1
          - class_path: rslearn.train.tasks.per_pixel_regression.PerPixelRegressionHead
            init_args:
              # The loss function to use, either "mse" (default) or "l1".
              loss_mode: "mse"
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    inputs:
      image:
        layers: ["sentinel2"]
        bands: ["B04", "B03", "B02"]
        dtype: FLOAT32
        passthrough: true
      targets:
        data_type: "raster"
        layers: ["label"]
        bands: ["lfmc"]
        dtype: FLOAT32
        is_target: true
    task:
      # see example above
```

### RegressionTask

RegressionTask trains a model to make global window-level regression predictions. For
example, the model may input a satellite image of a vessel at sea, and predict the
length of the vessel.

RegressionTask requires vector targets. It will scan the vector features for one with a
property name matching a configurable name, and read the ground truth real value from
there.

The configuration snippet below summarizes the most common options. See
`rslearn.train.tasks.regression` for all of the options.

```yaml
    task:
      class_path: rslearn.train.tasks.regression.RegressionTask
      init_args:
        # The property name from which to extract the ground truth regression value.
        # The value is read from the first matching feature.
        property_name: "length"
        # Multiply the label value by this factor for training.
        scale_factor: 0.01
        # What metric to use, either "mse" (default) or "l1".
        metric_mode: "mse"
```

For each training example, RegressionTask computes a target dict containing the
"value" (ground truth regression value) and "valid" (flag indicating whether the sample
is valid) keys.

Here is an example usage:

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
              model_identifier: "Sentinel2_SwinB_SI_MS"
        decoder:
          - class_path: rslearn.models.pooling_decoder.PoolingDecoder
            init_args:
              in_channels: 1024
              # Must output one channel for RegressionTask.
              out_channels: 1
              num_conv_layers: 1
              num_fc_layers: 1
          - class_path: rslearn.train.tasks.regression.RegressionHead
            init_args:
              # The loss function to use, either "mse" (default) or "l1"
              loss_mode: "mse"
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    inputs:
      image:
        layers: ["sentinel2"]
        bands: ["B04", "B03", "B02", "B05", "B06", "B07", "B08", "B11", "B12"]
        dtype: FLOAT32
        passthrough: true
      targets:
        data_type: "vector"
        layers: ["label"]
        is_target: true
    task:
      # see example above
```

### SegmentationTask

SegmentationTask trains a model to classify each pixel (semantic segmentation). For
example, a model can be trained to predict the land cover type at each pixel.

SegmentationTask requires a raster target with one band containing the ground
truth class ID at each pixel. If the ground truth is sparse or has missing portions, a
NODATA value can be configured.

The configuration snippet below summarizes the most common options. See
`rslearn.train.tasks.segmentation` for all of the options.

```yaml
    task:
      class_path: rslearn.train.tasks.segmentation.SegmentationTask
      init_args:
        # Multiply ground truth values by this factor before using it for training.
        scale_factor: 0.1
        # What metric to use, either "mse" (default) or "l1".
        metric_mode: "mse"
        # Optional value to treat as invalid. The loss will be masked at pixels where
        # the ground truth value is equal to nodata_value.
        nodata_value: -1
```

For each training example, PerPixelRegressionTask computes a target dict containing the
"values" (scaled ground truth values) and "valid" (mask indicating which pixels are
valid for training) keys.

Here is an example usage:

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.multitask.SingleTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.swin.Swin
            init_args:
              pretrained: true
              output_layers: [1, 3, 5, 7]
        decoder:
          # We apply a UNet-style decoder on the feature maps from the Swin encoder to
          # compute outputs at the input resolution.
          - class_path: rslearn.models.unet.UNetDecoder
            init_args:
              # These indicate the resolution (1/X relative to the input resolution)
              # and embedding sizes of the input feature maps.
              in_channels: [[4, 128], [8, 256], [16, 512], [32, 1024]]
              # Number of output channels, should be 1 for regression.
              out_channels: 1
          - class_path: rslearn.train.tasks.per_pixel_regression.PerPixelRegressionHead
            init_args:
              # The loss function to use, either "mse" (default) or "l1".
              loss_mode: "mse"
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    inputs:
      image:
        layers: ["sentinel2"]
        bands: ["B04", "B03", "B02"]
        dtype: FLOAT32
        passthrough: true
      targets:
        data_type: "raster"
        layers: ["label"]
        bands: ["lfmc"]
        dtype: FLOAT32
        is_target: true
    task:
      # see example above
```
