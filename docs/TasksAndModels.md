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

The expected output from the model is a list of dicts (one dict per example in the
batch) with the "boxes", "scores", and "labels" keys:

- boxes: a (N, 4) float tensor, where N is the number of predicted boxes for this example,
  containing the predicted bounding box coordinates. The coordinates are in
  (x1, y1, x2, y2) order, and in relative pixel coordinates corresponding to the input
  resolution.
- scores: a (N,) float tensor containing the output probabilities.
- labels: a (N,) integer tensor containing the predicted class ID for each box.

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
        # The number of classes to predict.
        # The raster label should contain values between 0 and (num_classes-1).
        num_classes: 10
        # The value to use for NODATA pixels, which will be excluded from the loss.
        # If null (default), all pixels are considered valid.
        # If the NODATA value falls within 0 to (num_classes-1), then it must be
        # counted in num_classes (higher class IDs won't automatically be remapped to
        # lower values).
        nodata_value: 255
        # Whether to compute mean IoU.
        enable_miou_metric: true
```

For each training example, SegmentationTask computes a target dict containing the
"classes" (ground truth class IDs) and "valid" (mask indicating which pixels are valid
for training) keys.

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
          # Similar to PerPixelRegression, we apply a UNet-style decoder.
          - class_path: rslearn.models.unet.UNetDecoder
            init_args:
              in_channels: [[4, 128], [8, 256], [16, 512], [32, 1024]]
              # Number of output channels, must match the number of classes.
              out_channels: 10
          # The SegmentationHead computes cross entropy loss on valid pixels between
          # the model output and the ground truth class IDs.
          - class_path: rslearn.train.tasks.segmentation.SegmentationHead
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
        bands: ["classes"]
        dtype: INT32
        is_target: true
    task:
      # see example above
```

## Models

rslearn includes a variety of model components that can be composed together, including
remote sensing foundation models like OlmoEarth, decoders like Faster R-CNN, and
intermediate components.

`SingleTaskModel` and `MultiTaskModel` provide a framework for composing encoders and
decoders. `SingleTaskModel` applies a single sequence of decoder components to make a
prediction for one task, while `MultiTaskModel` can be used with `MultiTask` to have
parallel decoders making multiple predictions for training no multiple tasks.

Here is an example of using `SingleTaskModel`:

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.multitask.SingleTaskModel
      init_args:
        encoder:
          # We compose two components in the encoder:
          # (1) A Swin encoder, which processes input images and computes feature maps.
          # (2) An Fpn, which inputs feature maps and outputs updated feature maps.
          - class_path: rslearn.models.swin.Swin
            init_args:
              arch: "swin_v2_b"
              pretrained: true
              input_channels: 9
              output_layers: [1, 3, 5, 7]
          - class_path: rslearn.models.fpn.Fpn
            init_args:
              in_channels: [128, 256, 512, 1024]
              out_channels: 128
        decoder:
          # We also compose two components in the decoder:
          # (1) A Conv layer, which applies a Conv2D on each input feature map.
          # (2) A FasterRCNN to predict bounding boxes.
          - class_path: rslearn.models.conv.Conv
            init_args:
              in_channels: 128
              out_channels: 128
              kernel_size: 3
          - class_path: rslearn.models.faster_rcnn.FasterRCNN
            init_args:
              downsample_factors: [4, 8, 16, 32]
              num_channels: 128
              num_classes: 2
              anchor_sizes: [[32], [64], [128], [256]]
```

This framework is somewhat rigid. The first component in the encoder is the feature
extractor, and should input the input dict from the dataset, which is initialized with
the passthrough DataInputs specified in the model config but then modified by the
transforms. It should output a list of 2D feature maps. For example, Swin inputs an
input dict like this:

```
{
  "image": BxCxHxW tensor,
}
```

It outputs a list of feature maps. With the selected Base architecture (swin_v2_b), and
the configured `output_layers`, this list is like this:

```
[
  B x 128 x (H/4) x (W/4) tensor,
  B x 256 x (H/8) x (W/8) tensor,
  B x 512 x (H/16) x (W/16) tensor,
  B x 1024 x (H/32) x (W/32) tensor,
]
```

In the Python code, the signature of the first encoder is:

```python
    def forward(
        self,
        inputs: list[dict[str, Any]],
    ) -> list[torch.Tensor]:
```

Subsequent components in the encoder should input feature maps and output updated
feature maps. For example, the Fpn (Feature Pyramid Network) inputs the feature map
above and outputs feature maps that have a consistent number of channels
(configured by `out_channels`, which we have set to 128). The `in_channels` specifies
the embedding size of each input feature map, in order. Then, the output is like this:

```
[
  B x 128 x (H/4) x (W/4) tensor,
  B x 128 x (H/8) x (W/8) tensor,
  B x 128 x (H/16) x (W/16) tensor,
  B x 128 x (H/32) x (W/32) tensor,
]
```

The signature of subsequent encoder components is:

```python
    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
```

In the decoder, all components except the last should input feature maps, along with
the original inputs to the model, and output updated feature maps. For example, the
Conv component applies the same `nn.Conv2d` layer on each input feature map, producing
an output with the same shapes. The signature of these decoder components is:

```python
    def forward(
        self,
        features: list[torch.Tensor],
        inputs: list[dict[str, Any]],
    ) -> list[torch.Tensor]:
```

Note that several components, including Conv, ignore the inputs.

The final component is a predictor that should accept the targets, and compute outputs
and the loss(es). For example, the output from Faster R-CNN is a list of dicts with the
"boxes", "scores", and "labels" keys. It outputs a loss dict with the "rpn_box_reg",
"objectness", "classifier", and "box_reg" keys. These will be logged separately, but
are summed for computing gradients during training. The signature of the final decoder
component is:

```python
    def forward(
        self,
        features: list[torch.Tensor],
        inputs: list[dict[str, Any]],
        targets: list[dict[str, Any]] | None = None,
    ) -> tuple[list[Any], dict[str, torch.Tensor]]:
```

## Feature Extractors

### Foundation Models

Several remote sensing foundation models are included in rslearn, and can be used as
the first component in the encoder list (the feature extractor).

- [SatlasPretrain](SatlasPretrain.md)
- TODO

### SimpleTimeSeries

SimpleTimeSeries wraps a unitemporal feature extractor and applies it on a time series.
It encodes each image in the time series individually using the unitemporal feature
extractor, and then pools the features temporally via max pooling, mean pooling, a
ConvRNN, 3D convolutions, or 1D convolutions.

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
              # One of "max" (default), "mean", "convrnn", "conv3d", or "conv1d".
              op: "max"
              # Number of layers for convrnn, conv3d, and conv1d ops.
              num_layers: null
              # A map from input dict keys to the number of bands per image. This is
              # used to split up the time series back into the individual images.
              image_keys:
                sentinel2: 12
                sentinel1: 2
          - ...
```

The [main README](../README.md) has an example of using SimpleTimeSeries with
SatlasPretrain.

## Encoder Components

This section documents model components that can be used in the encoder, after the
initial feature extractor.

### Feature Pyramid Network

Fpn implements a Feature Pyramid Network (FPN). The FPN inputs a multi-scale feature
map. At each scale, it computes new features of a configurable depth based on all input
features. So it is best used for maps that were computed sequentially, where earlier
features don't have the context from later features, but comprehensive features at each
resolution are desired.

Here is a summary, see `rslearn.models.fpn` for all of the available
options.

```yaml
        encoder:
          - # ...
          - class_path: rslearn.models.fpn.Fpn
            init_args:
              # in_channels lists the number of channels in each feature map from the
              # previous component. In this example, there are two feature maps, the
              # first with 128 channels and the second with 256 channels.
              in_channels: [128, 256]
              # The number of output channels. Since there are two feature maps in the
              # input, the output will have two feature maps at the same resolutions,
              # but with 128 channels.
              out_channels: 128
```

It is most often used for object detection tasks in conjunction with Faster R-CNN or
similar bounding box predictors. Here is an example:

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
              input_channels: 3
              # These are the typical feature maps used from Swin. They are at 1/4, 1/8,
              # 1/16, and 1/32 of the input resolution.
              output_layers: [1, 3, 5, 7]
          - class_path: rslearn.models.fpn.Fpn
            init_args:
              in_channels: [128, 256, 512, 1024]
              out_channels: 128
        decoder:
          # Since we have applied the FPN, the input to the Faster R-CNN has 128
          # channels at each resolution.
          - class_path: rslearn.models.faster_rcnn.FasterRCNN
            init_args:
              downsample_factors: [4, 8, 16, 32]
              num_channels: 128
              num_classes: 10
              anchor_sizes: [[32], [64], [128], [256]]
```

## Decoder Components

The predictors (final decoder components) are documented with the tasks. Here, we
document the available decoder components before the predictor.

### PickFeatures

`PickFeatures` picks a subset of feature maps from a multi-scale feature map list to pass
to the next component.

Here is a summary, see `rslearn.models.pick_features` for all of the available
options.

```yaml
        decoder:
          - class_path: rslearn.models.pick_features.PickFeatures
            init_args:
              # The indexes of the input feature map list to select.
              # In this example, we select only the first feature map.
              indexes: [0]
```

### PoolingDecoder

`PoolingDecoder` computes a flat vector from a 2D feature map.

It inputs multi-scale features, but only uses the last feature map. Then it applies a
configurable number of convolutional layers before pooling, and a configurable number
of fully connected layers after pooling.

The output is a vector and not a list of feature maps, so the next component is
typically a predictor (either `ClassificationHead` or `RegressionHead`).

Here is a summary, see `rslearn.models.pick_features` for all of the available
options.

```yaml
        decoder:
          - class_path: rslearn.models.pooling_decoder.PoolingDecoder
            init_args:
              # The number of channels in the input (specifically, the last feature map
              # in the list).
              in_channels: 1024
              # The number of output channels. This is typically tied to the task, e.g.
              # if there will be 8 classes then this should be 8.
              out_channels: 8
              # The number of extra convolutional layers to apply before pooling. The
              # default is 0.
              num_conv_layers: 0
              # The number of fully connected layers to apply after pooling. The
              # default is 0.
              num_fc_layers: 0
              # Number of hidden channels when using num_conv_layers / num_fc_layers.
              conv_channels: 128
              fc_channels: 512
          # This is an example for using PoolingDecoder with a classification task.
          - class_path: rslearn.train.tasks.classification.ClassificationHead
```

### Conv

`Conv` implements a standard 2D convolutional layer.

If there are multiple input feature maps, the same weights are convolved with each
feature map.

```yaml
        decoder:
          - class_path: rslearn.models.conv.Conv
            init_args:
              # The number of input channels. If there are multiple feature maps, they
              # can have different resolutions, but must all have the same number of
              # channels.
              in_channels: 128
              # The number of output channels.
              out_channels: 64
              # The kernel size, stride, and padding. See torch.nn.Conv2D.
              # The stride defaults to 1 and the padding defaults to "same", while
              # kernel_size must be configured. "same" padding keeps the same
              # resolution as the input. If stride is not 1, then padding must be set
              # since "same" is only accepted when the stride is 1.
              kernel_size: 3
              stride: 1
              padding: "same"
              # The activation to use. It defaults to ReLU.
              activation:
                class_path: torch.nn.ReLU
          # ...
```
