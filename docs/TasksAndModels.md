## Tasks and Models

This document details the tasks and model components available in rslearn.

## Tasks

Currently, all rslearn tasks are for supervised training for different types of
predictions (classification, bounding box detection, segmentation, etc.). Tasks
implement a `process_inputs` function that computes targets suitable for training from
the input dict, and a `process_output` function that computes raster or vector outputs
(either a CHW tensor or list of vector features) from the model output. All tasks
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

See [ModelConfig.md](./ModelConfig.md#inputs) for details about how to configure the
inputs section.

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

In `process_inputs`, ClassificationTask computes a target dict containing the "class"
(class ID) and "valid" (flag indicating whether it is valid) keys from the vector
targets.

In `process_output`, the output from the model must be a BxC tensor of predicted
probabilities for each class. A list of vector features is returned, where the geometry
of each feature corresponds to the input patch, and where each feature has a property
matching `property_name` containing the class name. Typically, the model output is
computed through the ClassificationHead predictor.

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

In `process_inputs`, DetectionTask computes a target dict containing the "boxes"
(bounding box coordinates), "labels" (class labels), "valid" (flag indicating whether
the example is valid), and "width"/"height" (window width and height) keys.

In `process_output`, the expected output from the model is a list of dicts (one dict
per example in the batch) with the "boxes", "scores", and "labels" keys:

- boxes: a (N, 4) float tensor, where N is the number of predicted boxes for this example,
  containing the predicted bounding box coordinates. The coordinates are in
  (x1, y1, x2, y2) order, and in relative pixel coordinates corresponding to the input
  resolution.
- scores: a (N,) float tensor containing the output probabilities.
- labels: a (N,) integer tensor containing the predicted class ID for each box.

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
	      class_path: rslearn.train.tasks.per_pixel_regression.PerPixelRegressionTask
	      init_args:
	        # Multiply ground truth values by this factor before using it for training.
	        scale_factor: 0.1
	        # Metric(s) to compute. Either a string or a list.
	        # Supported: "mse", "l1", "r2".
	        metric_mode: ["mse", "r2"]
	        # Optional value to treat as invalid. The loss will be masked at pixels where
	        # the ground truth value is equal to nodata_value.
	        nodata_value: -1
```

In `process_inputs`, PerPixelRegressionTask computes a target dict containing the
"values" (scaled ground truth values) and "valid" (mask indicating which pixels are
valid for training) keys.

In `process_output`, the output from the model must be a BHW tensor containing the
predicted scaled value for each pixel. The unscaled raster is returned, with a
singleton channel dimension. Typically, the model output is computed through the
PerPixelRegressionHead predictor.

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
              # The loss function to use: "mse" (default), "l1", or "huber".
              loss_mode: "mse"
              # Optional: delta for Huber loss (only used when loss_mode="huber").
              huber_delta: 1.0
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

In `process_inputs`, RegressionTask computes a target dict containing the "value"
(ground truth regression value) and "valid" (flag indicating whether the sample is
valid) keys.

In `process_output`, the output from the model must be a single-dimension tensor
containing the predicted scaled value for each example in the batch. A list of vector
features is returned, where the geometry of each feature corresponds to the input
patch, and where each feature has a property matching `property_name` containing the
unscaled value. Typically, the model output is computed through the RegressionHead
predictor.

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

In `process_inputs`, SegmentationTask computes a target dict containing the "classes"
(ground truth class IDs) and "valid" (mask indicating which pixels are valid for
training) keys.

In `process_output`, the output from the model must be a BCHW tensor containing the
predicted class probabilities for each pixel. A raster with a singleton channel
dimension is returned, containing the highest probability class IDs at each pixel.
Typically, the model output is computed through the SegmentationHead predictor.

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

### Introduction

rslearn includes a variety of model components that can be composed together, including
feature extractors like OlmoEarth, predictors like Faster R-CNN, and intermediate
components.

`SingleTaskModel` and `MultiTaskModel` provide a framework for composing feature
extractors, intermediate components, and predictors. These are composed into one
encoder (feature extractor plus an arbitrary number of intermediate components) and one
or more decoders (arbitrary intermediate components plus one predictor).
`SingleTaskModel` applies a single sequence of decoder components to make a prediction
for one task, while `MultiTaskModel` can be used with `MultiTask` to have parallel
decoders making multiple predictions for training on multiple tasks.

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
          # (1) A Swin feature extractor, which processes input images and computes FeatureMaps.
          # (2) An Fpn, which inputs a FeatureMaps and outputs updated FeatureMaps.
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
          # (1) A Conv layer, which applies a Conv2D on each feature map in the input FeatureMaps.
          # (2) A FasterRCNN that inputs FeatureMaps and predicts bounding boxes.
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

#### Intermediate Types

Feature extractors and intermediate components can output arbitrary types. However, all
components in rslearn output one of the following types, defined in
`rslearn.models.component`.

- FeatureMaps: a list of multi-scale feature maps. Each feature map is a BCHW tensor,
  where the channel dimension contains the features (embeddings).
- FeatureVector: a flat feature vector. It consists of a single BxC tensor.

#### Feature Extractor (First Encoder Component)

This framework is somewhat rigid. The first component in the encoder is the feature
extractor. It inputs a ModelContext object, which includes the list of input dicts from
the dataset (one input dict per example) and a corresponding list of SampleMetadatas
(describes the location and time range of each example). The input dicts are
initialized with the passthrough DataInputs specified in the model config but then
modified by the transforms. The output of the feature extractor can be an arbitrary
type.

It should output a list of 2D feature maps.

For example, Swin requires an input dict list like this:

```
[
  {
    "image": CxHxW tensor,
  },
  ... (B dicts)
]
```

It outputs a FeatureMaps. With the selected Base architecture (swin_v2_b) and the
configured `output_layers` in the example above, the FeatureMaps is like this:

```
[
  B x 128 x (H/4) x (W/4) tensor,
  B x 256 x (H/8) x (W/8) tensor,
  B x 512 x (H/16) x (W/16) tensor,
  B x 1024 x (H/32) x (W/32) tensor,
]
```

Above, B is the batch size, H/W are the input image height/width, and C is the number
of channels in the input image. 128, 256, 512, and 1024 are the embedding sizes from
Swin-Base at different resolutions.

In the Python code, the FeatureExtractor provides a forward function with this
signature:

```python
def forward(self, context: ModelContext) -> Any:
```

ModelContext is defined as follows in `rslearn.train.model_context`:

```python
class ModelContext:
    """Context to pass to all model components."""

    # One input dict per example in the batch.
    inputs: list[dict[str, torch.Tensor]]
    # One SampleMetadata per example in the batch.
    metadatas: list[SampleMetadata]
    # Arbitrary dict that components can add to.
    context_dict: dict[str, Any] = field(default_factory=lambda: {})
```

#### Intermediate Components

Intermediate components input and output arbitrary types. They can be used as elements
of the encoder after the FeatureExtractor, or elements of the decoder(s) before the
final Predictor.

For example, the Fpn (Feature Pyramid Network) inputs a FeatureMaps and outputs an
updated FeatureMaps that has a consistent number of channels (configured by
`out_channels`, which we have set to 128 in the example above). The `in_channels` specifies
the embedding size of each input feature map, in order. Then, in the above example, the
output of Fpn would be like this:

```
[
  B x 128 x (H/4) x (W/4) tensor,
  B x 128 x (H/8) x (W/8) tensor,
  B x 128 x (H/16) x (W/16) tensor,
  B x 128 x (H/32) x (W/32) tensor,
]
```

The IntermediateComponent provides a forward function with this signature:

```python
def forward(self, intermediates: Any, context: ModelContext) -> Any:
```

#### Predictor (Final Decoder Component)

The predictor accepts the targets, along with the arbitrary output of the previous
model component, and computes outputs compatible with the configured Task, along with
loss(es). The targets are those computed by the Task's `process_inputs` function, so
their form would depend on the configured Task.

For example, the output from Faster R-CNN is a list of dicts with the "boxes",
"scores", and "labels" keys. It outputs a loss dict with the "rpn_box_reg",
"objectness", "classifier", and "box_reg" keys. These will be logged separately, but
are summed for computing gradients during training.

The Predictor provides a forward function with this signature:

```python
def forward(
    self,
    intermediates: Any,
    context: ModelContext,
    targets: list[dict[str, torch.Tensor]] | None = None,
) -> ModelOutput:
```

`targets` is a list with one target dict per example in the batch, or None during the
predict stage.

ModelOutput is defined as follows in `rslearn.train.model_context`:

```python
class ModelOutput:
    """The output from the Predictor.

    Args:
        outputs: output compatible with the configured Task.
        loss_dict: map from loss names to scalar tensors.
        metadata: arbitrary dict that can be used to store other outputs.
    """

    outputs: Iterable[Any]
    loss_dict: dict[str, torch.Tensor]
    metadata: dict[str, Any] = field(default_factory=lambda: {})
```

## Feature Extractors

### Foundation Models

Several remote sensing foundation models are included in rslearn, and can be used as
the first component in the encoder list (the feature extractor).

- [OlmoEarth](foundation_models/OlmoEarth.md)
- [SatlasPretrain](foundation_models/SatlasPretrain.md)

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

## Intermediate Components

This section documents intermediate model components that can be used in the
encoder/decoder between the FeatureExtractor and the Predictor.

### Feature Pyramid Network

Fpn implements a Feature Pyramid Network (FPN). The FPN inputs a FeatureMaps. At each
scale, it computes new features of a configurable depth based on all input features. So
it is best used for maps that were computed sequentially, where earlier features don't
have the context from later features, but comprehensive features at each resolution are
desired.

Here is a summary, see `rslearn.models.fpn` for all of the available options.

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

### PickFeatures

`PickFeatures` picks a subset of feature maps from a FeatureMaps to pass to the next
component. It outputs the updated FeatureMaps list.

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

`PoolingDecoder` computes a FeatureVector from a FeatureMaps.

It inputs a FeatureMaps, but only uses the last feature map. Then it applies a
configurable number of convolutional layers before pooling, and a configurable number
of fully connected layers after pooling.

The output is a FeatureVector. Most intermediate components currently input a
FeatureMaps, so the next component is typically a predictor (either
`ClassificationHead` or `RegressionHead`).

Here is a summary, see `rslearn.models.pooling_decoder` for all of the available
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

It inputs a FeatureMaps. If there are multiple input feature maps, the same weights are
convolved with each feature map.

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

## Predictors

### ClassificationHead

ClassificationHead computes cross entropy loss given the logits and targets. It does
not take any arguments.

It inputs a FeatureVector of logits, where the channel dimension size must match the
number of classes. It outputs the class probabilities after applying softmax on those
input logits. It also produces a loss dict with one key, "cls", containing the softmax
cross entropy loss.

### PerPixelRegressionHead

PerPixelRegressionHead computes a per-pixel regression loss (MSE, L1, or Huber). It is
configured like this:

```yaml
        decoder:
          # ...
          - class_path: rslearn.train.tasks.per_pixel_regression.PerPixelRegressionHead
            init_args:
              # The loss function to use: "mse" (default), "l1", or "huber".
              loss_mode: "mse"
              # Optional: delta for Huber loss (only used when loss_mode="huber").
              huber_delta: 1.0
              # Whether to apply a sigmoid activation on the output. This requires the
              # targets to be between 0-1. Otherwise, the previous output is
              # unmodified.
              use_sigmoid: false
```

It inputs a FeatureMaps, which must contain a single feature map consisting of the
predicted values at each pixel. If `use_sigmoid` is false, those should correspond to
the scaled values (actual value multiplied by the scale factor configured in the task).

It outputs the scaled values as a BHW tensor. It also produces a loss dict with one
key, "regress", containing the configured regression loss.

### RegressionHead

RegressionHead computes a regression loss (MSE, L1, or Huber). It is configured like
this:

```yaml
        decoder:
          # ...
          - class_path: rslearn.train.tasks.regression.RegressionHead
            init_args:
              # The loss function to use: "mse" (default), "l1", or "huber".
              loss_mode: "mse"
              # Optional: delta for Huber loss (only used when loss_mode="huber").
              huber_delta: 1.0
              # Whether to apply a sigmoid activation on the output. This requires the
              # targets to be between 0-1. Otherwise, the previous output is
              # unmodified.
              use_sigmoid: false
```

It inputs a FeatureVector containing the predicted values for each example in the
batch. If `use_sigmoid` is false, those should correspond to the scaled values (actual
value multiplied by the scale factor configured in the task).

It outputs the scaled values as a single-dimension tensor. It also produces a loss dict
with one key, "regress", containing the configured regression loss.

### SegmentationHead

SegmentationHead computes cross entropy loss given the logits and targets. It does not
take any arguments.

It inputs a FeatureMaps, which must contain a single feature map of logits, with the
channel dimension size matching the number of classes. It outputs the class
probabilities after applying softmax on those input logits. It also produces a loss
dict with one key, "cls", containing the softmax cross entropy loss.
