## Model Configuration File

Typically, model training in rslearn involves fine-tuning remote sensing foundation
models for prediction tasks encapsulated in an rslearn dataset. Typically, the dataset
will contain one or more satellite image layers that contain the input that should be
passed to the model; these could be individual images, or image time series. The
dataset should also contain one or more raster or vector layers that act as labels, so
that the model can be trained to make predictions corresponding to those layers.

The model configuration file is a PyTorch Lightning-style YAML file that defines a
model architecture, the tasks that the model should be trained for, and how the
training should interface with the underlying rslearn dataset.

The overall configuration looks like this:

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      # ... Model architecture configuration.
    # ... Learning rate, scheduler, and other options.
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    path: # path to the rslearn dataset.
    task:
      # ... the task that defines how to train the model.
    inputs:
      # ... how layers in the dataset should be used
    # other data related options
trainer:
  # Lightning trainer options and callbacks.
```

The YAML is parsed by jsonargparse, so each section directly configures a Python class
or other type, and you can check those classes in the rslearn codebase for more
details.

Below, we dive into each of these components. This documentation is intended to
supplement the examples in [Examples](Examples.md).

## Model Section

The model section configures the RslearnLightningModule, which is intended to be
flexible enough for most supervised fine-tuning tasks.

Here is a summary of all of the options available in RslearnLightningModule:

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      # ... Model architecture configuration.
    optimizer:
      # ... Optimizer configuration.
    scheduler:
      # ... Learning rate scheduler configuration.
    visualize_dir: null
    metrics_file: null
    restore_config:
      # ... initial weight configuration
    print_parameters: false
    print_model: false
```

### model

The model argument is a `torch.nn.Module` that corresponds to the model architecture.
Typically, models will either employ SingleTaskModel (when training a model for one
task, like segmentation or detection) or MultiTaskModel (when training a model on
multiple tasks, e.g. predicting LFMC at each pixel while also classifying land cover at
each pixel). These classes provide scaffolding that makes foundation models like
OlmoEarth and SatlasPretrain compatible with decoders like U-Net and Faster R-CNN.

Here is an example with SingleTaskModel, for segmentation:

```yaml
    model:
      class_path: rslearn.models.singletask.SingleTaskModel
      init_args:
        encoder:
          # This section specifies a list of model components that should extract one
          # or more feature maps from the inputs. The components will be applied
          # sequentially.
          - class_path: rslearn.models.satlaspretrain.SatlasPretrain
            init_args:
              model_identifier: "Sentinel2_SwinB_SI_RGB"
        decoder:
          # This section specifies a list of model components for the decoder. The
          # final components in the list should compute a loss that can be optimized.
          # The UNetDecoder inputs a set of multi-scale feature maps, and produces
          # logits at the input resolution.
          - class_path: rslearn.models.unet.UNetDecoder
              init_args:
              in_channels: [[4, 128], [8, 256], [16, 512], [32, 1024]]
              out_channels: 2
              conv_layers_per_resolution: 2
          # The SegmentationHead computes the cross entropy loss between the logits and
          # the labels at each pixel. It also returns the output probabilities at each
          # pixel.
          - class_path: rslearn.train.tasks.segmentation.SegmentationHead
```

Here is an example with MultiTaskModel, for per-pixel regression + segmentation:

```yaml
    model:
      class_path: rslearn.models.singletask.MultiTaskModel
      init_args:
        encoder:
          # The encoder section is the same as with SingleTaskModel.
          - class_path: rslearn.models.satlaspretrain.SatlasPretrain
            init_args:
              model_identifier: "Sentinel2_SwinB_SI_RGB"
        decoders:
          # The decoder section now has a separate list of model components for each
          # prediction task. The keys for each decoder section are sub-task names which
          # can be arbitrary, but must match up with corresponding keys in MultiTask in
          # the data: section.
          regress:
            - class_path: rslearn.models.unet.UNetDecoder
                init_args:
                in_channels: [[4, 128], [8, 256], [16, 512], [32, 1024]]
                out_channels: 1
                conv_layers_per_resolution: 2
            - class_path: rslearn.train.tasks.per_pixel_regression.PerPixelRegressionHead
          segment:
            - class_path: rslearn.models.unet.UNetDecoder
                init_args:
                in_channels: [[4, 128], [8, 256], [16, 512], [32, 1024]]
                out_channels: 2
                conv_layers_per_resolution: 2
            - class_path: rslearn.train.tasks.segmentation.SegmentationHead
```

There are many foundation models, decoders, and miscellaneous model components
available. These are documented in [TasksAndModels](TasksAndModels.md).

### optimizer

This section defines the optimizer that should be used.

Currently, the only optimizer included in rslearn is AdamW.

```yaml
    optimizer:
      class_path: rslearn.train.optimizer.AdamW
      init_args:
        # Initial learning rate.
        lr: 0.001
        # Betas.
        betas: [0.9, 0.999]
        # Optional epsilon, see torch.optim.AdamW.
        eps: null
        # Optional weight decay, see torch.optim.AdamW.
        weight_decay: null
```

### scheduler

This section defines the learning rate scheduler.

There are three schedulers implemented in `rslearn.train.scheduler`:

- PlateauScheduler: wraps `torch.optim.lr_scheduler.ReduceLROnPlateau`
- CosineAnnealingScheduler: wraps `torch.optim.lr_scheduler.CosineAnnealingLR`
- CosineAnnealingWarmRestartsScheduler: wraps `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`

Here is an example of using PlateauScheduler:

```yaml
    scheduler:
      class_path: rslearn.train.scheduler.PlateauScheduler
      init_args:
        # Multiply the learning rate by this factor on plateau.
        factor: 0.2
        # Number of epochs without improvement after which learning rate should be reduced.
        patience: 2
        # Number of epochs to wait before resuming normal operation after learning rate has been reduced.
        cooldown: 10
```

For all of the available options, see `rslearn.train.scheduler` and `torch.optim.lr_scheduler`.

### visualize_dir

This can be set to a string containing a directory path. If set, during the test stage,
visualizations of the model outputs will be saved to this directory. The visualizations
are controlled by the task object.

### metrics_file

This can be set to a string containing a filename. If set, during the test stage, the
final test metrics will be saved to this file. The metrics that are enabled are
controlled by the task object.

### restore_config

This configures a PyTorch state dictionary file (`.pt` or `.pth`) from which some or
all of the parameters of the model architecture should be initialized.

This typically is not needed since most of the foundation models included in rslearn
automatically load weights from checkpoints (e.g. from Hugging Face Hub).

Here is an example for configuring a Swin-Base model, and using restore_config to load
an older SatlasPretrain checkpoint.

```yaml
    model:
      class_path: rslearn.models.multitask.MultiTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.swin.Swin
            init_args:
              pretrained: true
              input_channels: 9
              output_layers: [1, 3, 5, 7]
        decoders:
          # ...
    restore_config:
      restore_path: https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-model-v1-lowres-band-multi.pth
      remap_prefixes:
        - ["backbone.backbone.backbone.", "encoder.0.model."]
      ignore_prefixes: []
      selector: []
```

The `selector` is used in case the state dictionary is contained within a key in the
overall file. For example, if `torch.load("ckpt.pt")["state_dict"]` contains the state
dictionary, then `selector` can be set to `["state_dict"]`. Multiple list elements can
be used in case the state dictionary is buried under multiple levels. In the example
above, the pth file contains the state dictionary at the top-level, so we leave
`selector` as the default empty list.

`remap_prefixes` will remap prefixes of keys from one name to another. In the example,
any keys starting with "backbone.backbone.backbone.X" will be renamed to
"encoder.0.model.X".

`ignore_prefixes` will drop any keys that start with those prefixes. This is only
needed if the shape of the tensor at that key doesn't match the expected shape in the
model defined in the config.

### print_parameters and print_model

These options are useful for debugging.

- print_parameters: if set true (default false), prints the shape of all parameters
  registered with torch upon initialization.
- print_model: if set true (default false), prints the model architecture by calling
  `print(self.model)` upon initialization. The output is formatted by torch.

## Data Section

The data section defines the RslearnDataModule, which provides PyTorch Lightning with
the torch dataset and data loader to use for training, validation, testing, and
prediction.

Here is a summary of all of the options RslearnDataModule provides:

```yaml
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    inputs:
      # Specify DataInputs that define how to load data from the rslearn dataset.
    task:
      # The Task object that defines many parts of the supervised training task.
    path: "..." # path to the rslearn dataset
    batch_size: 1 # batch size
    num_workers: 0 # number of data loader workers
    init_workers: 0 # number of torch dataset initialization workers
    default_config:
      # SplitConfig shared across model stages.
    train_config:
      # Override SplitConfig options for training.
    val_config:
      # Override SplitConfig options for validation.
    test_config:
      # Override SplitConfig options for testing.
    predict_config:
      # Override SplitConfig options for prediction.
    retries: 0 # number of retries for __getitem__ in case the dataset is stored on flaky remote storage
```

The `DataInput` and `SplitConfig` configure classes defined in `rslearn.train.dataset`.

We detail each of the options below.

### inputs

This is a list of DataInput objects that define how to read data from the underlying
rslearn dataset.

Here is an example for a simple single-task training setup that inputs one modality:

```yaml
    inputs:
      # The top-level names "image" and "targets" can be arbitrary, but should
      # correspond to what the model and/or task expects, or to other parts of the
      # model configuration file.
      image:
        # Either "raster" or "vector".
        data_type: "raster"
        # The layer names in the rslearn dataset that should be read from.
        layers: ["sentinel2"]
        # The bands to read.
        bands: ["B04", "B03", "B02", "B05", "B06", "B07", "B08", "B11", "B12"]
        # If true, examples not containing the layers needed to read this input are
        # skipped. This should generally be left enabled (default).
        required: true
        # Currently, this option should be enabled for all inputs that are not used as
        # targets for training, including inputs to the model. Targets do not need
        # passthrough since they will be processed by the Task object into label dicts.
        passthrough: true
        # Whether this is a target that should only be read during the train, validate,
        # and test stages. If enabled, this input will be skipped for prediction, and
        # windows not containing it will not be skipped (even if required is true).
        is_target: false
        # For rasters, the data type that it should be cast to after reading.
        dtype: FLOAT32
        # The default behavior is to load a random layer listed in the layers option.
        # If load_all_layers is enabled, then rslearn will instead load all of the
        # layers, and concatenate (for raster data) or merge (for vector data) the data
        # across layers. Additionally, examples missing any one of those layers will be
        # skipped, instead of only skipping examples missing all of those layers.
        load_all_layers: false
        # If enabled, any layer appearing in the layers option will be expanded to
        # include all of its item groups.
        # Note that, when this option is false, the elements of layers are treated as
        # item groups, but when true, they are treated as overall layers.
        load_all_item_groups: false
      targets:
        data_type: "vector"
        layers: ["label"]
        required: true
        passthrough: false
        is_target: true
        load_all_layers: false
        load_all_item_groups: false
```

For raster data, the `bands` and `dtype` options are required, but they can be omitted
for vector data.

Time series in rslearn are represented as (T*C, H, W) tensors, where the timesteps are
concatenated on the channel dimension. Often, you may have an rslearn dataset with each
timestep in a different item group of the same layer, like "sentinel2", "sentinel2.1",
"sentinel2.2".

If `layers` is set to `["sentinel2"]`, and `load_all_layers` and `load_all_item_groups`
are both false, then only the first item group will be read.

If `layers` explicitly specifies all three item groups (set to
`["sentinel2", "sentinel2.1", "sentinel2.2]"`), and `load_all_layers` and
`load_all_item_groups` are both false, then for each `__getitem__` call, rslearn will
pick a random layer from the list, and only load that one image.

If `layers` is set to `["sentinel2"]`", and `load_all_item_groups` is set, then it is
equivalent to above: the layer will be expanded into its item groups, and one item
group will be randomly picked.

If `layers` is set to `["sentinel2"]`, and `load_all_layers` and `load_all_item_groups`
are both true, then the layer will be expanded into its item groups, and the images
from all of the item groups that are available (materialized/completed) for a given
window will be read, and the timesteps will be concatenated. However, if the dataset
contains some windows that have a subset of the item groups, then this can lead to
problems since most foundation models as currently implemented expect all inputs in the
batch to have the same number of timesteps.

In that case, you can set `layers` to explicitly specify all three item groups, and set
`load_all_layers` true but `load_all_item_groups` false. This way, only windows that
contain all three item groups will be used for training.

### task

This configures the Task object See `rslearn.train.tasks` for more details.

The Task object defines a supervised remote sensing task. Currently implemented tasks
in rslearn include:

- ClassificationTask: for image classification (one image-level classification per
  window).
- DetectionTask: for bounding box detection.
- PerPixelRegressionTask: for predicting a real value at each pixel.
- RegressionTask: for image-level regression (predict one real value for the entire
  window).
- SegmentationTask: for semantic segmentation (per-pixel classification).

The Task controls many aspects of training, including:

- How raw inputs read from the rslearn dataset should be converted to target dicts. For
  example, DetectionTask will convert GeoJSON labels into a dictionary of bounding
  boxes and class labels.
- How raw predictions should be converted into outputs that are suitable for writing to
  the rslearn dataset. For example, SegmentationTask will apply argmax on the class
  probabilities so that a GeoTIFF of predicted classes is saved.
- How visualizations should be created, in case visualize_dir is enabled in the model
  section.
- Which metrics should be computed.

[TasksAndModels](TasksAndModels.md) details all of the tasks.

There are a few generic visualization options that are shared across most tasks.

```yaml
    task:
      class_path: ...
      init_args:
        # Which bands from the input image should be used to create an 8-bit PNG.
        # Either one or three bands should be specified. The image is expected to be
        # under the "image" key.
        image_bands: [0, 1, 2]
        # This defines an optional linear scaling from the raw image values to pixel
        # values that should be saved. This is generally needed since usually the input
        # image to the model will have been normalized already.
        remap_values: [[0, 1], [0, 255]]
```

For example, suppose you have a DataInput that loads all Sentinel-2 L2A bands:

```yaml
    inputs:
      sentinel2:
        layers: ["sentinel2"]
        bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
        dtype: FLOAT32
        passthrough: true
```

And let's say you have configured transforms that normalize this input, and also rename
it in the input dict from sentinel2 to image (there is more info about this when we
cover the SplitConfig):

```yaml
    default_config:
      transforms:
        # Normalize the Sentinel-2 image with rough z-score normalization.
        - class_path: rslearn.train.transforms.normalize.Normalize
          init_args:
            mean: 2000
            std: 1500
        # In the input dict, copy it from "sentinel2" to "image".
        - class_path: rslearn.train.transforms.concatenate.Concatenate
          init_args:
            selections:
              sentinel2: []
            output_selector: "image"
```

One good true-color image is to use the red (B04), green (B03), and blue (B02) bands
from 0 to 2500.

Then, we can achieve that in the task with these settings:

```yaml
        # This picks the bands corresponding to B04, B03, and B02.
        image_bands: [3, 2, 1]
        # After normalization, 0 becomes -1.33 and 2500 becomes 0.33.
        # So this setting rescales that to 0-255.
        remap_values: [[-1.33, 0.33], [0, 255]]
```

The MultiTask task can be used along with MultiTaskModel to combine multiple tasks.
Here is an example of its usage.

```yaml
    task:
      class_path: rslearn.train.tasks.multi_task.MultiTask
      init_args:
        tasks:
          # The keys here are sub-task names, which must match those used for the
          # task-specific decoders in the model architecture section.
          regress:
            class_path: rslearn.train.tasks.per_pixel_regression.PerPixelRegressionTask
            init_args:
              # Multiply the regression labels by 0.1 for the purpose of training the
              # model.
              scale_factor: 0.1
              # Compute metric as the L1 (absolute error) between the predicted values
              # and the labels. Note that while the loss operates over the scaled
              # values, the metric operates over the unscaled values.
              metric_mode: "l1"
          segment:
            class_path: rslearn.train.tasks.segmentation.SegmentationTask
            init_args:
              num_classes: 2
              enable_miou_metric: true
        input_mapping:
          # This specifies a per-task remapping from the keys in inputs to keys
          # expected by the task. Currently, all tasks expect the labels to be under
          # the "targets" key. The tasks will process these into a form suitable for
          # training.
          regress:
            # The key here must match the name of the input under the inpust section.
            regress_input: "targets"
            segment_input: "targets
```

### num_workers and init_workers

`num_workers` controls the number of data loader workers.

Worker processes are also used to enumerate the windows in the rslearn dataset, and to
determine which ones contain the layers that are needed. By default, the same number of
workers used for data loader is used for this initialization. However, it can be
overridden by also setting `init_workers`.

By default, `num_workers` is 0, which means the main process is used. Common values
are between 4 and 32, with the goal of having enough data loader workers so that data
loading is not a bottleneck for training the model, but not too many that there is CPU
contention or excessive system memory consumption (especially shared memory).

### SplitConfigs

The `default_config`, `train_config`, `val_config`, `test_config`, and `predict_config`
specify SplitConfigs. The `default_config` defines a template shared across all model
stages, and this can be extended or overwritten by each stage-specific config.

Here is a summary of all of the options available in the SplitConfig.

```yaml
    default_config:
      # Only use windows in these groups in the rslearn dataset.
      # The default is null to read all groups.
      groups: ["group1", "group2"]
      # Only use these window names.
      names: null
      # Only use windows that have matching key-value pairs in their options
      # dictionary. This is often used to separate the windows used for training vs
      # validation and testing, see the main README tutorial for an example of that.
      # All the keys/values here must match.
      tags:
        split: default
      # Limit to training on this many windows from the underlying dataset. This option
      # is mainly used for low data regime experiments.
      num_samples: null
      transforms:
        # List of transforms to apply on the initial input and target dicts.
        - class_path: rslearn.train.transforms.flip.Flip
      # By default, each training epoch simply iterates over the windows in a random
      # order. A different sampler can be used to implement things like weighted random
      # sampling. Here is an example of that.
      sampler:
        class_path: rslearn.train.dataset.WeightedRandomSamplerFactory
        init_args:
          # The key in the window's options dict containing the per-window weights.
          option_key: weight
          # The number of samples per epoch. This does not need to match the number of
          # training windows, since the weighted sampling means we will not see each
          # window on each epoch.
          num_samples: 1000
      # By default (patch_size=null), data for the entire window bounds is read. This
      # can be cropped using transforms, but if a random crop is desired, it is more
      # efficient to crop it with this option, since this way the cropping will happen
      # when reading GeoTIFFs. However, setting it here is less flexible, since it only
      # supports random cropping.
      patch_size: 128
      # For validation, testing, and prediction, patch_size can be combined with
      # load_all_crops to perform sliding window inference. For training, it should
      # usually be left false so that each training epoch sees a different crop.
      load_all_patches: false
      # This should typically be enabled for predict_config, so that windows without
      # layers containing targets are skipped. For training, validation, and testing,
      # targets are needed so it should be false.
      skip_targets: false
```

The transforms will adjust the initial input and target dicts that come from reading
raw inputs from the rslearn dataset (based on the defined DataInputs), and processing
the ones that are targets through the Task object.

Some transforms can be used to perform normalization or renaming of inputs so that they
match with the keys expected by foundation models or by decoders. Other transforms are
used as augmentations.

Here is an example where a Sentinel-2 image time series is intiially loaded under
sentinel2_l2a, which is the key expected by the OlmoEarth pre-trained model. But we
also copy it to the "image" key so that it can be accessed when creating visualizations
(by the Task object), and since the Faster R-CNN also expects an "image" key to know
the size of the original input images. We also show a flipping augmentation, which
needs to flip the boxes and not just the images.

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.multitask.SingleTaskModel
      init_args:
        encoder:
          - class_path: rslp.olmoearth_pretrain.model.OlmoEarth
            init_args:
              forward_kwargs:
                patch_size: 4
        decoder:
          - class_path: rslearn.models.faster_rcnn.FasterRCNN
            init_args:
              downsample_factors: [4]
              num_channels: 768
              num_classes: 3
              anchor_sizes: [[32]]
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    inputs:
      sentinel2_l2a:
        layers: ["sentinel2"]
        bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
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
        classes: ["unknown", "class1", "class2"]
    train_configs:
      # Flip the images and boxes.
      - class_path: rslearn.train.transforms.flip.Flip
        # This must match the name of the input defined above.
        image_selectors: ["sentinel2_l2a"]
        # This tells Flip to also flip boxes in the target dict.
        box_selectors: ["targets/"]
      # Copy the sentinel2_l2a image to the "image" key since, while OlmoEarth model
      # expects it under "sentinel2_l2a", Faster R-CNN and the task visualization will
      # look for it under "image".
      - class_path: rslearn.train.transforms.concatenate.Concatenate
        selections:
          sentinel2_l2a: []
        output_selector: image
```

The selectors in the transforms refer to keys under the input or target dicts. A
selector beginning with "inputs/" references the input dict, while a selector beginning
with "targets/" references the target dict; if it doesn't begin with either, it is
assumed to be selecting the input dict. Above, the target dict is initially populated
with boxes by the DetectionTask, while the "sentinel2_l2a" key in the input dict is
passed through from the raw inputs.

[Transforms](Transforms.md) covers the transforms built in to rslearn.

## Trainer Section

The Trainer offers miscellaneous options to configure the training procedure. It is
unmodified from the Lightning Trainer, so see
https://lightning.ai/docs/pytorch/stable/common/trainer.html#init for the available
options.

Common options are summarized below:

```yaml
trainer:
  # Train for up to this many epochs. It will train for this many epochs unless an
  # early stopping callback is used.
  max_epochs: 100
  # Lightning uses callbacks to perform various actions during training, like saving
  # model checkpoints. rslearn also includes some custom checkpoints for things like
  # writing predictions to the rslearn dataset during the prediction stage.
  callbacks:
    # We show some common callbacks here.
    # The LearningRateMonitor will log the current learning rate. We configure it to
    # log once per epoch.
    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
      init_args:
        logging_interval: "epoch"
    # The RslearnWriter is responsible for saving predictions to the rslearn dataset.
    # It is only active during the predict stage.
    - class_path: rslearn.train.prediction_writer.RslearnWriter
      init_args:
        # This can be left as a placehloder -- rslearn will override it with the
        # rslearn dataset path from data.init_args.path.
        path: placeholder
        # This is the name of the layer in the rslearn dataset under which the
        # predictions should be saved. It must exist in the dataset config.
        output_layer: output
        # This defines how to access the output that should be saved from the
        # dictionary returned by the model. When using SingleTaskModel, this option can
        # generally be omitted. When using MultiTaskModel, this option should usually
        # match with the sub-task name.
        selector: ["detect"]
    # The ModelCheckpoint callback saves model checkpoints.
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        # We keep the checkpoint that has the maximum mAP value on the detect task.
        save_top_k: 1
        monitor: val_detect/mAP
        mode: max
        # We also keep the latest checkpoint.
        save_last: true
  # The logger can be set to log to something other than the local filesystem, like
  # W&B.
  logger:
    class_path: lightning.pytorch.loggers.WandbLogger
    init_args:
      # This is the W&B project name, and run name.
      # You could set entity here as well, otherwise it will use the default based on
      # the API key being used.
      project: land_cover_model
      name: version_00
```

## Using Custom Classes

Each section of the model configuration file that specifies a class instantiation can
be changed to instantiate a custom class.

For example, you could develop a new optimizer class:

```python
import lightning as L
import torch.optim
from rslearn.train.optimizer import OptimizerFactory
from torch.optim import Optimizer

@dataclass
class Adadelta(OptimizerFactory):
    """Factory for Adadelta optimzier."""

    lr: float = 0.001
    rho: float | None = None
    eps: float | None = None
    weight_decay: float | None = None

    def build(self, lm: L.LightningModule) -> Optimizer:
        """Build the Adadelta optimizer."""
        params = [p for p in lm.parameters() if p.requires_grad]
        kwargs = {k: v for k, v in asdict(self).items() if v is not None}
        return torch.optim.Adadelta(params, **kwargs)

```

Suppose this is in `your_pkg.optimizer`. Then, you can configure it as follows:

```yaml
model
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    # ...
    optimizer:
      class_path: your_pkg.optimizer.Adadelta
      init_args:
        lr: 0.0001
```
