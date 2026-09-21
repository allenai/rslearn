## rslearn.models.fpn.Fpn

Fpn implements a Feature Pyramid Network (FPN). The FPN inputs a FeatureMaps. At each
scale, it computes new features of a configurable depth based on all input features. So
it is best used for maps that were computed sequentially, where earlier features don't
have the context from later features, but comprehensive features at each resolution are
desired.

It inputs a FeatureMaps and outputs an updated FeatureMaps with the same number of
feature maps at the same resolutions, but where every feature map has `out_channels`
channels.

### Configuration

Here is a summary, see `rslearn.models.fpn` for all of the available options.

```yaml
        encoder:
          - # ...
          - class_path: rslearn.models.fpn.Fpn
            init_args:
              # in_channels lists the number of channels in each feature map
              # from the previous component. In this example, there are two
              # feature maps, the first with 128 channels and the second with
              # 256 channels.
              in_channels: [128, 256]
              # The number of output channels. Since there are two feature maps
              # in the input, the output will have two feature maps at the same
              # resolutions, but with 128 channels.
              out_channels: 128
```

### Example

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
              # These are the typical feature maps used from Swin. They are at
              # 1/4, 1/8, 1/16, and 1/32 of the input resolution.
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
