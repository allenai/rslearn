## rslearn.models.conv.Conv

Conv implements a standard 2D convolutional layer.

It inputs a FeatureMaps. If there are multiple input feature maps, the same weights are
convolved with each feature map. It outputs a FeatureMaps with the same number of
feature maps, each with `out_channels` channels.

### Configuration

```yaml
        decoder:
          - class_path: rslearn.models.conv.Conv
            init_args:
              # The number of input channels. If there are multiple feature
              # maps, they can have different resolutions, but must all have the
              # same number of channels.
              in_channels: 128
              # The number of output channels.
              out_channels: 64
              # The kernel size, stride, and padding. See torch.nn.Conv2d. The
              # stride defaults to 1 and the padding defaults to "same", while
              # kernel_size must be configured. "same" padding keeps the same
              # resolution as the input. If stride is not 1, then padding must
              # be set since "same" is only accepted when the stride is 1.
              kernel_size: 3
              stride: 1
              padding: "same"
              # The activation to use. It defaults to ReLU.
              activation:
                class_path: torch.nn.ReLU
          # ...
```

### Example

See the [SingleTaskModel example](../TasksAndModels.md#introduction), which applies a
Conv on the FPN output before Faster R-CNN.
