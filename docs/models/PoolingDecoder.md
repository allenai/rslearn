## rslearn.models.pooling_decoder.PoolingDecoder

PoolingDecoder computes a FeatureVector from a FeatureMaps.

It inputs a FeatureMaps, but only uses the last feature map. Then it applies a
configurable number of convolutional layers before pooling, and a configurable number
of fully connected layers after pooling.

The output is a FeatureVector. Most intermediate components currently input a
FeatureMaps, so the next component is typically a predictor (either
[ClassificationHead](ClassificationHead.md) or [RegressionHead](RegressionHead.md)).

### Configuration

Here is a summary, see `rslearn.models.pooling_decoder` for all of the available
options.

```yaml
        decoder:
          - class_path: rslearn.models.pooling_decoder.PoolingDecoder
            init_args:
              # The number of channels in the input (specifically, the last
              # feature map in the list).
              in_channels: 1024
              # The number of output channels. This is typically tied to the
              # task, e.g. if there will be 8 classes then this should be 8.
              out_channels: 8
              # The number of extra convolutional layers to apply before
              # pooling. The default is 0.
              num_conv_layers: 0
              # The number of fully connected layers to apply after pooling. The
              # default is 0.
              num_fc_layers: 0
              # Number of hidden channels when using num_conv_layers /
              # num_fc_layers.
              conv_channels: 128
              fc_channels: 512
          # This is an example for using PoolingDecoder with a classification
          # task.
          - class_path: rslearn.train.tasks.classification.ClassificationHead
```

### Example

See the [ClassificationTask](../TasksAndModels.md#classificationtask) and
[RegressionTask](../TasksAndModels.md#regressiontask) examples.
