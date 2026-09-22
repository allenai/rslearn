## rslearn.train.tasks.segmentation.SegmentationHead

SegmentationHead is the predictor for
[SegmentationTask](../TasksAndModels.md#segmentationtask). It computes cross entropy
loss given the logits and targets. It does not take any arguments.

It inputs a FeatureMaps, which must contain a single feature map of logits, with the
channel dimension size matching the number of classes. It outputs the class
probabilities after applying softmax on those input logits. It also produces a loss
dict with one key, "cls", containing the softmax cross entropy loss.

### Configuration

```yaml
        decoder:
          # ...
          - class_path: rslearn.train.tasks.segmentation.SegmentationHead
```

### Example

See the [SegmentationTask](../TasksAndModels.md#segmentationtask) example, which pairs
SegmentationHead with a UNetDecoder. The [TokensToChannels](TokensToChannels.md) and
[BreakpointScan](BreakpointScan.md) pages have examples of using SegmentationHead with
per-timestep tokens from OlmoEarth.
