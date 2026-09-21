## rslearn.train.tasks.classification.ClassificationHead

ClassificationHead is the predictor for
[ClassificationTask](../TasksAndModels.md#classificationtask). It computes cross entropy
loss given the logits and targets. It does not take any arguments.

It inputs a FeatureVector of logits, where the channel dimension size must match the
number of classes. It outputs the class probabilities after applying softmax on those
input logits. It also produces a loss dict with one key, "cls", containing the softmax
cross entropy loss.

### Configuration

```yaml
        decoder:
          # ...
          - class_path: rslearn.train.tasks.classification.ClassificationHead
```

### Example

See the [ClassificationTask](../TasksAndModels.md#classificationtask) example, which
pairs ClassificationHead with a [PoolingDecoder](PoolingDecoder.md).
