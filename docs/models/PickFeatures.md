## rslearn.models.pick_features.PickFeatures

PickFeatures picks a subset of feature maps from a FeatureMaps to pass to the next
component. It inputs a FeatureMaps and outputs the updated FeatureMaps list containing
only the selected feature maps.

### Configuration

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

### Example

See the [DetectionTask](../TasksAndModels.md#detectiontask) example, which uses
PickFeatures to drop the upsampled feature map produced by SatlasPretrain with FPN
enabled before passing the remaining feature maps to Faster R-CNN.
