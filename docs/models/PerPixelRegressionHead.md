## rslearn.train.tasks.per_pixel_regression.PerPixelRegressionHead

PerPixelRegressionHead is the predictor for
[PerPixelRegressionTask](../TasksAndModels.md#perpixelregressiontask). It computes a
per-pixel regression loss (MSE, L1, or Huber).

It inputs a FeatureMaps, which must contain a single feature map consisting of the
predicted values at each pixel. If `use_sigmoid` is false, those should correspond to
the scaled values (actual value multiplied by the scale factor configured in the task).

It outputs the scaled values as a BHW tensor. It also produces a loss dict with one
key, "regress", containing the configured regression loss.

### Configuration

```yaml
        decoder:
          # ...
          - class_path: rslearn.train.tasks.per_pixel_regression.PerPixelRegressionHead
            init_args:
              # The loss function to use: "mse" (default), "l1", or "huber".
              loss_mode: "mse"
              # Optional: delta for Huber loss (only used when
              # loss_mode="huber").
              huber_delta: 1.0
              # Whether to apply a sigmoid activation on the output. This
              # requires the targets to be between 0-1. Otherwise, the previous
              # output is unmodified.
              use_sigmoid: false
```

### Example

See the [PerPixelRegressionTask](../TasksAndModels.md#perpixelregressiontask) example,
which pairs PerPixelRegressionHead with a UNetDecoder.
