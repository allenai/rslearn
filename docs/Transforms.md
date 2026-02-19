## Transforms

Transforms in rslearn can be used to implement augmentations, to normalize data, to
perform specialized data pre-processing, etc.

Transforms are specified via the `SplitConfig` object. The `RslearnDataModule` accepts
a default `SplitConfig` that is shared across all model stages, as well as a train,
val, test, and predict `SplitConfig` that overrides those default options. See
[ModelConfig.md](ModelConfig.md) for more details. Note that there is currently no way
to append to the default transforms in a stage-specific transform list, so e.g. if you
have data pre-processing steps in the default `SplitConfig` but want to add
augmentations for training, you would need to repeat the pre-processing steps when
overriding the transforms in the train `SplitConfig`.

Transforms are applied after initial data loading and task processing. First, the
`DataInputs` in the model configuration file are read from the rslearn dataset. The
`passthrough` inputs are copied to an "input dict". Labels are passed to the Task
object(s), which process them into tensors suitable for training that are placed in the
"target dict".

The transforms are then applied on the per-example input and target dicts, and can make
arbitrary adjustments to them.

Many transforms accept selectors that specify which key(s) in the input and/or target
dicts the transform should be applied on. Slashes are used to separate path components
in the selector, where "input/image" accesses `input_dict["image"]` while
"target/task1/boxes" accesses `target_dict["task1"]["boxes"]`. If the selector does not
start with either "input/" or "target/", it is assumed to access the input dict, so
"image" also accesses `input_dict["image"]`.

## Concatenate

The `Concatenate` transform concatenates bands across multiple image inputs. It can
also be used to copy an image under one name in the input dict to another name.

Here is a summary of the configuration options:

```yaml
      transforms:
        - class_path: rslearn.train.transforms.concatenate.Concatenate
          init_args:
            # This is a map from selectors to a list of band indices that should be
            # taken from that image. The images across the selectors are concatenated
            # together. An empty list of band indices can be used to indicate that all
            # bands from that image should be retained.
            selections:
              # Take first three bands from Sentinel-2.
              sentinel2: [0, 1, 2]
              # And all bands from Sentinel-1.
              sentinel1: []
            # The selector to write the output. Here it will be saved at
            # input_dict["image"].
            output_selector: image
```

Some decoders like `FasterRCNN` use `input_dict["image"]` to get the original input
dimensions. However, some encoders expect the input image to be named something else,
e.g. OlmoEarth expects the Sentinel-2 time series input to be called `sentinel2_l2a`.
Then, `Concatenate` can be used to copy it:

```yaml
      transforms:
        # Normalize the Sentinel-2 images for the OlmoEarth model.
        - class_path: rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize
          init_args:
            band_names:
              sentinel2_l2a: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
        # Apply Concatenate to copy sentinel2_l2a -> image for FasterRCNN.
        - class_path: rslearn.train.transforms.concatenate.Concatenate
          init_args:
            selections:
              sentinel2_l2a: []
            output_selector: image
```

## Flip

The `Flip` transform randomly flips inputs horizontally and/or vertically. Typically it
only used for training.

Here is a summary of the configuration options:

```yaml
      transforms:
        # ...
        - class_path: rslearn.train.transforms.flip.Flip
          init_args:
            # Selectors for images that should be flipped. It defaults to ["image"].
            image_selectors: ["image"]
            # Selectors for DetectionTask targets that should be flipped. Note that the
            # DetectionTask targets are a dict with keys for boxes, labels, etc., but
            # the selectors here should correspond to the overall dict. There example
            # here would work for single-task training; if using MultiTask, then if the
            # detection task name is "detect", you can set box_selectors to
            # ["target/detect"] (not ["target/detect/boxes"]).
            # The default is an empty list.
            box_selectors: ["target/"]
```

Here is an example to flip an input image along with segmentation targets:

```yaml
      transforms:
        # ...
        - class_path: rslearn.train.transforms.flip.Flip
          init_args:
            # Similar to DetectionTask, the SegmentationTask targets are a dict with
            # the "classes" and "valid" keys. Unlike with box_selectors, we need to
            # specify those images directly here. With MultiTask, there would be
            # another level in the dict, e.g. "target/segment_task_name/classes".
            image_selectors:
                - image
                - target/classes
                - target/valid
```

## Mask

The `Mask` transform uses one (mask) image input to mask another (target) image input.
The value of the target image is set to the mask value everywhere where the mask image
is 0.

Here is a summary of the configuration options:

```yaml
      transforms:
        - class_path: rslearn.train.transforms.mask.Mask
          init_args:
            # Selectors specifying which images to mask.
            selectors: ["image"]
            # The selector for the image to use as the mask.
            mask_selector: "mask"
            # The mask value. Each image in selectors will be set to this value where
            # the mask_selector image is 0.
            mask_value: 0
```

The original use case for Mask is for cases where only a portion of the window was
labeled for bounding box detection tasks. For SegmentationTask, the invalid portions of
the label can be set to the NODATA value, but this is not possible for DetectionTask.
The example above could be coupled with a `Task` and `DataInputs` like this:

```yaml
    task:
      class_path: rslearn.train.tasks.detection.DetectionTask
      init_args:
        property_name: "category"
        classes: ["unknown", "platform", "turbine"]
        box_size: 15
    inputs:
      image:
        data_type: "raster"
        layers: ["sentinel2"]
        bands: ["B04", "B03", "B02", "B05", "B06", "B07", "B08", "B11", "B12"]
        passthrough: true
        dtype: FLOAT32
      mask:
        data_type: "raster"
        layers: ["mask"]
        bands: ["mask"]
        passthrough: true
        dtype: FLOAT32
        is_target: true
      targets:
        data_type: "vector"
        layers: ["label"]
        is_target: true
```

Then, the dataset should contain a `mask` layer with a raster that is 0 at pixels that
were outside the context used for labeling, and 1 otherwise, and a `sentinel2` layer
containing the input image. The image will be masked to 0 at regions not considered
during labeling, so that the model can learn to predict no boxes in those regions.

## Sentinel2SCLToMask

The `Sentinel2SCLToMask` transform converts a Sentinel-2 Scene Classification Layer
(SCL) raster into a binary mask image (1 = valid, 0 = masked). This can be used with
`Mask` to apply SCL-based masking at training time.

```yaml
      transforms:
        - class_path: rslearn.train.transforms.sentinel2.Sentinel2SCLToMask
          init_args:
            scl_selector: "scl"
            output_selector: "mask"
            # Default: [3, 8, 9, 10] (cloud shadow, medium/high cloud, cirrus)
            exclude_scl_values: [3, 8, 9, 10]
        - class_path: rslearn.train.transforms.mask.Mask
          init_args:
            selectors: ["image"]
            mask_selector: "mask"
            mask_value: 0
```

## Normalize

The `Normalize` transform implements linear normalization of images.

Here is a summary of the configuration options:

```yaml
      transforms:
        - class_path: rslearn.train.transforms.normalize.Normalize
          init_args:
            # The mean and standard deviation for z-score normalization.
            # The image will be normalized as (image - mean) / std.
            # Both mean and std can either be one value (to apply on all bands), or a
            # list with one value per band.
            mean: 10
            std: 200
            # Optionally clip the result to this range after the linear rescaling. The
            # default is null to not perform any clipping.
            valid_range: [0, 1]
            # The image selectors to apply the normalization on. The default is
            # ["image"].
            selectors: ["image"]
            # Optionally limit the normalization to specific bands. This list specifies
            # the bands to normalize. If mean/std are a list, they should have the same
            # size as this bands list.
            bands: [0, 1, 2]
            # If normalizing specific bands in image time series, num_bands should be
            # set to the number of bands at each timestep. If num_bands is set, then
            # the bands is repeated for each timestep, e.g. if bands=[2] then we apply
            # normalization on image[2], image[2+num_bands], etc. num_bands can also be
            # set without setting bands if mean and std are lists, in which case those
            # means and stds are repeated for each timestep.
            num_bands: 9
```

## Pad

The `Pad` transform performs deterministic padding or cropping.

Here is a summary of the configuration options:

```yaml
      transforms:
        - class_path: rslearn.train.transforms.pad.Pad
          init_args:
            # The size to pad to, or a min/max of pad sizes. If the image is larger
            # than this size, it is cropped instead.
            size: 32
            # "topleft" (default) to only apply padding on the bottom and right sides,
            # or "center" to apply padding equally on all sides.
            mode: "topleft"
            # The image and box selectors to apply padding on.
            image_selectors: ["image"]
            box_selectors: []
```

- With topleft mode and an image smaller than `size`, padding is added to the bottom
  and right so that the original image content is in the topleft.
- With topleft mode and an image larger than `size`, the image is cropped to the
  topleft `size x size` portion.
- With center mode and an image smaller than `size`, padding is added equally on all
  sides so that the original image content is in the center.
- With center mode and an image larger than `size`, center cropping is applied to the
  center `size x size` portion.
- If `size` is a list like `[128, 256]`, then a random size between 128 and 256 is
  sampled for each example before applying the operation. This may not be useful
  currently since we do not support batch transforms yet.

## SelectBands

The `SelectBands` transform selects a subset of bands from an image. `Concatenate` can
be used for this purpose too (with one selector), but `SelectBands` supports operating
over image time series whereas `Concatenate` does not.

Here is a summary of the configuration options:

```yaml
      transforms:
        - class_path: rslearn.train.transforms.select_bands.SelectBands
          init_args:
            # The band indices to select.
            band_indices: [0, 1, 2]
            # The selector to read the input image, default "image".
            input_selector: "image"
            # The output selector under which to save the image, default "image".
            output_selector: "image"
            # Optional number of bands per image, to distinguish between stacked images
            # in an image time series. If set, then the band_indices are selected for
            # each image in the time series.
            num_bands_per_timestep: 9
```

## Sentinel1ToDecibels

The `Sentinel1ToDecibels` transform converts Sentinel-1 data from raw intensities
(linear scale) to or from decibels.

Here is a summary of the configuration options:

```yaml
      transforms:
        - class_path: rslearn.train.transforms.sentinel1.Sentinel1ToDecibels
          init_args:
            # The image selectors to apply the conversion on.
            selectors: ["image"]
            # Set true to convert from decibels to linear scale.
            from_decibels: false
            # Values less than epsilon are clipped to epsilon when converting to
            # decibels.
            epsilon: 1e-6
```
