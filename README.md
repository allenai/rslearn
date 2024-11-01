Overview
--------

rslearn is a library and tool for developing remote sensing datasets and models.

rslearn helps with:

1. Developing remote sensing datasets, starting with defining spatiotemporal windows
   (roughly equivalent to training examples) that should be annotated.
2. Importing raster and vector data from various online or local data sources into the
   dataset.
3. Annotating new categories of vector data (like points, polygons, and classification
   labels) using integrated web-based labeling apps.
4. Fine-tuning remote sensing foundation models on these datasets.
5. Applying models on new locations and times.


Quick links:
- [CoreConcepts](CoreConcepts.md) summarizes key concepts in rslearn, including
  datasets, windows, layers, and data sources.
- [Examples](Examples.md) contains more examples, including customizing different
  stages of rslearn with additional code.


Setup
-----

rslearn requires Python 3.10+ (Python 3.12 is recommended).

    git clone https://github.com/allenai/rslearn.git
    cd rslearn
    pip install .[extra]


Example Usage
-------------

This is an example of building a remote sensing dataset, and then training a model
on that dataset, using rslearn. Specifically, we will train a model that inputs
Sentinel-2 images and predicts land cover through a semantic segmentation task.

Let's start by defining a region of interest and obtaining Sentinel-2 images. Create a
directory `/path/to/dataset` and corresponding configuration file at
`/path/to/dataset/config.json` as follows:

    {
        "layers": {
            "sentinel2": {
                "type": "raster",
                "band_sets": [{
                    "dtype": "uint8",
                    "bands": ["R", "G", "B"]
                }],
                "data_source": {
                    "name": "rslearn.data_sources.gcp_public_data.Sentinel2",
                    "index_cache_dir": "cache/sentinel2/",
                    "max_time_delta": "1d",
                    "sort_by": "cloud_cover",
                    "use_rtree_index": false
                }
            }
        },
        "tile_store": {
            "name": "file",
            "root_dir": "tiles"
        }
    }

Here, we have initialized an empty dataset and defined a raster layer called
`sentinel2`. Because it specifies a data source, it will be populated automatically. In
particular, the data will be sourced from a
[public Google Cloud Storage bucket containing Sentinel-2 imagery](https://cloud.google.com/storage/docs/public-datasets/sentinel-2).
The `sort_by` option sorts scenes in ascending order by cloud cover, so we will end up
choosing the scenes with minimal cloud cover.

Next, let's create our spatiotemporal windows. These will correspond to training
examples.

    export DATASET_PATH=/path/to/dataset
    rslearn dataset add_windows --root $DATASET_PATH --group default --utm --resolution 10 --grid_size 128 --src_crs EPSG:4326 --box=-122.6901,47.2079,-121.4955,47.9403 --start 2024-06-01T00:00:00+00:00 --end 2024-08-01T00:00:00+00:00 --name seattle

This creates windows along a 128x128 grid in the specified projection (i.e.,
appropriate UTM zone for the location with 10 m/pixel resolution) covering the
specified bounding box, which is centered at Seattle.

We can now obtain the Sentinel-2 images by running prepare, ingest, and materialize.

* Prepare: lookup items (in this case, Sentinel-2 scenes) in the data source that match with the spatiotemporal windows we created.
* Ingest: retrieve those items. This step populates the `tiles` directory within the dataset.
* Materialize: crop/mosaic the items to align with the windows. This populates the `layers` folder in each window directory.

    rslearn dataset prepare --root $DATASET_PATH --workers 32 --batch-size 8
    rslearn dataset ingest --root $DATASET_PATH --workers 32 --no-use-initial-job --jobs-per-process 1
    rslearn dataset materialize --root $DATASET_PATH --workers 32 --no-use-initial-job

For ingestion, you may need to reduce the number of workers depending on the available
memory on your system.

You should now be able to open the GeoTIFF images. Let's find the window that
corresponds to downtown Seattle:

    import shapely
    from rslearn.const import WGS84_PROJECTION
    from rslearn.dataset import Dataset
    from rslearn.utils import Projection, STGeometry
    from upath import UPath

    # Define longitude and latitude for downtown Seattle.
    downtown_seattle = shapely.Point(-122.333, 47.606)

    # Iterate over the windows and find the closest one.
    dataset = Dataset(path=UPath("/path/to/dataset"))
    best_window_name = None
    best_distance = None
    for window in dataset.load_windows(workers=32):
        shp = window.get_geometry().to_projection(WGS84_PROJECTION).shp
        distance = shp.distance(downtown_seattle)
        if best_distance is None or distance < best_distance:
            best_window_name = window.name
            best_distance = distance

    print(best_window_name)

It should be `seattle_54912_-527360`, so let's open it in qgis (or your favorite GIS
software):

    qgis $DATASET_PATH/windows/default/seattle_54912_-527360/layers/sentinel2/R_G_B/geotiff.tif


### Adding Land Cover Labels

Before we can train a land cover prediction model, we need labels. Here, we will use
the ESA WorldCover land cover map as labels.

Start by downloading the WorldCover data from https://worldcover2021.esa.int

    wget https://worldcover2021.esa.int/data/archive/ESA_WorldCover_10m_2021_v200_60deg_macrotile_N30W180.zip
    mkdir world_cover_tifs
    unzip ESA_WorldCover_10m_2021_v200_60deg_macrotile_N30W180.zip -d world_cover_tifs/

It would require some work to write a script to re-project and crop these GeoTIFFs so
that they align with the windows we have previously defined (and the Sentinel-2 images
we have already ingested). We can use the LocalFiles data source to have rslearn
automate this process. Update the dataset `config.json` with a new layer:

    "layers": {
        "sentinel2": {
            ...
        },
        "worldcover": {
            "type": "raster",
            "band_sets": [{
                "dtype": "uint8",
                "bands": ["B1"]
            }],
            "resampling_method": "nearest",
            "data_source": {
                "name": "rslearn.data_sources.local_files.LocalFiles",
                "src_dir": "file:///path/to/world_cover_tifs/"
            }
        }
    },
    ...

Repeat the materialize process so we populate the data for this new layer:

    rslearn dataset prepare --root $DATASET_PATH --workers 32 --batch-size 8
    rslearn dataset ingest --root $DATASET_PATH --workers 32 --no-use-initial-job --jobs-per-process 1
    rslearn dataset materialize --root $DATASET_PATH --workers 32 --no-use-initial-job

We can visualize both the GeoTIFFs together in qgis:

    qgis $DATASET_PATH/windows/default/seattle_54912_-527360/layers/*/*/geotiff.tif


### Training a Model

Create a model configuration file `land_cover_model.yaml`:

    model:
      class_path: rslearn.train.lightning_module.RslearnLightningModule
      init_args:
        # This part defines the model architecture.
        # Essentially we apply the SatlasPretrain Sentinel-2 backbone with a UNet decoder
        # that terminates at a segmentation prediction head.
        # The backbone outputs four feature maps at different scales, and the UNet uses
        # these to compute a feature map at the input scale.
        # Finally the segmentation head applies per-pixel softmax to compute the land
        # cover class.
        model:
          class_path: rslearn.models.singletask.SingleTaskModel
          init_args:
            encoder:
              - class_path: rslearn.models.satlaspretrain.SatlasPretrain
                init_args:
                  model_identifier: "Sentinel2_SwinB_SI_RGB"
            decoder:
              - class_path: rslearn.models.unet.UNetDecoder
                init_args:
                  in_channels: [[4, 128], [8, 256], [16, 512], [32, 1024]]
                  # We use 101 classes because the WorldCover classes are 10, 20, 30, 40
                  # 50, 60, 70, 80, 90, 95, 100.
                  # We could process the GeoTIFFs to collapse them to 0-10 (the 11 actual
                  # classes) but the model will quickly learn that the intermediate
                  # values are never used.
                  out_channels: 101
                  conv_layers_per_resolution: 2
              - class_path: rslearn.train.tasks.segmentation.SegmentationHead
        # Remaining parameters in RslearnLightningModule define different aspects of the
        # training process like initial learning rate.
        lr: 0.0001
    data:
      class_path: rslearn.train.data_module.RslearnDataModule
      init_args:
        # Replace this with the dataset path.
        path: /path/to/dataset/
        # This defines the layers that should be read for each window.
        # The key ("image" / "targets") is what the data will be called in the model,
        # while the layers option specifies which layers will be read.
        inputs:
          image:
            data_type: "raster"
            layers: ["sentinel2"]
            bands: ["R", "G", "B"]
            passthrough: true
          targets:
            data_type: "raster"
            layers: ["worldcover"]
            bands: ["B1"]
            is_target: true
        task:
          # Train for semantic segmentation.
          # The remap option is only used when visualizing outputs during testing.
          class_path: rslearn.train.tasks.segmentation.SegmentationTask
          init_args:
            num_classes: 101
            remap_values: [[0, 1], [0, 255]]
        batch_size: 8
        num_workers: 32
        # These define different options for different phases/splits, like training,
        # validation, and testing.
        # Here we use the same transform across splits except training where we add a
        # flipping augmentation.
        # For now we are using the same windows for training and validation.
        default_config:
          transforms:
            - class_path: rslearn.train.transforms.normalize.Normalize
              init_args:
                mean: 0
                std: 255
        train_config:
          transforms:
            - class_path: rslearn.train.transforms.normalize.Normalize
              init_args:
                mean: 0
                std: 255
            - class_path: rslearn.train.transforms.flip.Flip
              init_args:
                image_selectors: ["image", "target/classes", "target/valid"]
          groups: ["default"]
        val_config:
          groups: ["default"]
        test_config:
          groups: ["default"]
        predict_config:
          groups: ["predict"]
          load_all_patches: true
          skip_targets: true
          patch_size: 512
    trainer:
      max_epochs: 10
      callbacks:
        - class_path: lightning.pytorch.callbacks.ModelCheckpoint
          init_args:
            save_top_k: 1
            save_last: true
            monitor: val_accuracy
            mode: max

Now we can train the model:

    rslearn model fit --config land_cover_model.yaml


### Apply the Model

Let's apply the model on Portland, OR (you can change it to Portland, ME if you like).
We start by defining a new window around Portland. This time, instead of creating
windows along a grid, we just create one big window. This is because we are just going
to run the prediction over the whole window rather than use different windows as
different training examples.

    rslearn dataset add_windows --root $DATASET_PATH --group predict --utm --resolution 10 --src_crs EPSG:4326 --box=-122.712,45.477,-122.621,45.549 --start 2024-06-01T00:00:00+00:00 --end 2024-08-01T00:00:00+00:00 --name portland
    rslearn dataset prepare --root $DATASET_PATH --workers 32 --batch-size 8
    rslearn dataset ingest --root $DATASET_PATH --workers 32 --no-use-initial-job --jobs-per-process 1
    rslearn dataset materialize --root $DATASET_PATH --workers 32 --no-use-initial-job

We also need to add an RslearnPredictionWriter to the trainer callbacks in the model
configuration file, as it will handle writing the outputs from the model to a GeoTIFF.

    trainer:
      callbacks:
        - class_path: lightning.pytorch.callbacks.ModelCheckpoint
          ...
        - class_path: rslearn.train.prediction_writer.RslearnWriter
          init_args:
            path: /path/to/dataset/
            output_layer: output

Because of our `predict_config`, when we run `model predict` it will apply the model on
windows in the "predict" group, which is where we added the Portland window.

And it will be written in a new output_layer called "output". But we have to update the
dataset configuration so it specifies the layer:


    "layers": {
        "sentinel2": {
            ...
        },
        "worldcover": {
            ...
        },
        "output": {
            "type": "raster",
            "band_sets": [{
                "dtype": "uint8",
                "bands": ["output"]
            }]
        }
    },

Now we can apply the model:

    # Find model checkpoint in lightning_logs dir.
    ls lightning_logs/*/checkpoints/last.ckpt
    rslearn model predict --config land_cover_model.yaml --ckpt_path lightning_logs/version_0/checkpoints/last.ckpt

And visualize the Sentinel-2 image and output in qgis:

    qgis $DATASET_PATH/windows/predict/portland/layers/*/*/geotiff.tif


### Defining Train and Validation Splits

We can visualize the logged metrics using Tensorboard:

    tensorboard --logdir=lightning_logs/

However, because our training and validation data are identical, the validation metrics
are not meaningful.

There are two suggested ways to split windows into different subsets:

1. Assign windows to different groups.
2. Use different key-value pairs in the windows' options dicts for different splits.

We will use the second approach. The script below sets a "split" key in the options
dict (which is stored in each window's `metadata.json` file) to "train" or "val"
based on the SHA-256 hash of the window name.

    import hashlib
    import tqdm
    from rslearn.dataset import Dataset, Window
    from upath import UPath

    ds_path = UPath("/path/to/dataset/")
    dataset = Dataset(ds_path)
    windows = dataset.load_windows(show_progress=True, workers=32)
    for window in tqdm.tqdm(windows):
        if hashlib.sha256(window.name.encode()).hexdigest()[0] in ["0", "1"]:
            split = "val"
        else:
            split = "train"
        if "split" in window.options and window.options["split"] == split:
            continue
        window.options["split"] = split
        window.save()

Now we can update the model configuration file to use these splits:

    default_config:
      transforms:
        - class_path: rslearn.train.transforms.normalize.Normalize
          init_args:
            mean: 0
            std: 255
    train_config:
      transforms:
        - class_path: rslearn.train.transforms.normalize.Normalize
          init_args:
            mean: 0
            std: 255
        - class_path: rslearn.train.transforms.flip.Flip
          init_args:
            image_selectors: ["image", "target/classes", "target/valid"]
      groups: ["default"]
      tags:
        split: train
    val_config:
      groups: ["default"]
      tags:
        split: val
    test_config:
      groups: ["default"]
      tags:
        split: val
    predict_config:
      groups: ["predict"]
      load_all_patches: true
      skip_targets: true
      patch_size: 512

The `tags` option that we are adding here tells rslearn to only load windows with a
matching key and value in the window options.

Previously when we run `model fit`, it should show the same number of windows for
training and validation:

    got 4752 examples in split train
    got 4752 examples in split val

With the updates, it should show different numbers like this:

    got 4167 examples in split train
    got 585 examples in split val


### Visualizing with `model test`

Coming soon


### Inputting Multiple Sentinel-2 Images

Coming soon


### Logging to Weights & Biases

Coming soon


Contact
-------

For questions and suggestions, please open an issue on GitHub.
