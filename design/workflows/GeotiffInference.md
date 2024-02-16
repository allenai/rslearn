Here we explore applying rslearn models on GeoTIFF images that have been obtained
separately from rslearn.

We assume the model is solar farm model that inputs all the bands from three Sentinel-2
images.


Model Configuration
-------------------

First we detail the model configuration since it helps to determine what specification
is available to guide the inference procedure.

There is the dataset that the model was trained on:

```json
{
  "layers": {
    "sentinel2": {
      "band_sets": [{
        "bands": ["R", "G", "B"],
        "dtype": "uint8",
        "format": "png"
      }, {
        "bands": ["B01", "B05", "B06", "B07", "B08", "B11", "B12", "B8A"],
        "dtype": "uint16",
        "format": "geotiff"
      }],
      "data_source": {
        "name": "rslearn.data_sources.aws_open_data.Sentinel2",
        "modality": "L1C",
        "metadata_cache_dir": "/mnt/data/cache/sentinel2_metadata/",
        "max_time_delta": "0d"
      },
      "query_config": {
        "max_matches": 4,
        "space_mode": "mosaic",
        "time_mode": "within"
      },
    },
    "polygons": {
      "type": "polygon",
      "categories": [
        "Solar Farm"
      ]
    }
  }
}
```

Then there is the model itself:

```json
{
  "datasets": [{
    "root": "/path/to/dataset",
    "inputs": [{
      "layer": "sentinel2",
      "bands": ["R", "G", "B", "B01", "B05", "B06", "B07", "B08", "B11", "B12", "B8A"],
      "transforms": [{
        "name": "normalize",
        "mean": 128,
        "stddev": 64,
      }],
      "min_images": 4,
      "max_images": 4
    }],
    "targets": [{
      "layer": "polygons",
      "type": "pixel_classification"
    }],
  }],
  "batch_size": 16,
  "train_transforms": [{
    "name": "CropFlip",
    "horizontal_flip": true,
    "vertical_flip": true,
    "crop_min": 384,
    "crop_max": 512
  }],
  "train_batch_transforms": [{
      "name": "Resize",
      "resize_min": 384,
      "resize_max": 512,
      "multiple_of": 32
  }],
  "val_max_tiles": 4096,
  "restore": {
    "type": "huggingface",
    "path": "..."
  },
  "freeze": {
    "prefixes": ["backbone.", "intermediates."],
    "unfreeze": 65536
  },
  "warmup": {
    "examples": 65536,
    "delay": 65536
  },
  "epochs": 100,
  "model": {
    "name": "SatlasNet",
    "backbone": {
        "name": "aggregation",
        "image_channels": 9,
        "aggregation_op": "max",
        "groups": [[0, 1, 2, 3]],
        "backbone": {
            "name": "swin",
            "arch": "swin_v2_b"
        }
    },
    "intermediates": [{
        "name": "fpn"
    }, {
        "name": "upsample"
    }],
    "heads": [{
        "name": "simple"
    }]
  },
  "optimizer": {
      "name": "adam",
      "initial_lr": 0.0001
  },
  "scheduler": {
      "name": "plateau",
      "factor": 0.5,
      "patience": 1,
      "cooldown": 10,
      "min_lr": 1e-6
  },
  "summary_examples": 32768
}
```


Command-Line Usage
------------------

From command-line you could import each GeoTIFF image into a new dataset, and then
apply the model on the dataset.

Use one arbitrary GeoTIFF to initialize windows on a grid (CRS is not specified, so it
comes from the specified file):

```
rslearn dataset add_windows --root /path/to/dataset --fname blah.tif --resolution 10
  --mode grid --size 2048 --group infer
```

Import each set of rasters:

```
rslearn dataset import --root /path/to/dataset
  --rasters '[{"fname":"im1_rgb.tif", "bands": ["R", "G", "B"]}, ...]'
```

Then apply the model:

```
rslearn model apply --model_root /path/to/model --ds_root /path/to/dataset
  --group infer
```


Programmatic Usage
------------------

Should be able to initialize the model, and then run on crops of the image manually.
The model and transforms all should be able to be loaded and used independently.
