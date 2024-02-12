Here we explore a standard task in rslearn, training a model to detect solar
farm polygons in four Sentinel-2 images from trailing calendar months.


Dataset
-------

First, create the dataset:

    rslearn dataset create --root data/solar_farms/

This just creates a folder with a placeholder dataset configuration file. We
modify the configuration file:

```json
{
  "layers": {
    "sentinel2_0": {
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
        "max_matches": 1,
        "space_mode": "mosaic",
        "time_mode": "within"
      },
    },
    "sentinel2_1": {
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
        "max_matches": 1,
        "space_mode": "mosaic",
        "time_mode": "within"
      },
      "retrieve_config": {
        "time_offset": "-30d"
      }
    },
    "sentinel2_2": {
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
        "max_matches": 1,
        "space_mode": "mosaic",
        "time_mode": "within"
      },
      "retrieve_config": {
        "time_offset": "-60d"
      }
    },
    "sentinel2_3": {
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
        "max_matches": 1,
        "space_mode": "mosaic",
        "time_mode": "within"
      },
      "retrieve_config": {
        "time_offset": "-90d"
      }
    },
    "google": {
      "band_sets": [{
        "bands": ["R", "G", "B"],
        "dtype": "uint8",
        "format": "png"
      }],
      "data_source": {
        "name": "rslearn.data_sources.webmercator_tiles.Tiles",
        "url": "https://tile.googleapis.com/v1/2dtiles/{z}/{x}/{y}?key=YOUR_API_KEY"
      },
      "query_config": {
        "max_matches": 1,
        "space_mode": "mosaic",
        "time_mode": "within"
      },
      "retrieve_config": {
        "ingest": false
      }
    },
    "polygons": {
      "type": "polygon",
      "categories": [
        "Solar Farm"
      ]
    }
  },
  "annotation": {
    "siv": {
      "reference_layers": [{
        "name": "Sentinel-2 (current)",
        "layer": "sentinel2_0",
        "bands": ["R", "G", "B"]
      }, {
        "name": "Sentinel-2 (1 month old)",
        "layer": "sentinel2_1",
        "bands": ["R", "G", "B"]
      }, {
        "name": "Sentinel-2 (2 months old)",
        "layer": "sentinel2_2",
        "bands": ["R", "G", "B"]
      }, {
        "name": "Sentinel-2 (3 months old)",
        "layer": "sentinel2_3",
        "bands": ["R", "G", "B"]
      }, {
        "name": "Google Satellite",
        "layer": "google",
        "bands": ["R", "G", "B"]
      }],
      "annotation_layer": "polygons"
    }
  }
}
```

- We use four layers since we want each image to come from a different month
  but there is no way to do that with the `query_config` in one layer. Later we
  will need to create windows that are 30 days long and set at the end of each
  four-month time range we want to label.

Next, suppose we have separately prepared a GeoJSON file of candidate solar
farm polygons, e.g. from OpenStreetMap. We can use this data to populate
windows in the dataset:

  rslearn dataset add_windows --root data/solar_farms/
  --fname solar_farms.geojson --crs epsg:3857 --zoom 13
  --time 2021-01-01,2022-01-01 --mode grid --size 512 --group osm

- Since we use epsg:3857 (WebMercator), we can specify `--zoom` instead of
  `--resolution` which lets the system compute the resolution based on the zoom
  level.
- `--mode grid` tells rslearn to create a window at any tile on a 512x512 grid
  under the specified projection that intersects with any feature in the
  GeoJSON file.

Now we prepare and materialize the dataset:

  rslearn dataset prepare --root data/solar_farms/
  rslearn dataset materialize --root data/solar_farms/

And launch annotation tool:

  rslearn annotate siv --ds_root data/solar_farms/


Model
-----

First, create the model:

  rslearn model create --root models/solar_farms/

Modify the configuration file:

```json
{
  "inputs": [{
    "layer": "sentinel2_0",
    "bands": ["R", "G", "B"],
    "transforms": [{
      "name": "normalize",
      "mean": 128,
      "stddev": 64,
    }]
  }, {
    "layer": "sentinel2_0",
    "bands": ["B01", "B05", "B06", "B07", "B08", "B11", "B12", "B8A"],
    "transforms": [{
      "name": "normalize",
      "mean": 1000,
      "stddev": 500
    }]
  }, {
    "layer": "sentinel2_1",
    "bands": ["R", "G", "B"],
    "transforms": [{
      "name": "normalize",
      "mean": 128,
      "stddev": 64,
    }]
  }, {
    "layer": "sentinel2_1",
    "bands": ["B01", "B05", "B06", "B07", "B08", "B11", "B12", "B8A"],
    "transforms": [{
      "name": "normalize",
      "mean": 1000,
      "stddev": 500
    }]
  }, {
    "layer": "sentinel2_2",
    "bands": ["R", "G", "B"],
    "transforms": [{
      "name": "normalize",
      "mean": 128,
      "stddev": 64,
    }]
  }, {
    "layer": "sentinel2_2",
    "bands": ["B01", "B05", "B06", "B07", "B08", "B11", "B12", "B8A"],
    "transforms": [{
      "name": "normalize",
      "mean": 1000,
      "stddev": 500
    }]
  }, {
    "layer": "sentinel2_3",
    "bands": ["R", "G", "B"],
    "transforms": [{
      "name": "normalize",
      "mean": 128,
      "stddev": 64,
    }]
  }, {
    "layer": "sentinel2_3",
    "bands": ["B01", "B05", "B06", "B07", "B08", "B11", "B12", "B8A"],
    "transforms": [{
      "name": "normalize",
      "mean": 1000,
      "stddev": 500
    }]
  }],
  "targets": [{
    
  }],
  "batch_size": 16,
  "train_transforms": [{
    "name": "CropFlip",
    "horizontal_flip": true,
    "vertical_flip": true,
    "crop_min": 384,
    "crop_max": 512
  }]
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