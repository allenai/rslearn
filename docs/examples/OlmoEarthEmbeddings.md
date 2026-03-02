## Computing Embeddings using OlmoEarth

This tutorial shows how to compute OlmoEarth embeddings on a target location and time
of interest. We will use rslearn to materialize satellite images that we will then pass
to the OlmoEarth encoder. For an introduction to rslearn, see
[the main README](../../README.md) and [CoreConcepts](../CoreConcepts.md).

We proceed in three steps:

1. [Create windows](#create-windows): create an rslearn dataset and add windows that define the spatiotemporal
   boxes for which we want to compute embeddings.
2. [Materialize](#materialize-satellite-images): download, re-project, and crop satellite images to align with the windows.
3. [Compute embeddings](#compute-and-save-embeddings): initialize the OlmoEarth pre-trained model and compute and save embeddings.

## Create Windows

First, create a new folder to contain the rslearn dataset (e.g. `./dataset`), and copy
this dataset configuration file to `./dataset/config.json`. It obtains Sentinel-2,
Sentinel-1, and Landsat satellite images identical in format to those used for
pre-training:

```
{
  "layers": {
    "landsat": {
      "band_sets": [{
          "bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"],
          "dtype": "uint16"
      }],
      "data_source": {
        "class_path": "rslearn.data_sources.aws_landsat.LandsatOliTirs",
        "init_args": {
          "metadata_cache_dir": "cache/landsat",
          "sort_by": "cloud_cover"
        },
        "ingest": false,
        "query_config": {
          "max_matches": 12,
          "period_duration": "30d",
          "space_mode": "PER_PERIOD_MOSAIC"
        }
      },
      "type": "raster"
    },
    "sentinel1": {
      "band_sets": [{
          "bands": ["vv", "vh"],
          "dtype": "float32",
          "nodata_vals": [-32768, -32768]
      }],
      "data_source": {
        "class_path": "rslearn.data_sources.planetary_computer.Sentinel1",
        "init_args": {
          "cache_dir": "cache/planetary_computer",
          "query": {
            "sar:instrument_mode": {"eq": "IW"},
            "sar:polarizations": {"eq": ["VV", "VH"]}
          }
        },
        "ingest": false,
        "query_config": {
          "max_matches": 12,
          "period_duration": "30d",
          "space_mode": "PER_PERIOD_MOSAIC"
        }
      },
      "type": "raster"
    },
    "sentinel2_l2a": {
      "band_sets": [{
          "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
          "dtype": "uint16"
      }],
      "data_source": {
        "class_path": "rslearn.data_sources.planetary_computer.Sentinel2",
        "init_args": {
          "cache_dir": "cache/planetary_computer",
          "harmonize": true,
          "sort_by": "eo:cloud_cover"
        },
        "ingest": false,
        "query_config": {
          "max_matches": 12,
          "period_duration": "30d",
          "space_mode": "PER_PERIOD_MOSAIC"
        }
      },
      "type": "raster"
    }
  }
}
```

Now, create a window corresponding to the spatiotemporal box of interest. We use a
10 m/pixel resolution and UTM projection since that matches what was used for
pre-training.

```
export DATASET_PATH=./dataset
rslearn dataset add_windows --root $DATASET_PATH --group default --name default --utm --resolution 10 --src_crs EPSG:4326 --box=-122.4,47.6,-122.3,47.7 --start 2024-01-01T00:00:00+00:00 --end 2025-01-01T00:00:00+00:00
```

Above, the `--box` argument is in the form `lon1,lat1,lon2,lat2`.

The duration of the time range can be adjusted depending on the application -- where
possible, we recommend using a one-year time range, since that is the maximum time
range used during pre-training. For features that change more quickly, it may make
sense to use a shorter time range. If you want to compute embeddings on a specific
satellite image, you can narrow the time range to the minute around the timestamp of
that image. The `dataset_config.json` specifies to create one image mosaic per 30-day
period within the time range, which is recommended since it matches pre-training, but
you could try obtaining images more frequently if desired.

If the box exceeds 10 km x 10 km, we recommend passing `--grid_size` to create multiple
windows that are each limited to 1024x1024:

```
rslearn dataset add_windows --root $DATASET_PATH --group default --name default --utm --resolution 10 --src_crs EPSG:4326 --box=-122.6,47.4,-122.1,47.9 --start 2024-06-01T00:00:00+00:00 --end 2024-08-01T00:00:00+00:00 --grid_size 1024
```

## Materialize Satellite Images

Now, we can use rslearn to materialize the satellite images for the window(s):

```
rslearn dataset prepare --root $DATASET_PATH --workers 32 --disabled-layers landsat --retry-max-attempts 5 --retry-backoff-seconds 5
rslearn dataset materialize --root $DATASET_PATH --workers 32 --no-use-initial-job --disabled-layers landsat --retry-max-attempts 5 --retry-backoff-seconds 5
```

Here, we only obtain Sentinel-2 and Sentinel-1 images. To also obtain Landsat images,
you will need to setup AWS credentials (set the `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` environment variables) for access to the
[`usgs-landsat` requester pays bucket](https://registry.opendata.aws/usgs-landsat/),
however for most tasks we find that OlmoEarth produces high-quality embeddings from
Sentinel-2 and Sentinel-1 alone.

If you used a single window, then the first Sentinel-2 L2A GeoTIFF should appear here:

```
qgis $DATASET_PATH/windows/default/default/layers/sentinel2_l2a/B01_B02_B03_B04_B05_B06_B07_B08_B8A_B09_B11_B12/geotiff.tif
```

With multiple timesteps, you should see folders like `layers/sentinel2_l2a.1`, `layers/sentinel2_l2a.2`, and so on.

## Compute and Save Embeddings

Now we can create a model configuration file that will compute and save the embeddings
for each window. Save this model config as `model.yaml`:

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.singletask.SingleTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.olmoearth_pretrain.model.OlmoEarth
            init_args:
              model_id: OLMOEARTH_V1_BASE
              patch_size: 4
        decoder:
          # The EmbeddingHead is a wrapper that works with EmbeddingTask below to save
          # the embeddings computed by the encoder.
          - class_path: rslearn.train.tasks.embedding.EmbeddingHead
    # The optimizer here is not used but needs to be passed.
    optimizer:
      class_path: rslearn.train.optimizer.AdamW
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    path: ${DATASET_PATH}
    inputs:
      # Read the Sentinel-2 and Sentinel-1 images materialized above.
      # You may need to adjust the number of layers below to match your time range.
      sentinel2_l2a:
        data_type: "raster"
        layers: ["sentinel2_l2a", "sentinel2_l2a.1", "sentinel2_l2a.2", "sentinel2_l2a.3"]
        # This is the band order expected by OlmoEarth.
        bands: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
        passthrough: true
        dtype: FLOAT32
        load_all_layers: true
      sentinel1:
        data_type: "raster"
        layers: ["sentinel1", "sentinel1.1", "sentinel1.2", "sentinel1.3"]
        # This is the band order expected by OlmoEarth.
        bands: ["vv", "vh"]
        passthrough: true
        dtype: FLOAT32
        load_all_layers: true
    task:
      # The EmbeddingTask is a dummy task setup so that the output feature map can be
      # written to the rslearn dataset during `model predict`.
      class_path: rslearn.train.tasks.embedding.EmbeddingTask
    batch_size: 8
    num_workers: 32
    predict_config:
      transforms:
        - class_path: rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize
          init_args:
            band_names:
              sentinel2_l2a: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
              sentinel1: ["vv", "vh"]
      # We apply sliding-window inference (using 64x64 input crops) with overlap.
      load_all_crops: true
      # This is the crop size for inference.
      crop_size: 64
      overlap_pixels: 32
trainer:
  callbacks:
   # The RslearnWriter will write our embeddings to a layer in the rslearn dataset.
    - class_path: rslearn.train.prediction_writer.RslearnWriter
      init_args:
        # This references the "embeddings" layer that we will add to our dataset config
        # file to store the embeddings.
        output_layer: embeddings
        merger:
          # The RasterMerge will merge the outputs across the different sliding-window
          # crops that pertain to the same rslearn windows.
          class_path: rslearn.train.prediction_writer.RasterMerger
          init_args:
            # This removes the border from the overlap_pixels. With model patch size 4
            # and input crop size 64, the model produces a 16x16 output. We have
            # overlap_pixels=32 at input resolution, which is 8 pixels at output
            # resolution, so we remove 4 pixels from each side.
            overlap_pixels: 8
            # This lets the merger know what output resolution to expect relative to
            # the window's resolution. Here, our output will be 1/patch_size relative
            # to the window resolution (input resolution), since we compute one
            # embedding per patch in the input, so we set the downsample_factor to
            # patch_size.
            downsample_factor: 4
```

The model config uses `EmbeddingTask` and `RslearnWriter` to write the embeddings to a
layer called "embeddings" in the rslearn dataset. We need to add this layer to our
dataset configuration file:

```jsonc
{
  "layers": {
    // ...
    "embeddings": {
      "band_sets": [{
          "dtype": "float32",
          "num_bands": 768
      }],
      "type": "raster"
    }
  }
}
```

Finally, we can run the `model predict` command:

```
rslearn model predict --config model.yaml
```

You can visualize the output embeddings in qgis:

```
qgis $DATASET_PATH/windows/default/default/layers/embeddings/*/geotiff.tif
```
