## Computing Embeddings using OlmoEarth with Local Attention

This tutorial extends the [OlmoEarth embeddings tutorial](OlmoEarthEmbeddings.md) to
use the **local attention** feature. With local attention, each token only attends to
tokens within a spatial window rather than the full sequence. This reduces the quadratic
cost of self-attention and enables processing larger input crops.

We will compute embeddings on a 512x512 window over downtown Seattle and Elliott Bay.

We proceed in three steps:

1. [Create windows](#create-windows): create an rslearn dataset and add a window covering
   downtown Seattle and the waterfront.
2. [Materialize](#materialize-satellite-images): download, re-project, and crop satellite
   images.
3. [Compute embeddings](#compute-and-save-embeddings): run the OlmoEarth encoder with
   local attention and save embeddings.

## Create Windows

Create a folder `./dataset` and save this dataset configuration as `./dataset/config.json`:

```json
{
  "layers": {
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

Now create a window covering downtown Seattle and Elliott Bay. The box is chosen so the
window is approximately 512x512 pixels at 10 m/pixel (about 5 km x 5 km), capturing
both the city center and the water:

```
export DATASET_PATH=./dataset
rslearn dataset add_windows --root $DATASET_PATH --group default --name default --utm --resolution 10 --src_crs EPSG:4326 --box=-122.36,47.59,-122.295,47.635 --start 2024-01-01T00:00:00+00:00 --end 2025-01-01T00:00:00+00:00
```

## Materialize Satellite Images

Download Sentinel-2 imagery:

```
rslearn dataset prepare --root $DATASET_PATH --workers 32 --retry-max-attempts 5 --retry-backoff-seconds 5
rslearn dataset materialize --root $DATASET_PATH --workers 32 --no-use-initial-job --retry-max-attempts 5 --retry-backoff-seconds 5
```

Verify with:

```
qgis $DATASET_PATH/windows/default/default/layers/sentinel2_l2a/B01_B02_B03_B04_B05_B06_B07_B08_B8A_B09_B11_B12/geotiff.tif
```

## Compute and Save Embeddings

The key difference from the [standard embeddings tutorial](OlmoEarthEmbeddings.md) is
the `forward_kwargs` parameter on the OlmoEarth encoder. Setting
`local_attention_window` restricts each token's self-attention to a square spatial window
(in token units) instead of attending to all tokens in the crop. This reduces the cost
of attention from O(N^2) to O(N * W^2) where W is the window size.

With `patch_size: 4` and `crop_size: 128`, each crop becomes a 32x32 token grid. A
`local_attention_window` of 16 means each token attends to tokens within ±8 positions in
both row and column, covering a 64x64 pixel receptive field at input resolution.

Save this as `model.yaml`:

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
              forward_kwargs:
                local_attention_window: 16
        decoder:
          - class_path: rslearn.train.tasks.embedding.EmbeddingHead
    optimizer:
      class_path: rslearn.train.optimizer.AdamW
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    path: ${DATASET_PATH}
    inputs:
      sentinel2_l2a:
        data_type: "raster"
        layers: ["sentinel2_l2a", "sentinel2_l2a.1"]
        bands: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
        passthrough: true
        dtype: FLOAT32
        load_all_layers: true
    task:
      class_path: rslearn.train.tasks.embedding.EmbeddingTask
    batch_size: 1
    num_workers: 32
    predict_config:
      transforms:
        - class_path: rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize
          init_args:
            band_names:
              sentinel2_l2a: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
      load_all_crops: true
      crop_size: 128
      overlap_pixels: 64
trainer:
  callbacks:
    - class_path: rslearn.train.prediction_writer.RslearnWriter
      init_args:
        output_layer: embeddings
        merger:
          class_path: rslearn.train.prediction_writer.RasterMerger
          init_args:
            # crop_size=128 with patch_size=4 produces 32x32 output per crop.
            # overlap_pixels=64 at input resolution is 16 at output resolution,
            # so we remove 8 pixels from each side.
            overlap_pixels: 16
            downsample_factor: 4
```

Compared to the standard tutorial, the changes are:

- **`forward_kwargs.local_attention_window: 16`**: restricts attention to a 16x16 token
  window (64x64 pixels). This is the only model-level change needed.
- **`crop_size: 128`** (up from 64): larger crops are feasible because local attention
  keeps memory usage manageable.
- **`overlap_pixels: 64`**: overlap between adjacent crops so the merger can blend them
  smoothly.
- **`batch_size: 1`**: the larger crop size uses more memory per sample; reduce batch
  size accordingly.

Add the embeddings layer to `config.json`:

```jsonc
{
  "layers": {
    // ... existing layers ...
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

Set `num_bands` based on the model: NANO=128, TINY=192, BASE=768, LARGE=1024.

Run inference:

```
rslearn model predict --config model.yaml
```

Visualize the output:

```
qgis $DATASET_PATH/windows/default/default/layers/embeddings/*/geotiff.tif
```

### Choosing the Window Size

The `local_attention_window` is specified in **token units** (not pixels). To convert:

```
pixel receptive field = local_attention_window × patch_size
```

With `patch_size=4`:

| `local_attention_window` | Pixel receptive field | Notes |
|---|---|---|
| 8 | 32 px (320 m) | Very local, fast |
| 16 | 64 px (640 m) | Good default |
| 32 | 128 px (1280 m) | Full crop (equivalent to global attention within each 128-pixel crop) |

Smaller windows are faster and use less memory but limit the spatial context each token
can incorporate. A window of 16 is a reasonable starting point.
