# Tessera

rslearn includes a wrapper for Tessera v1.1 QAT encoder checkpoints from
https://github.com/ucam-eo/tessera.

The wrapper does not download weights automatically. We recommend using the
Microsoft Planetary Computer checkpoint because rslearn does not currently
support OPERA RTC-S1 here. Download the encoder-only
[`tessera_v1_1_mpc_encoder.pt`](https://drive.google.com/file/d/1t-gfTxi3Hg_uJXpJ9etROCRgKt2myfJ2/view?usp=drive_link)
checkpoint from the official Tessera README, then pass it with
`checkpoint_path`.

Tessera inputs should be normalized with `rslearn.models.tessera.TesseraNormalize`
before they reach the model. Set the transform's `data_source` to match the
checkpoint family:

- `mpc`: Microsoft Planetary Computer Sentinel-2 L2A and Sentinel-1 RTC.
- `aws`: AWS Open Data Earth-search Sentinel-2 L2A and ASF OPERA RTC-S1. This
  is not currently recommended because rslearn does not have an OPERA data source yet.

Tessera uses different normalization statistics for the two checkpoint families.
The model wrapper still accepts a `data_source` argument for compatibility, but
normalization statistics are now selected by `TesseraNormalize`.

## Inputs

The model expects three raster inputs:

- `s2`: Sentinel-2 bands in Tessera order:
  `B04, B02, B03, B08, B8A, B05, B06, B07, B11, B12`.
- `s1_ascending`: Sentinel-1 ascending orbit bands `vv, vh`.
- `s1_descending`: Sentinel-1 descending orbit bands `vv, vh`.

All inputs must have timestamps. The wrapper uses each timestamp midpoint to compute
day-of-year features.

`TesseraNormalize` expects Sentinel-1 values in standard power decibels
(`10 * log10(linear)`). For raw linear RTC sources, apply
`Sentinel1ToDecibels` before `TesseraNormalize`. If your Sentinel-1 layers are
already stored in standard dB, skip `Sentinel1ToDecibels` and still run
`TesseraNormalize`.

## Example

Here is an example of a model config to compute embeddings with Tessera:

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.singletask.SingleTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.tessera.Tessera
            init_args:
              # Replace with the path to the downloaded checkpoint!
              checkpoint_path: /path/to/tessera_v1_1_mpc_encoder.pt
              pixel_batch_size: 1024
        decoder:
          - class_path: rslearn.train.tasks.embedding.EmbeddingHead
    optimizer:
      class_path: rslearn.train.optimizer.AdamW
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    inputs:
      s2:
        data_type: raster
        layers: ["sentinel2_l2a"]
        bands: ["B04", "B02", "B03", "B08", "B8A", "B05", "B06", "B07", "B11", "B12"]
        passthrough: true
        load_all_layers: true
      s1_ascending:
        data_type: raster
        layers: ["sentinel1_ascending"]
        bands: ["vv", "vh"]
        passthrough: true
        load_all_layers: true
      s1_descending:
        data_type: raster
        layers: ["sentinel1_descending"]
        bands: ["vv", "vh"]
        passthrough: true
        load_all_layers: true
    transforms:
      - class_path: rslearn.train.transforms.sentinel1.Sentinel1ToDecibels
        init_args:
          selectors: ["s1_ascending", "s1_descending"]
      - class_path: rslearn.models.tessera.TesseraNormalize
        init_args:
          data_source: mpc
```

The wrapper returns float32 feature maps with 192 channels by default, so it can be
used with `EmbeddingHead` and `RslearnWriter`.

## Data Source Example

Here is an example for obtaining Sentinel-2 L2A and Sentinel-1 RTC (with separated
ascending and descending images) that are compatible with Tessera. It creates 12
chronological 30-day mosaics per layer, but note that Tessera can input more frequent
images.

```json
{
  "layers": {
    "sentinel2_l2a": {
      "type": "raster",
      "band_sets": [
        {
          "bands": [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12"
          ],
          "dtype": "uint16"
        }
      ],
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
          "min_matches": 12,
          "per_period_mosaic_reverse_time_order": false,
          "period_duration": "30d",
          "space_mode": "MOSAIC"
        }
      }
    },
    "sentinel1_ascending": {
      "type": "raster",
      "band_sets": [
        {
          "bands": ["vv", "vh"],
          "dtype": "float32"
        }
      ],
      "data_source": {
        "class_path": "rslearn.data_sources.planetary_computer.Sentinel1",
        "init_args": {
          "cache_dir": "cache/planetary_computer",
          "query": {
            "sar:instrument_mode": {
              "eq": "IW"
            },
            "sar:polarizations": {
              "eq": [
                "VV",
                "VH"
              ]
            },
            "sat:orbit_state": {
              "eq": "ascending"
            }
          }
        },
        "ingest": false,
        "query_config": {
          "max_matches": 12,
          "min_matches": 12,
          "per_period_mosaic_reverse_time_order": false,
          "period_duration": "30d",
          "space_mode": "MOSAIC"
        }
      }
    },
    "sentinel1_descending": {
      "type": "raster",
      "band_sets": [
        {
          "bands": ["vv", "vh"],
          "dtype": "float32"
        }
      ],
      "data_source": {
        "class_path": "rslearn.data_sources.planetary_computer.Sentinel1",
        "init_args": {
          "cache_dir": "cache/planetary_computer",
          "query": {
            "sar:instrument_mode": {
              "eq": "IW"
            },
            "sar:polarizations": {
              "eq": [
                "VV",
                "VH"
              ]
            },
            "sat:orbit_state": {
              "eq": "descending"
            }
          }
        },
        "ingest": false,
        "query_config": {
          "max_matches": 12,
          "min_matches": 12,
          "per_period_mosaic_reverse_time_order": false,
          "period_duration": "30d",
          "space_mode": "MOSAIC"
        }
      }
    }
  }
}
```
