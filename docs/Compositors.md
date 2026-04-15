## Compositors

This document lists compositors available for raster layers in rslearn. Set
`compositing_method` in a layer config to either:

- a built-in enum name (for example `"FIRST_VALID"`), or
- a custom compositor config (`class_path` + `init_args`).

See [Dataset Configuration](DatasetConfig.md) for where this field appears.

### Built-In Compositors

| Compositing Method | Notes |
|---|---|
| [FIRST_VALID](compositors/built_in.md#first_valid) | Default. First non-nodata pixel in item order. |
| [MEAN](compositors/built_in.md#mean) | Per-pixel mean across valid pixels. |
| [MEDIAN](compositors/built_in.md#median) | Per-pixel median across valid pixels. |
| [SPATIAL_MOSAIC_TEMPORAL_STACK](compositors/built_in.md#spatial_mosaic_temporal_stack) | Spatial first-valid per timestep, stacked along T. |
| [TEMPORAL_MEAN](compositors/built_in.md#temporal_reducers) | Temporal reduction over stacked timesteps. |
| [TEMPORAL_MAX](compositors/built_in.md#temporal_reducers) | Temporal reduction over stacked timesteps. |
| [TEMPORAL_MIN](compositors/built_in.md#temporal_reducers) | Temporal reduction over stacked timesteps. |

### Custom Cloud-Aware Ranking Compositors

These compositors reorder items inside each materialized item group, then apply
FIRST_VALID in that ranked order.

| Class Path | Description |
|---|---|
| [rslearn.dataset.omni_cloud_mask.OmniCloudMaskFirstValid](compositors/omni_cloud_mask_OmniCloudMaskFirstValid.md) | Uses OmniCloudMask model inference on R/G/NIR. |
| [rslearn.dataset.sentinel2_scl.Sentinel2SCLFirstValid](compositors/sentinel2_scl_Sentinel2SCLFirstValid.md) | Uses Sentinel-2 SCL classes to score cloudiness. |
