## Built-In Compositors

Built-in compositors are configured by enum name in `compositing_method`:

```jsonc
{
  "compositing_method": "FIRST_VALID"
}
```

### FIRST_VALID

Selects the first non-nodata value for each pixel/band from items in group order.
This is the default and works well for most mosaicking workflows.

### MEAN

Computes the per-pixel mean across all valid (non-nodata) item values.

Requires `nodata_value` to be known for the band set.

### MEDIAN

Computes the per-pixel median across all valid (non-nodata) item values.

Requires `nodata_value` to be known for the band set.

### SPATIAL_MOSAIC_TEMPORAL_STACK

Builds a `(C, T, H, W)` raster by:

1. Spatially compositing items with first-valid logic at each timestep.
2. Stacking the union of timesteps across all items along T.
3. Clipping timesteps to the window request time range.

This is typically used with `space_mode: "SINGLE_COMPOSITE"` for
multi-temporal products.

Requires `nodata_value` to be known and item rasters to carry timestamps.

### Temporal Reducers

`TEMPORAL_MEAN`, `TEMPORAL_MAX`, and `TEMPORAL_MIN` first build a
SPATIAL_MOSAIC_TEMPORAL_STACK and then reduce along T to a single timestep.

Requires `nodata_value` to be known.
