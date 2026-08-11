## rslearn.data_sources.meta_canopy_height.MetaCanopyHeightV2

This data source is for the Meta/WRI Canopy Height Maps v2 (CHMv2), a global
1-meter resolution top-of-canopy height map derived from Maxar/Vantor Vivid2
satellite imagery using a DINOv3-based model.

For details, see https://registry.opendata.aws/dataforgood-fb-forestsv2/ and the
preprint https://arxiv.org/abs/2603.06382.

The ~213,000 tiles are served as Cloud-Optimized GeoTIFFs from the public
`s3://dataforgood-fb-data` bucket (region `us-east-1`), under the prefix
`forests/v2/global/dinov3_global_chm_v2_ml3/chm/`. Tiles are named by their
zoom-10 web mercator quadkey and stored in EPSG:3857; reprojection to the window
projection happens automatically on read.

The bucket includes a `tiles.geojson` index (tile extent + quadkey) that we
download and cache locally for use during the prepare stage. Because it is the
authoritative list of existing tiles, we only create items for tiles that exist,
so ocean / no-data areas are simply skipped rather than producing empty items.

Data is read directly from the COGs at materialize time; this data source does
not support ingest, so set `"ingest": false` on the layer.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.meta_canopy_height.MetaCanopyHeightV2",
  "init_args": {
    // Directory to cache the tiles.geojson index.
    "metadata_cache_dir": "cache/meta_canopy_height"
  }
}
```

Example layer configuration:

```jsonc
{
  "canopy_height": {
    "type": "raster",
    "band_sets": [{ "dtype": "uint8", "bands": ["canopy_height"] }],
    "data_source": {
      "name": "rslearn.data_sources.meta_canopy_height.MetaCanopyHeightV2",
      "init_args": { "metadata_cache_dir": "cache/meta_canopy_height" },
      "ingest": false
    }
  }
}
```

### Available Bands

A single band `canopy_height` (uint8) contains the top-of-canopy height above
ground in meters. The nodata value is 255.

Because the source data is 1m resolution, set the layer/window resolution or a
`zoom_offset` appropriate to your target resolution; coarser windows are
resampled from the native 1m pixels on read.
