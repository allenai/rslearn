## rslearn.data_sources.worldcover.WorldCover

This data source is for the ESA WorldCover 2021 land cover map.

For details about the land cover map, see
https://registry.opendata.aws/esa-worldcover-vito/.

The 2,651 tiles are served as Cloud-Optimized GeoTIFFs from the public
`s3://esa-worldcover` bucket. There is a GeoJSON index that we download and cache
locally for use during the prepare stage. Direct materialization from the COGs is
suppotred.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.worldcover.WorldCover",
  "init_args": {
    // Directory to cache the tile grid GeoJSON index.
    "metadata_cache_dir": "cache/worldcover"
  }
}
```

### Available Bands

A single band "B1" (uint8) contains the WorldCover class ID. The class IDs are:

| Value | Label                    |
|------:|--------------------------|
|    10 | Tree cover               |
|    20 | Shrubland                |
|    30 | Grassland                |
|    40 | Cropland                 |
|    50 | Built-up                 |
|    60 | Bare / sparse vegetation |
|    70 | Snow and ice             |
|    80 | Permanent water bodies   |
|    90 | Herbaceous wetland       |
|    95 | Mangroves                |
|   100 | Moss and lichen          |
