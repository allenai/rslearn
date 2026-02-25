## rslearn.data_sources.worldcover.WorldCover

This data source is for the ESA WorldCover 2021 land cover map.

For details about the land cover map, see https://worldcover2021.esa.int/.

This data source downloads the 18 zip files that contain the map. They are then
extracted, yielding 2,651 GeoTIFF files. These are then used with
`rslearn.data_sources.local_files.LocalFiles` to implement the data source.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.worldcover.WorldCover",
  "init_args": {
    // Required local path to store the downloaded zip files and extracted GeoTIFFs.
    "worldcover_dir": "cache/worldcover"
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
