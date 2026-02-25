## rslearn.data_sources.planetary_computer.Hls2S30

HLS v2 Sentinel-2 ([S30](https://planetarycomputer.microsoft.com/dataset/hls2-s30))
data on Microsoft Planetary Computer. Direct materialization is supported.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.planetary_computer.Hls2S30",
  "init_args": {
    // Optional list of bands to expose.
    "band_names": null,
    // Optional list of Sentinel-2 platforms to include.
    "platforms": ["sentinel-2a", "sentinel-2b", "sentinel-2c"],
    // Optional STAC query filter. Defaults to platform filter above if omitted.
    "query": null,
    // See rslearn.data_sources.planetary_computer.PlanetaryComputer.
    "sort_by": null,
    "sort_ascending": true,
    "timeout_seconds": 10
  }
}
```

### Available Bands

The default band set includes the reflectance bands:
- B01 (coastal)
- B02 (blue)
- B03 (green)
- B04 (red)
- B08 (nir)
- B10 (cirrus)
- B11 (swir16)
- B12 (swir22)

Band names may be provided as either asset keys (B01, B02, ...) or common names
(coastal, blue, green, red, nir, cirrus, swir16, swir22).

## rslearn.data_sources.planetary_computer.Hls2L30

HLS v2 Landsat ([L30](https://planetarycomputer.microsoft.com/dataset/hls2-l30))
ata on Microsoft Planetary Computer. Direct materialization is supported.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.planetary_computer.Hls2L30",
  "init_args": {
    // Optional list of bands to expose.
    "band_names": null,
    // Optional list of Landsat platforms to include.
    "platforms": ["landsat-8", "landsat-9"],
    // Optional STAC query filter. Defaults to platform filter above if omitted.
    "query": null,
    // See rslearn.data_sources.planetary_computer.PlanetaryComputer.
    "sort_by": null,
    "sort_ascending": true,
    "timeout_seconds": 10
  }
}
```

### Available Bands

The default band set includes:
- B01 (coastal)
- B02 (blue)
- B03 (green)
- B04 (red)
- B05 (nir)
- B06 (swir16)
- B07 (swir22)
- B09 (cirrus)
- B10 (lwir11)
- B11 (lwir12)

Band names may be provided as either asset keys (B01, B02, ...) or common names
(coastal, blue, green, red, nir, swir16, swir22, cirrus, lwir11, lwir12).
