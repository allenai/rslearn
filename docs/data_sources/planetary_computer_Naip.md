## rslearn.data_sources.planetary_computer.Naip

NAIP imagery on Microsoft Planetary Computer. Direct materialization is supported.

This data source uses the Planetary Computer `naip` collection, and reads the `image`
asset which contains four bands: `R`, `G`, `B`, `NIR`.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.planetary_computer.Naip",
  "init_args": {
    // See rslearn.data_sources.planetary_computer.PlanetaryComputer.
    "query": null,
    "sort_by": null,
    "sort_ascending": true,
    "timeout_seconds": 10
  }
}
```

### Available Bands

Pixel values are uint8.

- R
- G
- B
- NIR

Note: NAIP provides a single 4-band GeoTIFF asset (`image`). Internally, rslearn will
still ingest/read this full 4-band asset, but you can configure your raster layer band
set to materialize any subset of `["R", "G", "B", "NIR"]` (for example `["NIR"]`).
