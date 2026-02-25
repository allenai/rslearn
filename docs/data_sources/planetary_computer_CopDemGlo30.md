## rslearn.data_sources.planetary_computer.CopDemGlo30

Copernicus DEM GLO-30 (30m) data on Microsoft Planetary Computer. Direct materialization
is supported.

This is a "static" dataset (no meaningful temporal coverage), so it ignores window time
ranges when searching and matching STAC items.

The Copernicus DEM items expose the DEM GeoTIFF as the `data` asset, and this data
source maps it to a single band.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.planetary_computer.CopDemGlo30",
  "init_args": {
    // See rslearn.data_sources.planetary_computer.PlanetaryComputer.
    "timeout_seconds": 10
  }
}
```

### Available Bands

The data source should be configured with a single band set containing a single band.
The band name can be set arbitrarily, but "dem" is suggested.
