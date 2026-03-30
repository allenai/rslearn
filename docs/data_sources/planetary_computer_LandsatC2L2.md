## rslearn.data_sources.planetary_computer.LandsatC2L2

Landsat 8/9 Collection 2 Level-2 data on Microsoft Planetary Computer. Direct
materialization is supported.

See the dataset page: https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2

This data source uses Landsat-style band identifiers (for compatibility with
`rslearn.data_sources.aws_landsat.LandsatOliTirs`).

Note: this is Level-2 data rather than Level-1, so it does not provide Level-1 bands
like B8 (panchromatic), B9 (cirrus / OLI_B9), and B11 (thermal / TIRS_B11). If you need those, use the slower
`rslearn.data_sources.aws_landsat.LandsatOliTirs` data source.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.planetary_computer.LandsatC2L2",
  "init_args": {
    // Optional list of bands to expose. Values can be Landsat band identifiers
    // ("B1", "B2", ..., "B10") or the STAC common names / STAC `eo:bands[].name`
    // aliases listed below (e.g. "red" or "OLI_B4").
    "band_names": null,
    // The optional STAC query filter to use. Defaults to selecting Landsat 8 and 9. For example, set
    // to {"platform": ["landsat-8"]} to use Landsat 8 only.
    "query": {"platform": ["landsat-8", "landsat-9"]},
    // See rslearn.data_sources.planetary_computer.PlanetaryComputer.
    "sort_by": null,
    "sort_ascending": true,
    "timeout_seconds": 10
  }
}
```

### Available Bands

Default bands (uint16):
- B1
- B2
- B3
- B4
- B5
- B6
- B7
- B10

Band mapping (rslearn band → STAC `common_name` / STAC `eo:bands[].name`):
- B1 → coastal / OLI_B1
- B2 → blue / OLI_B2
- B3 → green / OLI_B3
- B4 → red / OLI_B4
- B5 → nir08 / OLI_B5
- B6 → swir16 / OLI_B6
- B7 → swir22 / OLI_B7
- B10 → lwir11 / TIRS_B10
