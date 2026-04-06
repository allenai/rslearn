## rslearn.data_sources.chelsa.CHELSADaily

CHELSA-daily is a global (~1km) climate dataset with one GeoTIFF per day and
variable.

This data source currently targets:
- URL base: `https://os.unil.cloud.switch.ch/chelsa02/chelsa`
- Dataset path: `{extent}/daily/{variable}/{year}/CHELSA_{variable}_{dd}_{mm}_{yyyy}_V.2.1.tif`
- Default time range: 1979-01-01 to 2025-08-29 (inclusive)

Use `query_config.space_mode = SINGLE_COMPOSITE` so all matching daily items are
returned in one group.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.chelsa.CHELSADaily",
  "init_args": {
    // CHELSA variable names to read (if omitted, inferred from layer_config bands).
    // For precipitation, either "pr" or "prec" works: rslearn automatically
    // switches URL variable by year (before 2020 -> "pr", after 2020 -> "prec",
    // overlap year 2020 preserves your requested alias).
    "band_names": ["tas", "pr"],

    // URL path extent, usually "global".
    "extent": "global",

    // Optional dataset bounds (inclusive). Can be date or YYYY-MM-DD string.
    "start_date": "1979-01-01",
    "end_date": "2025-08-29",

    // Optional override for file naming version suffix.
    "version": "V.2.1"
  },
  "query_config": {
    "space_mode": "SINGLE_COMPOSITE"
  }
}
```

### Precipitation Alias Handling

CHELSA daily precipitation has two variable names:
- `pr` in years before 2020 (and also available in 2020)
- `prec` in years after 2020 (and also available in 2020)

When you configure either precipitation alias (`pr` or `prec`), rslearn
automatically chooses the URL variable for each item date so multi-year runs across
the transition work without manual layer changes.

### Supported Variables

| Code | Name | Unit | Description |
|---|---|---|---|
| `clt` | Total Cloud Cover Percentage | `percent` | Total cloud area fraction (reported as percentage) for the full atmospheric column, including large-scale and convective cloud. |
| `hurs` | Near-Surface Relative Humidity | `percent` | Relative humidity with respect to liquid water for `T > 0°C` and with respect to ice for `T < 0°C`. |
| `pr` | Precipitation | `kg m-2 day-1` | Daily precipitation including both liquid and solid phases. |
| `prec` | Precipitation | `kg m-2 day-1` | Alias of CHELSA daily precipitation used in later years. |
| `ps` | Surface Air Pressure | `hPa` | Surface pressure (not mean sea-level pressure). |
| `rsds` | Surface Downwelling Shortwave Flux in Air | `W m-2` | Surface solar irradiance, commonly used for UV-related calculations. |
| `sfcWind` | Near-Surface Wind Speed | `m s-1` | Near-surface (usually 10 m) wind speed. |
| `tas` | Daily Mean Near-Surface Air Temperature | `K` | Near-surface (usually 2 m) daily mean air temperature. |
| `tasmax` | Daily Maximum Near-Surface Air Temperature | `K` | Near-surface (usually 2 m) daily maximum air temperature. |
| `tasmin` | Daily Minimum Near-Surface Air Temperature | `K` | Near-surface (usually 2 m) daily minimum air temperature. |
| `tz` | Air Temperature Lapse Rate | `K m-1` | Rate of change in air temperature with altitude (centennial-period lapse-rate product). |
