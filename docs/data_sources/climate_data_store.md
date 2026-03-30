## rslearn.data_sources.climate_data_store.ERA5Land

Base class for ingesting ERA5 land data from the Copernicus Climate Data Store.

An API key must be provided either in the data source configuration or via the CDSAPI_KEY
environment variable. You can acquire an API key by registering at the Climate Data Store
website (https://cds.climate.copernicus.eu/).

Valid bands are the variable names listed on the CDS dataset page (use the **API request**
tool to check valid values). Note it is necessary to replace "_" with "-" in the variable
names, e.g. `total_precipitation` becomes `total-precipitation`.

**Performance:** Specifying the `bounds` parameter to limit the geographic extent of data
retrieval is highly recommended, especially for hourly data, as downloading global data can
be very slow and resource-intensive.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.climate_data_store.ERA5Land",
  "init_args": {
    // Optional API key. If not provided in the data source configuration, it must be set
    // via the CDSAPI_KEY environment variable.
    "api_key": null,
    // Optional bounding box as [min_lon, min_lat, max_lon, max_lat]. Recommended to speed
    // up ingestion, especially for hourly data.
    // Example: [-122.4, 47.6, -122.3, 47.7]
    "bounds": null
  }
}
```

## rslearn.data_sources.climate_data_store.ERA5LandMonthlyMeans

This data source is for ingesting ERA5 land monthly averaged data from the Copernicus
Climate Data Store. This corresponds to the `reanalysis-era5-land-monthly-means` dataset.

See rslearn.data_sources.climate_data_store.ERA5Land above for common configuration
and usage information.

Valid bands: See the [CDS dataset page](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means?tab=download).

## rslearn.data_sources.climate_data_store.ERA5LandHourly

This data source is for ingesting ERA5 land hourly data from the Copernicus Climate Data
Store. This corresponds to the `reanalysis-era5-land` dataset.

See rslearn.data_sources.climate_data_store.ERA5Land above for common configuration
and usage information. **Note:** The `bounds` parameter is especially important for hourly
data to avoid very slow global downloads.

Valid bands: See the [CDS dataset page](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download).
