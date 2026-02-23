## Data Sources

This document details the built-in data sources in rslearn and how they should be
configured in the [dataset configuration file](DatasetConfig.md).

We also include source-specific recommendations for settings for the `dataset prepare`,
`dataset ingest`, and `dataset materialize` commands below. Unless otherwise noted, it
is generally suggested to use:

```
rslearn dataset prepare --root ... --workers NUM_WORKERS
rslearn dataset ingest --root ... --workers NUM_WORKERS --no-use-initial-job
rslearn dataset materialize --root ... --workers NUM_WORKERS --no-use-initial-job
```

Replace NUM_WORKERS with a number of workers depending on the available system memory
(may require trial and error).

When using multiple workers, rslearn by default first processes one task in the main
thread before parallelizing the remaining tasks across workers, but
`--no-use-initial-job` disables this functionality. We use the default functionality
for `dataset prepare` since data sources often perform processing, like downloading and
caching an index file, that should not be parallellized.

If dataset operations are failing due to errors, you can enable retries by passing e.g.
`--retry-max-attempts 5 --retry-backoff-seconds 5`.

### rslearn.data_sources.aws_landsat.LandsatOliTirs

This data source is for Landsat 8/9 OLI-TIRS imagery on AWS. It uses the usgs-landsat
S3 bucket maintained by USGS. It includes Tier 1/2 scenes but not Real-Time scenes. See
https://aws.amazon.com/marketplace/pp/prodview-ivr4jeq6flk7u for details about the
bucket.

This data source supports direct materialization: if the "ingest" flag is set false,
then ingestion will be skipped and windows will be directly populated from windowed
reads of the underlying cloud-optimized GeoTIFFs on S3.

```jsonc
{
  "class_path": "rslearn.data_sources.aws_landsat.LandsatOliTirs",
  "init_args": {
    // Required cache directory to cache product metadata files. Unless prefixed by a
    // protocol (like "file://..."), it is joined with the dataset path (i.e., specifies
    // a sub-directory within the dataset folder.
    "metadata_cache_dir": "cache/landsat",
    // Sort by this attribute, either null (default, meaning arbitrary ordering) or
    // "cloud_cover".
    "sort_by": null
  }
}
```

Available bands:
- B1
- B2
- B3
- B4
- B5
- B6
- B7
- B8
- B9
- B10
- B11

### rslearn.data_sources.aws_open_data.Naip

This data source is for NAIP imagery on AWS. It uses the naip-source requester pays
bucket maintained by Esri. See https://registry.opendata.aws/naip/ for more
information. AWS credentials must be configured for use with boto3.

```jsonc
{
  "class_path": "rslearn.data_sources.aws_open_data.Naip",
  "init_args": {
    // Required cache directory to cache index shapefiles. Unless prefixed by a protocol
    // (like "file://..."), it is joined with the dataset path.
    "index_cache_dir": "cache/naip",
    // Whether to build an rtree index to accelerate prepare lookups, default false. It
    // is recommended to set this true when processing more than a few windows.
    "use_rtree_index": false,
    // Limit the search to these states (list of their two-letter codes). This can
    // substantially accelerate lookups when the rtree index is disabled, since by
    // default (null) it has to scan through all of the states.
    // Example: ["wa", "or"]
    "states": null,
    // Limit the search to these years. Like with states, this can speed up lookups when
    // the rtree index is disabled.
    // Example: [2023, 2024]
    "years": null
  }
}
```

Available bands:
- R
- G
- B
- IR

### rslearn.data_sources.aws_open_data.Sentinel2

This data source is for Sentinel-2 L1C and L2A imagery on AWS. It uses the
sentinel-s2-l1c and sentinel-s2-l2a S3 buckets maintained by Sinergise. They state the
data is "added regularly, usually within few hours after they are available on
Copernicus OpenHub".

See https://aws.amazon.com/marketplace/pp/prodview-2ostsvrguftb2 for details about the
buckets. AWS credentials must be configured for use with boto3.

```jsonc
{
  "class_path": "rslearn.data_sources.aws_open_data.Sentinel2",
  "init_args": {
    // Required modality, either "L1C" or "L2A".
    "modality": "L1C",
    // Required cache directory to cache product metadata files.
    "metadata_cache_dir": "cache/sentinel2",
    // Sort by this attribute, either null (default, meaning arbitrary ordering) or
    // "cloud_cover".
    "sort_by": null,
    // Flag (default false) to harmonize pixel values across different processing
    // baselines (recommended), see
    // https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
    "harmonize": false
  }
}
```

Available bands:
- B01
- B02
- B03
- B04
- B05
- B06
- B07
- B08
- B09
- B10 (L1C only)
- B11
- B12
- B8A
- R (from TCI asset; derived from B04)
- G (from TCI asset; derived from B03)
- B (from TCI asset; derived from B02)

### rslearn.data_sources.aws_sentinel1.Sentinel1

This data source is for Sentinel-1 GRD imagery on AWS. It uses the sentinel-s1-l1c S3
bucket maintained by Sinergise. See
https://aws.amazon.com/marketplace/pp/prodview-uxrsbvhd35ifw for details about the
bucket.

Although other Sentinel-1 scenes are available, the data source currently only supports
the GRD IW DV scenes (vv+vh bands). It uses the Copernicus API for metadata search
(prepare step).

```jsonc
{
  "class_path": "rslearn.data_sources.aws_sentinel1.Sentinel1",
  "init_args": {
    // Optional orbit direction to filter by, either "ASCENDING" or "DESCENDING". The
    // default is to not filter (so both types of scenes are included/mixed).
    "orbit_direction": null
  }
}
```

Available bands:
- vv
- vh

### rslearn.data_sources.aws_sentinel2_element84.Sentinel2

This data source is for Sentinel-2 L2A imagery from the Element 84 Earth Search STAC
API on AWS. It uses the `s3://sentinel-cogs` S3 bucket which provides Cloud-Optimized
GeoTIFFs, enabling direct materialization without ingestion.

See https://aws.amazon.com/marketplace/pp/prodview-ykj5gyumkzlme for details.

The bucket is public and free so no credentials are needed.

```jsonc
{
  "class_path": "rslearn.data_sources.aws_sentinel2_element84.Sentinel2",
  "init_args": {
    // Optional STAC query filter.
    // Example: {"eo:cloud_cover": {"lt": 20}}
    "query": null,
    // Sort by this STAC property, e.g. "eo:cloud_cover".
    "sort_by": null,
    // Whether to sort ascending or descending (default ascending).
    "sort_ascending": true,
    // Optional directory to cache discovered items.
    "cache_dir": null,
    // Flag (default false) to harmonize pixel values across different processing
    // baselines (recommended), see
    // https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
    "harmonize": false,
    // Timeout for requests.
    "timeout": "10s"
  }
}
```

Available bands:
- B01 (uint16, from coastal asset)
- B02 (uint16, from blue asset)
- B03 (uint16, from green asset)
- B04 (uint16, from red asset)
- B05 (uint16, from rededge1 asset)
- B06 (uint16, from rededge2 asset)
- B07 (uint16, from rededge3 asset)
- B08 (uint16, from nir asset)
- B09 (uint16, from nir09 asset)
- B11 (uint16, from swir16 asset)
- B12 (uint16, from swir22 asset)
- B8A (uint16, from nir08 asset)
- R, G, B (uint8, from visual asset)

### rslearn.data_sources.climate_data_store.ERA5Land

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

### rslearn.data_sources.climate_data_store.ERA5LandMonthlyMeans

This data source is for ingesting ERA5 land monthly averaged data from the Copernicus
Climate Data Store. This corresponds to the `reanalysis-era5-land-monthly-means` dataset.

See rslearn.data_sources.climate_data_store.ERA5Land above for common configuration
and usage information.

Valid bands: See the [CDS dataset page](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means?tab=download).

### rslearn.data_sources.climate_data_store.ERA5LandHourly

This data source is for ingesting ERA5 land hourly data from the Copernicus Climate Data
Store. This corresponds to the `reanalysis-era5-land` dataset.

See rslearn.data_sources.climate_data_store.ERA5Land above for common configuration
and usage information. **Note:** The `bounds` parameter is especially important for hourly
data to avoid very slow global downloads.

Valid bands: See the [CDS dataset page](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download).

### rslearn.data_sources.earthdatahub.ERA5LandDailyUTCv1

ERA5-Land daily UTC (v1) hosted on EarthDataHub.

See https://earthdatahub.destine.eu/collections/era5/datasets/era5-land-daily for details.

Authentication requires configuring the netrc file (`~/.netrc` on Linux and MacOS) as follows:

```
machine data.earthdatahub.destine.eu
  password <write your personal access token here>
```

Supported bands:
- `d2m`: 2m dewpoint temperature (units: K)
- `e`: evaporation (units: m of water equivalent)
- `pev`: potential evaporation (units: m)
- `ro`: runoff (units: m)
- `sp`: surface pressure (units: Pa)
- `ssr`: surface net short-wave (solar) radiation (units: J m-2)
- `ssrd`: surface short-wave (solar) radiation downwards (units: J m-2)
- `str`: surface net long-wave (thermal) radiation (units: J m-2)
- `swvl1`: volumetric soil water layer 1 (units: m3 m-3)
- `swvl2`: volumetric soil water layer 2 (units: m3 m-3)
- `t2m`: 2m temperature (units: K or °C; see `temperature_unit`)
- `tp`: total precipitation (units: m)
- `u10`: 10m U wind component (units: m s-1)
- `v10`: 10m V wind component (units: m s-1)

```jsonc
{
  "class_path": "rslearn.data_sources.earthdatahub.ERA5LandDailyUTCv1",
  "init_args": {
    // Optional: URL/path to the EarthDataHub Zarr store.
    "zarr_url": "https://data.earthdatahub.destine.eu/era5/era5-land-daily-utc-v1.zarr",
    // Optional bounding box as [min_lon, min_lat, max_lon, max_lat] (WGS84).
    // Recommended for performance. Dateline-crossing bounds (min_lon > max_lon) are not
    // supported.
    "bounds": null,
    // Units to return for `t2m` (default "kelvin"): "celsius" or "kelvin".
    "temperature_unit": "kelvin",
    // Whether to allow the underlying HTTP client to read environment configuration
    // (including netrc) for auth/proxies (default true).
    "trust_env": true
  }
}
```

### rslearn.data_sources.earthdaily.Sentinel2

Sentinel-2 L2A data on [EarthDaily](https://earthdaily.com/) platform (collection: `sentinel-2-c1-l2a`).

This data source requires the optional `earthdaily[platform]` dependency and EarthDaily
credentials in the environment. The following environment variables must be set:
- `EDS_CLIENT_ID`
- `EDS_SECRET`
- `EDS_AUTH_URL`
- `EDS_API_URL`

By default, this data source applies per-asset scale/offset values from STAC
`raster:bands` metadata (`apply_scale_offset: true`) to convert raw pixel values into
physical units using `physical = raw * scale + offset`. Set `apply_scale_offset: false`
to keep raw values.

Note: EarthDaily may include a preview `thumbnail` asset; rslearn does not ingest/materialize it.

```jsonc
{
  "class_path": "rslearn.data_sources.earthdaily.Sentinel2",
  "init_args": {
    // Whether to apply STAC `raster:bands` scale/offset (default true). Set to false to
    // keep raw values.
    "apply_scale_offset": true,
    // Optional: EarthDaily Sentinel-2 STAC asset keys to fetch (default null).
    // If null and the layer config is available, assets are inferred from the layer's
    // requested band names.
    //
    // Note: this is different from the "Available bands" list below:
    // - "assets" uses EarthDaily STAC asset keys (e.g. "red", "nir", "visual", "scl").
    // - "band_sets[].bands" uses rslearn band names (e.g. "B04", "B08", "R", "scl").
    //
    // Example: ["red", "green", "blue", "nir", "swir16", "swir22", "visual", "scl"]
    "assets": null,
    // Optional: maximum cloud cover (%) to filter items at search time.
    // If set, it takes precedence over cloud_cover_threshold and overrides any
    // `eo:cloud_cover` filter in `query`.
    "cloud_cover_max": null,
    // Optional: default max cloud cover (%) to apply when cloud_cover_max is not set.
    // If set, it overrides any `eo:cloud_cover` filter in `query`.
    // If both cloud_cover_max and cloud_cover_threshold are null, `query` (including
    // any `eo:cloud_cover` filter) is passed through unchanged.
    "cloud_cover_threshold": null,
    // Maximum number of STAC items to fetch per window before rslearn grouping/matching.
    "search_max_items": 500,
    // Optional ordering of items before grouping (useful with SpaceMode.COMPOSITE +
    // CompositingMethod.FIRST_VALID): "cloud_cover" (default), "datetime", or null.
    "sort_items_by": "cloud_cover",
    // Optional: STAC API `query` filter passed to searches.
    // Example: {"s2:product_type": {"eq": "S2MSI2A"}}
    // Note: if cloud_cover_max/cloud_cover_threshold is set, the effective query also
    // includes an `eo:cloud_cover` upper bound.
    "query": null,
    // Optional: STAC item property to sort by before grouping/matching (default null).
    // If set, it takes precedence over sort_items_by.
    "sort_by": null,
    // Whether to sort ascending when sort_by is set (default true).
    "sort_ascending": true,
    // Optional cache directory for cached item metadata.
    "cache_dir": null,
    // Timeout for HTTP asset downloads.
    "timeout": "10s",
    // Retry settings for EarthDaily API client requests (search/get item).
    "max_retries": 3,
    "retry_backoff_factor": 5.0
  }
}
```

Available rslearn band names (select via `band_sets[].bands`; rslearn infers required
EarthDaily assets when `assets` is null):
- B01
- B02
- B03
- B04
- B05
- B06
- B07
- B08
- B09
- B11
- B12
- B8A
- R, G, B (from the `visual` asset)
- scl, aot, wvp

Common EarthDaily asset key → rslearn band name mapping:
- coastal → B01
- blue → B02
- green → B03
- red → B04
- rededge1 → B05
- rededge2 → B06
- rededge3 → B07
- nir → B08
- nir08 → B8A
- nir09 → B09
- swir16 → B11
- swir22 → B12
- visual → R, G, B
- scl → scl
- aot → aot
- wvp → wvp

### rslearn.data_sources.copernicus.Copernicus

This data source is for images from the ESA Copernicus OData API. See
https://documentation.dataspace.copernicus.eu/APIs/OData.html for details about the API
and how to get an access token.

```jsonc
{
  "class_path": "rslearn.data_sources.copernicus.Copernicus",
  "init_args": {
    // Required dictionary mapping from a filename or glob string of an asset inside the
    // product zip file, to the list of bands that the asset contains. An example for
    // Sentinel-2 images is shown.
    "glob_to_bands": {
      "*/GRANULE/*/IMG_DATA/*_B01.jp2": ["B01"],
      "*/GRANULE/*/IMG_DATA/*_TCI.jp2": ["R", "G", "B"]
    },
    // Optional API access token. See https://documentation.dataspace.copernicus.eu/APIs/OData.html
    // for how to get a token. If not set, it is read from the environment variable
    // COPERNICUS_ACCESS_TOKEN. If that environment variable doesn't exist, then we
    // attempt to read the username/password from COPERNICUS_USERNAME and
    // COPERNICUS_PASSWORD (this is useful since access tokens are only valid for an hour).
    "access_token": null,
    // Optional query filter string to include when searching for items. This will be
    // appended to other name, geographic, and sensing time filters where applicable. For
    // example, "Collection/Name eq 'SENTINEL-2'". See the API documentation for more
    // examples.
    "query_filter": null,
    // Optional order by string to include when searching for items. For example,
    // "ContentDate/Start asc". See the API documentation for more examples.
    "order_by": null,
    // Optional product attribute name to sort returned products by that attribute. If
    // set, attributes will be expanded when listing products. Note that while order_by
    // uses the API to order products, the API provides limited options, and sort_by
    // instead is done after the API call.
    "sort_by": null,
    // If sort_by is set, sort in descending order instead of ascending order.
    "sort_desc": false,
    // Timeout for requests in seconds.
    "timeout": 10
  }
}
```

### rslearn.data_sources.copernicus.Sentinel2

This data source is for Sentinel-2 images from the ESA Copernicus OData API.

```jsonc
{
  "class_path": "rslearn.data_sources.copernicus.Sentinel2",
  "init_args": {
    // Required product type, either "L1C" or "L2A".
    "product_type": "L1C",
    // Flag (default false) to harmonize pixel values across different processing
    // baselines (recommended), see
    // https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
    "harmonize": false,
    // See rslearn.data_sources.copernicus.Copernicus for details about the configuration
    // options below.
    "access_token": null,
    "order_by": null,
    "sort_by": null,
    "sort_desc": false,
    "timeout": 10
  }
}
```

Available bands:
- B01
- B02
- B03
- B04
- B05
- B06
- B07
- B08
- B09
- B11
- B12
- B8A
- TCI
- B10 (L1C only)
- AOT (L2A only)
- WVP (L2A only)
- SCL (L2A only)

### rslearn.data_sources.copernicus.Sentinel1

This data source is for Sentinel-1 images from the ESA Copernicus OData API. Currently
only IW GRDH VV+VH products are supported, even though all Sentinel-1 scenes are
available in the data source.

```jsonc
{
  "class_path": "rslearn.data_sources.copernicus.Sentinel1",
  "init_args": {
    // Required product type, must be "IW_GRDH".
    "product_type": "IW_GRDH",
    // Required polarisation, must be "VV_VH".
    "polarisation": "VV_VH",
    // Optional orbit direction to filter by, either "ASCENDING" or "DESCENDING". The
    // default is to not filter (so both types of scenes are included/mixed).
    "orbit_direction": null,
    // See rslearn.data_sources.copernicus.Copernicus for details about the configuration
    // options below.
    "access_token": null,
    "order_by": null,
    "sort_by": null,
    "sort_desc": false,
    "timeout": 10
  }
}
```

### rslearn.data_sources.eurocrops.EuroCrops

This data source is for EuroCrops vector data (v11).

See https://zenodo.org/records/14094196 for details.

While the source data is split into country-level files, this data source uses one item
per year for simplicity. So each item corresponds to all of the country-level files for
that year.

Note that the RO_ny.zip file is not used.

There is no data-source-specific configuration.

The vector features should have `EC_hcat_c` and `EC_hcat_n` properties indicating the
HCAT category code and name respectively.

### rslearn.data_sources.gcp_public_data.Sentinel2

This data source is for Sentinel-2 data on Google Cloud Storage.

Sentinel-2 imagery is available on Google Cloud Storage as part of the Google
Public Cloud Data Program. The images are added with a 1-2 day latency after
becoming available on Copernicus.

See https://cloud.google.com/storage/docs/public-datasets/sentinel-2 for details.

The bucket is public and free so no credentials are needed.

```jsonc
{
  "class_path": "rslearn.data_sources.gcp_public_data.Sentinel2",
  "init_args": {
    // Required cache directory to cache product metadata files and the optional rtree
    // index.
    "index_cache_dir": "cache/sentinel2",
    // Sort by this attribute, either null (default, meaning arbitrary ordering) or
    // "cloud_cover".
    "sort_by": null,
    // Flag (default true) to build an rtree index to speed up product lookups. This can
    // be set false to avoid lengthy (multiple hours) rtree creation time if you are only
    // using a few windows.
    "use_rtree_index": true,
    // Flag (default false) to harmonize pixel values across different processing
    // baselines (recommended), see
    // https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
    "harmonize": false,
    // When using rtree index, only create it for products within this time range. It
    // defaults to null, meaning to create rtree index for entire time range.
    // Example: ["2024-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00"]
    "rtree_time_range": null,
    // By default, if use_rtree_index is true, the rtree index is stored in the
    // index_cache_dir. Set this to override the path for the rtree index and only use
    // index_cache_dir for the product metadata files.
    "rtree_cache_dir": null
  }
}
```

Available bands:
- B01
- B02
- B03
- B04
- B05
- B06
- B07
- B08
- B09
- B10
- B11
- B12
- B8A
- R (from TCI asset; derived from B04)
- G (from TCI asset; derived from B03)
- B (from TCI asset; derived from B02)

### rslearn.data_sources.hf_srtm.SRTM

Elevation data from the Shuttle Radar Topography Mission (SRTM), served from the AI2
Hugging Face mirror at https://huggingface.co/datasets/allenai/srtm-global-void-filled.

The data is split into 1x1-degree tiles. SRTM1 (1 arc-second, ~30m resolution) is
available for some regions (primarily US territories), while SRTM3 (3 arc-second, ~90m
resolution) is available globally. By default, SRTM1 is preferred when available for
higher resolution.

No credentials are needed.

```jsonc
{
  "class_path": "rslearn.data_sources.hf_srtm.SRTM",
  "init_args": {
    // Timeout for requests.
    "timeout": "10s",
    // Optional directory to cache the file list.
    "cache_dir": null,
    // If true, always use 3 arc-second (SRTM3) data even when 1 arc-second (SRTM1) is
    // available. Defaults to false, which prefers SRTM1 for higher resolution.
    "always_use_3arcsecond": false
  }
}
```

The data source should be configured with a single band set containing a single band.
The band name can be set arbitrarily, but "dem" or "srtm" is suggested. The data type
should be int16 to match the source data.

Items from this data source do not come with a time range.

### rslearn.data_sources.google_earth_engine.GEE

This data source is for ingesting images from Google Earth Engine (GEE).

It must be configured with the name of an ee.ImageCollection on GEE. Each ee.Image in
the ee.ImageCollection is treated as a different data source item. A Cloud Storage
bucket is also required to store the intermediate outputs from GEE export jobs.

During the prepare stage, it will first export the metadata (geometry and time range)
of all ee.Image objects in the ee.ImageCollection. Then it will use this to build an
rtree from which prepare requests can be satisfied.

During the ingest stage, it will start export jobs to export images to the bucket. Each
worker will start one job and poll until it finishes before proceeding onto the next
ee.Image to export. After the export finishes, the resulting GeoTIFF(s) are read and
processed into the tile store. Note that export jobs can take several minutes to
complete depending on the size of the image.

This data source does support direct materialization, which can greatly speed up
materialization for sparse windows. Whereas exporting a 10Kx10K image make take 5000
EECU-seconds (and potentially several minutes), exporting a 256x256 image should take
only a few seconds.

```jsonc
{
  "class_path": "rslearn.data_sources.google_earth_engine.GEE",
  "init_args": {
    // Required name of the ee.ImageCollection, e.g. "COPERNICUS/S1_GRD".
    "collection_name": "COPERNICUS/S1_GRD",
    // Required name of the GCS bucket to use to store intermediate outputs from export
    // jobs. You could set up lifecycle rules on this bucket to delete outputs after 1
    // day.
    "gcs_bucket_name": "...",
    // Required service account name.
    "service_account_name": "...",
    // Required path to a local file containing the service account credentials.
    "service_account_credentials": "/etc/credentials/gee_credentials.json",
    // Required directory to store rtree index over the exported ee.Image metadata.
    "index_cache_dir": "cache/gee",
    // Optional filters to aply on the ee.ImageCollection. See Sentinel-1 example below.
    // Currently only equality filters are supported.
    "filters": null
  }
}
```

The available bands depends on the chosen ee.ImageCollection. Here is an example layer
configuration for Sentinel-1. The filters match only ee.Image objects where the
"transmitterReceiverPolarisation" attribute is ["VV", "VH"] and the "instrumentMode"
attribute is "IW".

```json
{
  "sentinel1": {
    "band_sets": [
      {
        "bands": [
          "VV",
          "VH"
        ],
        "dtype": "uint16",
        "format": "geotiff"
      }
    ],
    "data_source": {
      "class_path": "rslearn.data_sources.google_earth_engine.GEE",
      "init_args": {
        "collection_name": "COPERNICUS/S1_GRD",
        "dtype": "float32",
        "filters": [
          [
            "transmitterReceiverPolarisation",
            [
              "VV",
              "VH"
            ]
          ],
          [
            "instrumentMode",
            "IW"
          ]
        ],
        "gcs_bucket_name": "YOUR_BUCKET_NAME",
        "index_fname": "cache/sentinel1_index",
        "service_account_credentials": "/etc/credentials/gee_credentials.json",
        "service_account_name": "YOUR_SERVICE_ACCOUNT_NAME"
      },
      "query_config": {
        "max_matches": 1
      }
    },
    "type": "raster"
  }
}
```

### rslearn.data_sources.google_earth_engine.GoogleSatelliteEmbeddings

This data source is for Google Satellite Embeddings (AlphaEarth Embeddings) from Google
Earth Engine. The embedding values are stored as unsigned 16-bit integers from 0 to
16383, computed by multiplying the original [-1, 1] floating point values by 8192 and
adding 8192.

```jsonc
{
  "class_path": "rslearn.data_sources.google_earth_engine.GoogleSatelliteEmbeddings",
  "init_args": {
    // See rslearn.data_sources.google_earth_engine.GEE for details about these
    // required configuration options.
    "gcs_bucket_name": "...",
    "service_account_name": "...",
    "service_account_credentials": "/etc/credentials/gee_credentials.json",
    "index_cache_dir": "cache/gee"
  }
}
```

### rslearn.data_sources.local_files.LocalFiles

This data source supports ingesting data from local raster or vector files. It is
configured by a source directory that should be a flat structure with the raster or
vector files. Raster files must be readable by rasterio. Vector files must be readable
by fiona.

Each source file is treated as a separate item, so for raster files, each file must
contain the full range of bands, and different files should cover different locations.

```jsonc
{
  "class_path": "rslearn.data_sources.local_files.LocalFiles",
  "init_args": {
    // Required source directory containing the flat structure of raster or vector files.
    // It is relative to the dataset root, so include a protocol if it is outside.
    // Example: "file:///path/to/files/".
    "src_dir": null
  }
}
```

For raster data, the bands will be named "B1", "B2", and so on depending on the number
of bands in the source files.

The time range of all items is null (infinite).

For this dataset, use `--workers 0` (default) so that processing is done in the main
thread. This is because most of the work is spent initializing the data source, due to
the need for identifying the bounds of all of the local files, and so it is best to
just have this done once rather than once in each worker.

### rslearn.data_sources.openstreetmap.OpenStreetMap

This data source is for ingesting OpenStreetMap data from a PBF file.

An existing local PBF file can be used, or if the provided path doesn't exist, then the
global OSM PBF will be downloaded.

This data source uses a single item. If more windows are added, data in the TileStore
will need to be completely re-computed.

```jsonc
{
  "class_path": "rslearn.data_sources.openstreetmap.OpenStreetMap",
  "init_args": {
    // Required list of PBF filenames to read from.
    // If a single filename is provided and it doesn't exist, the latest planet PBF will
    // be downloaded there.
    "pbf_fnames": ["planet-latest.osm.pbf"],
    // Required file to cache the bounds of the different PBF files.
    "bounds_fname": "bounds.json",
    // Required map of categories to extract from the OSM data.
    // Each category specifies a set of restrictions that extract only a certain type of
    // OSM feature, and convert it to a GeoJSON feature.
    "categories": {
      // The key will be added as a "category" property in the resulting GeoJSON
      // features.
      "aerialway_pylon": {
        // Optional limit on the types of features to match. If set, valid list values
        // are "node", "way", "relation".
        // Example: ["node"] to only match nodes.
        "feature_types": null,
        // Optional tag conditions. For each entry (tag_name, values list), only match
        // OSM features with that tag, and if values list is not empty, only match if the
        // tag value matches one element of the values list.
        // The default is null. The example below will only match OSM features with the
        // "aerialway" tag set to "pylon".
        "tag_conditions": {
          "aerialway": [
            "pylon"
          ]
        },
        // Optional tag properties. This is used to save properties of the OSM feature in
        // the resulting GeoJSON feature. It is a list of [tag name, prop name]. If tag
        // tag name exists on the OSM feature, then it will be populated into the prop
        // name property on the GeoJSON feature.
        // Example: [["aerialway:heating", "aerialway:heating"]]
        "tag_properties": null,
        // Optionally convert the OpenStreetMap feature to the specified geometry type
        // (one of "Point", "LineString", "Polygon"). Otherwise, matching nodes result in
        // Points, matching ways result in LineStrings, and matching relations result in
        // Polygons. Note that nodes cannot be converted to LineString/Polygon.
        "to_geometry": "Point"
      }
    }
  }
}
```

### rslearn.data_sources.planet.Planet

This data source is still experimental.

### rslearn.data_sources.planet_basemap.PlanetBasemap

This data source is still experimental.

### rslearn.data_sources.planetary_computer.PlanetaryComputer

This data source is for raster data from Microsoft Planetary Computer. See their
[Data Catalog](https://planetarycomputer.microsoft.com/catalog).

This data source supports direct materialization: if the "ingest" flag is set false,
then ingestion will be skipped and windows will be directly populated from windowed
reads of the underlying cloud-optimized GeoTIFFs on Azure Blob Storage.

```jsonc
{
  "class_path": "rslearn.data_sources.planetary_computer.PlanetaryComputer",
  "init_args": {
    // Required collection name, e.g. "landsat-c2-l2" or "modis-17A2HGF-061".
    "collection_name": null,
    // Required map from asset name to list of bands in the asset to download.
    // You may need to perform a STAC search to see what the asset names are.
    // Example: {"B8A": ["B8A"], "visual": ["R", "G", "B"]}
    "asset_bands": null,
    // Include this query argument for STAC searches.
    // Example: {"sar:instrument_mode": {"eq": "IW"}}
    "query": null,
    // Sort by this property in the STAC items.
    // Example: "eo:cloud_cover"
    "sort_by": null,
    // Whether to sort ascending or descending (default ascending).
    "sort_ascending": true,
    // Timeout for requests.
    "timeout_seconds": 10
  }
}
```

### rslearn.data_sources.planetary_computer.Sentinel1

Sentinel-1 radiometrically-terrain-corrected data on Microsoft Planetary Computer.
Direct materialization is supported.

It automatically determines the bands to download from the band sets, so all parameters
are optional. The band names are "hh", "hv", "vv", and "vh" depending on the scene.

```jsonc
{
  "class_path": "rslearn.data_sources.planetary_computer.Sentinel1",
  "init_args": {
    // See rslearn.data_sources.planetary_computer.PlanetaryComputer.
    "query": null,
    "sort_by": null,
    "sort_ascending": true,
    "timeout_seconds": 10
  }
}
```

### rslearn.data_sources.planetary_computer.Sentinel2

Sentinel-2 L2A data on Microsoft Planetary Computer. Direct materialization is
supported.

The bands to download are determined from the band sets.

```jsonc
{
  "class_path": "rslearn.data_sources.planetary_computer.Sentinel2",
  "init_args": {
    // Flag (default false) to harmonize pixel values across different processing
    // baselines (recommended), see
    // https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
    "harmonize": false,
    // See rslearn.data_sources.planetary_computer.PlanetaryComputer.
    "query": null,
    "sort_by": null,
    "sort_ascending": true,
    "timeout_seconds": 10
  }
}
```

Available bands:
- B01
- B02
- B03
- B04
- B05
- B06
- B07
- B08
- B09
- B11
- B12
- B8A
- visual

Note that B10 is not present in L2A.

### rslearn.data_sources.planetary_computer.Sentinel3SlstrLST

[Sentinel-3 SLSTR Level-2 Land Surface Temperature (LST) data on Microsoft Planetary
Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-3-slstr-lst-l2-netcdf). This dataset is provided as netCDF swaths; the data source uses the `lst-in`
asset for measurements and the `slstr-geodetic-in` asset for geolocation. During
ingestion it interpolates the swath onto a regular lat/lon grid using linear
weights (this is an approximation; for precise geolocation you may need a custom
workflow).
Direct materialization is not supported, so keep `ingest` set to true.

Available bands:
- LST

```jsonc
{
  "class_path": "rslearn.data_sources.planetary_computer.Sentinel3SlstrLST",
  "init_args": {
    // Stride for sampling geolocation arrays when estimating grid resolution.
    "sample_step": 20,
    // Nodata value used for missing pixels (default 0.0).
    "nodata_value": 0.0,
    // Optional output grid resolution in degrees. If omitted, estimate from geodetic arrays.
    "grid_resolution": null,
    // See rslearn.data_sources.planetary_computer.PlanetaryComputer.
    "query": null,
    "sort_by": null,
    "sort_ascending": true,
    "timeout_seconds": 10
  }
}
```

### rslearn.data_sources.planetary_computer.LandsatC2L2

Landsat 8/9 Collection 2 Level-2 data on Microsoft Planetary Computer. Direct
materialization is supported.

See the dataset page: https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2

This data source uses Landsat-style band identifiers (for compatibility with
`rslearn.data_sources.aws_landsat.LandsatOliTirs`). It defaults to:
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

Note: this is Level-2 data rather than Level-1, so it does not provide Level-1 bands
like B8 (panchromatic), B9 (cirrus / OLI_B9), and B11 (thermal / TIRS_B11). If you need those, use the slower
`rslearn.data_sources.aws_landsat.LandsatOliTirs` data source.

```jsonc
{
  "class_path": "rslearn.data_sources.planetary_computer.LandsatC2L2",
  "init_args": {
    // Optional list of bands to expose. Values can be Landsat band identifiers
    // ("B1", "B2", ..., "B10") or the STAC common names / STAC `eo:bands[].name`
    // aliases listed above (e.g. "red" or "OLI_B4").
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

### rslearn.data_sources.planetary_computer.Naip

NAIP imagery on Microsoft Planetary Computer. Direct materialization is supported.

This data source uses the Planetary Computer `naip` collection, and reads the `image`
asset which contains four bands: `R`, `G`, `B`, `NIR`.

Note: NAIP provides a single 4-band GeoTIFF asset (`image`). Internally, rslearn will
still ingest/read this full 4-band asset, but you can configure your raster layer band
set to materialize any subset of `["R", "G", "B", "NIR"]` (for example `["NIR"]`).
If you need a different asset/band mapping, use
`rslearn.data_sources.planetary_computer.PlanetaryComputer` directly with a custom
`asset_bands` mapping.

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

### rslearn.data_sources.planetary_computer.CopDemGlo30

Copernicus DEM GLO-30 (30m) data on Microsoft Planetary Computer. Direct materialization
is supported.

This is a "static" dataset (no meaningful temporal coverage), so it ignores window time
ranges when searching and matching STAC items.

The Copernicus DEM items expose the DEM GeoTIFF as the `data` asset, and this data
source maps it to a single band.

```jsonc
{
  "class_path": "rslearn.data_sources.planetary_computer.CopDemGlo30",
  "init_args": {
    // Optional band name to use if the layer config is missing from context (default "DEM").
    "band_name": "DEM",
    // See rslearn.data_sources.planetary_computer.PlanetaryComputer.
    "timeout_seconds": 10
  }
}
```

### rslearn.data_sources.usda_cdl.CDL

This data source is for the USDA Cropland Data Layer.

The GeoTIFF data will be downloaded from the USDA website. See
https://www.nass.usda.gov/Research_and_Science/Cropland/SARS1a.php for details about
the data.

There is one GeoTIFF item per year from 2008. Each GeoTIFF spans the entire continental
US, and has a single band.

```jsonc
{
  "class_path": "rslearn.data_sources.usda_cdl.CDL",
  "init_args": {
    // Optional timeout for HTTP requests.
    "timeout_seconds": 10
  }
}
```

The data source yields one band, and the name will match whatever is configured in the
band set. It should be uint8.

### rslearn.data_sources.usgs_landsat.LandsatOliTirs

This data source is for Landsat data from the USGS M2M API.

You can request access at https://m2m.cr.usgs.gov/.

```jsonc
{
  "class_path": "rslearn.data_sources.usgs_landsat.LandsatOliTirs",
  "init_args": {
    // Required M2M API username.
    "username": null,
    // Required M2M API authentication token.
    "token": null,
    // Sort by this attribute, either null (default, meaning arbitrary ordering) or
    // "cloud_cover".
    "sort_by": null
  }
}
```

Available bands:
- B1
- B2
- B3
- B4
- B5
- B6
- B7
- B8
- B9
- B10
- B11

### rslearn.data_sources.soilgrids.SoilGrids

This data source provides access to [ISRIC SoilGrids](https://www.isric.org/explore/soilgrids)
via the public WCS endpoints (e.g. `https://maps.isric.org/mapserv?map=/map/clay.map`).

This source is intended for **direct materialization** (set `"ingest": false` in the
layer's `data_source` config), since data is fetched on-demand per window.

Example (clay, user-provided WCS subset parameters):

```jsonc
{
  "class_path": "rslearn.data_sources.soilgrids.SoilGrids",
  "init_args": {
    "service_id": "clay",
    "coverage_id": "clay_0-5cm_mean"
    // Optional request CRS, defaults to EPSG:3857. You can specify either "EPSG:3857"
    // or the URN form "urn:ogc:def:crs:EPSG::3857".
    // "crs": "EPSG:3857"
  }
}
```

If `"width"`/`"height"` and `"resx"`/`"resy"` are omitted, rslearn will default to
requesting at ~250 m resolution in the request CRS and then reprojecting to the window
grid. For EPSG:4326 requests, SoilGrids requires `"width"`/`"height"` so rslearn will
default those to the window pixel size.

Available bands:
- B1 (float32 recommended; scale/offset applied; set `nodata_vals` to `-32768`)

### rslearn.data_sources.soildb.SoilDB

This data source reads OpenLandMap-SoilDB rasters from the [OpenLandMap static STAC
catalog](https://stac.openlandmap.org/)

Each SoilDB collection links to a single STAC Item which contains multiple GeoTIFF/COG
assets (e.g., different depth ranges, resolutions, and summary statistics). rslearn
expects you to configure a **single-band** band set per layer and choose which STAC
asset to read via `"asset_key"`.

If `"asset_key"` is omitted, rslearn selects a per-collection default when available
(otherwise you must specify `"asset_key"` explicitly):

- `bd.core_iso.11272.2017.g.cm3` → `bd.core_iso.11272.2017.g.cm3_m_30m_b0cm..30cm`
- `oc_iso.10694.1995.wpml` → `oc_iso.10694.1995.wpml_m_30m_b0cm..30cm`
- `oc_iso.10694.1995.mg.cm3` → `oc_iso.10694.1995.mg.cm3_m_30m_b0cm..30cm`
- `ph.h2o_iso.10390.2021.index` → `ph.h2o_iso.10390.2021.index_m_30m_b0cm..30cm`
- `clay.tot_iso.11277.2020.wpct` → `clay.tot_iso.11277.2020.wpct_m_30m_b0cm..30cm`
- `sand.tot_iso.11277.2020.wpct` → `sand.tot_iso.11277.2020.wpct_m_30m_b0cm..30cm`
- `silt.tot_iso.11277.2020.wpct` → `silt.tot_iso.11277.2020.wpct_m_30m_b0cm..30cm`

For `soil.types_ensemble_probabilities`, you must specify `"asset_key"` (there is no
meaningful single default).

Example (one soil type probability layer):
```jsonc
{
  "collection_id": "soil.types_ensemble_probabilities",
  "asset_key": "soil.types_ensemble.aquic.udifluvents_p_30m_s",
  "cache_dir": "cache/soildb",
  "timeout": "30s"
}
```

Detailed config specification:
```jsonc
{
  "class_path": "rslearn.data_sources.soildb.SoilDB",
  "init_args": {
    // Required SoilDB collection id, e.g. "clay.tot_iso.11277.2020.wpct".
    "collection_id": null,
    // Optional STAC asset key. If null, rslearn uses a per-collection default when
    // available; otherwise you must set it explicitly.
    "asset_key": null,
    // Optional cache directory (relative to the dataset path if provided).
    "cache_dir": "cache/soildb",
    // Optional request timeout (jsonargparse accepts strings like \"30s\").
    "timeout": "30s"
  }
}
```

### rslearn.data_sources.worldcereal.WorldCereal

This data source is for the ESA WorldCereal 2021 agricultural land cover map. For
details about the land cover map, see https://esa-worldcereal.org/en.

This data source will download and extract all of the WorldCereal GeoTIFFs to a local
directory. Since different regions are covered with different bands, the data source is
designed to only be configured with one band per layer; to materialize multiple bands,
repeat the data source across multiple layers (with different bands).

```jsonc
{
  "class_path": "rslearn.data_sources.worldcereal.WorldCereal",
  "init_args": {
    // Required local path to extract the WorldCereal GeoTIFF files. For high performance,
    // this should be a local directory; if the dataset is remote, prefix with a protocol
    // ("file://") to use a local directory.
    "worldcereal_dir": "cache/worldcereal"
  }
}
```

Available bands (specify one per layer, with a single-band band set):
- tc-annual_temporarycrops_confidence
- tc-annual_temporarycrops_classification
- tc-maize-main_irrigation_confidence
- tc-maize-main_irrigation_classification
- tc-maize-main_maize_confidence
- tc-maize-main_maize_classification
- tc-maize-second_irrigation_confidence
- tc-maize-second_irrigation_classification
- tc-maize-second_maize_confidence
- tc-maize-second_maize_classification
- tc-springcereals_springcereals_confidence
- tc-springcereals_springcereals_classification
- tc-wintercereals_irrigation_confidence
- tc-wintercereals_irrigation_classification
- tc-wintercereals_wintercereals_confidence
- tc-wintercereals_wintercereals_classification

### rslearn.data_sources.worldcover.WorldCover

This data source is for the ESA WorldCover 2021 land cover map.

For details about the land cover map, see https://worldcover2021.esa.int/.

This data source downloads the 18 zip files that contain the map. They are then
extracted, yielding 2,651 GeoTIFF files. These are then used with
`rslearn.data_sources.local_files.LocalFiles` to implement the data source.

```jsonc
{
  "class_path": "rslearn.data_sources.worldcover.WorldCover",
  "init_args": {
    // Required local path to store the downloaded zip files and extracted GeoTIFFs.
    "worldcover_dir": "cache/worldcover"
  }
}
```

Available bands:
- B1 (uint8)

### rslearn.data_sources.worldpop.WorldPop

This data source is for world population data from worldpop.org.

Currently, this only supports the WorldPop Constrained 2020 100 m Resolution dataset.
See https://hub.worldpop.org/project/categories?id=3 for details.

The data is split by country. We implement with LocalFiles data source for simplicity,
but it means that all of the data must be downloaded first.

```jsonc
{
  "class_path": "rslearn.data_sources.worldpop.WorldPop",
  "init_args": {
    // Required local path to store the downoladed WorldPop data.
    "worldpop_dir": "cache/worldpop"
  }
}
```

### rslearn.data_sources.xyz_tiles.XyzTiles

This data source is for web xyz image tiles (slippy tiles).

These tiles are usually in WebMercator projection, but different CRS can be configured.

```jsonc
{
  "class_path": "rslearn.data_sources.xyz_tiles.XyzTiles",
  "init_args": {
    // Required list of URL templates. The templates must include placeholders for {x}
    // (column), {y} (row), and {z} (zoom level).
    // Example: ["https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.jpg"]
    "url_templates": null,
    // Required list of time ranges. It should match the list of URL templates. This is
    // primarily useful with multiple URL templates, to distinguish which one should be
    // used depending on the window time range. If time is not important, then you can
    // set it arbitrarily.
    // Example: [["2024-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00"]]
    "time_ranges": null,
    // Required zoom level. Currently, a single zoom level must be specified, and tiles
    // will always be read at that zoom level, rather than varying depending on the
    // window resolution.
    // Example: 17 to use zoom level 17.
    "zoom": null,
    // The CRS of the xyz image tiles. Defaults to WebMercator.
    "crs": "EPSG:3857",
    // The total projection units along each axis. Defaults to 40075016.6856 which
    // corresponds to WebMercator. This is used to compute the pixel resolution, i.e. the
    // tiles split the world into 2^zoom tiles along each axis so the resolution is
    // (total_units / 2^zoom / tile_size) units/pixel.
    "total_units": 40075016.6856,
    // Apply an offset to the projection units when converting tile positions. Without an
    // offset, the WebMercator tile columns and rows would range from -2^(zoom-1) to
    // 2^(zoom-1). The default offset is half the default total units so that it
    // corresponds to the standard range from 0 to 2^zoom.
    "offset": 20037508.3428,
    // The size of tiles. The default is 256x256 which is typical.
    "tile_size": 256
  }
}
```
