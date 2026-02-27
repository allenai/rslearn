## rslearn.data_sources.google_earth_engine.GoogleSatelliteEmbeddings

This data source is for Google Satellite Embeddings (AlphaEarth Embeddings) from Google
Earth Engine. The embedding values are stored as unsigned 16-bit integers from 0 to
16383, computed by multiplying the original [-1, 1] floating point values by 8192 and
adding 8192.

### Configuration

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

### Available Bands

The bands are named "A00" through "A63". Here is an example configuration:

```json
{
  "layers": {
    "gse": {
      "band_sets": [
        {
          "bands": [
            "A00",
            "A01",
            "A02",
            "A03",
            "A04",
            "A05",
            "A06",
            "A07",
            "A08",
            "A09",
            "A10",
            "A11",
            "A12",
            "A13",
            "A14",
            "A15",
            "A16",
            "A17",
            "A18",
            "A19",
            "A20",
            "A21",
            "A22",
            "A23",
            "A24",
            "A25",
            "A26",
            "A27",
            "A28",
            "A29",
            "A30",
            "A31",
            "A32",
            "A33",
            "A34",
            "A35",
            "A36",
            "A37",
            "A38",
            "A39",
            "A40",
            "A41",
            "A42",
            "A43",
            "A44",
            "A45",
            "A46",
            "A47",
            "A48",
            "A49",
            "A50",
            "A51",
            "A52",
            "A53",
            "A54",
            "A55",
            "A56",
            "A57",
            "A58",
            "A59",
            "A60",
            "A61",
            "A62",
            "A63"
          ],
          "dtype": "uint16"
        }
      ],
      "data_source": {
        "gcs_bucket_name": "YOUR_GCS_BUCKET_NAME",
        "index_cache_dir": "cache/gse",
        "ingest": false,
        "name": "rslearn.data_sources.google_earth_engine.GoogleSatelliteEmbeddings",
        "service_account_credentials": "/path/to/gee_credentials.json",
        "service_account_name": "SERVICE_ACCOUNT_NAME",
      },
      "resampling_method": "nearest",
      "type": "raster"
    }
  }
}
```
