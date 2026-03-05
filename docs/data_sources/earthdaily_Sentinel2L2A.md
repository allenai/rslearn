## rslearn.data_sources.earthdaily.Sentinel2L2A

Sentinel-2 L2A data on [EarthDaily](https://earthdaily.com/) platform using collection
`sentinel-2-l2a`. The underlying assets are from the AWS Open Data Sentinel-2 L2A COG
collection.

This class uses the same Sentinel-2 asset keys as
`rslearn.data_sources.planetary_computer.Sentinel2` (`B01`-`B12` except `B10`, plus
`B8A` and `visual`).

Authentication and dependency requirements are the same as
`rslearn.data_sources.earthdaily.Sentinel2C1L2A` (optional `earthdaily[platform]`,
`EDS_CLIENT_ID`, `EDS_SECRET`, `EDS_AUTH_URL`, `EDS_API_URL`).

For collection lifecycle context (`sentinel-2-c1-l2a` replacing `sentinel-2-l2a`) and
known archive gaps, see [earthdaily.Sentinel2C1L2A](earthdaily_Sentinel2C1L2A.md#collection-status).

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.earthdaily.Sentinel2L2A",
  "init_args": {
    // Flag (default false) to harmonize pixel values across different processing
    // baselines.
    "harmonize": false,
    // Optional: list of Sentinel-2 asset keys to fetch.
    // Example: ["B02", "B03", "B04", "B08"]
    "assets": null,
    // Optional STAC query/sorting controls from rslearn.data_sources.earthdaily.EarthDaily.
    "query": null,
    "sort_by": null,
    "sort_ascending": true,
    "cache_dir": null,
    "timeout": "10s",
    "max_retries": 3,
    "retry_backoff_factor": 5.0
  }
}
```

### Available Bands

These bands are available:

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
- R
- G
- B

### Harmonization

Harmonization first tries to build the callback from Sentinel-2 metadata XML
(`product_metadata`) using `get_harmonize_callback(...)` when that asset is present.

If metadata is unavailable or cannot be read, rslearn falls back to date-based
harmonization:

- scenes with baseline 04.00+ include a +1000 DN offset for non-visual bands,
- harmonization undoes this by subtracting 1000 DN (with clipping at zero),
- fallback applies this adjustment for acquisitions on/after 2022-01-25,
- `visual` (TCI RGB) is not harmonized.
