## rslearn.data_sources.earthdaily.Sentinel2L2A

Sentinel-2 L2A data on [EarthDaily](https://earthdaily.com/) platform using collection
`sentinel-2-l2a`. The underlying assets are from the AWS Open Data Sentinel-2 L2A COG
collection.

Naming note: `earthdaily.Sentinel2L2A` is the compatibility source for the older
`sentinel-2-l2a` collection. For EarthDaily Collection 1 (`sentinel-2-c1-l2a`) with
scale/offset-applied reflectance, use `rslearn.data_sources.earthdaily.Sentinel2`.

This class uses the same Sentinel-2 asset keys as
`rslearn.data_sources.planetary_computer.Sentinel2` (`B01`-`B12` except `B10`, plus
`B8A`, `SCL`, and `visual`).

Authentication and dependency requirements are the same as
`rslearn.data_sources.earthdaily.Sentinel2` (optional `earthdaily[platform]`,
`EDS_CLIENT_ID`, `EDS_SECRET`, `EDS_AUTH_URL`, `EDS_API_URL`).

For collection lifecycle context (`sentinel-2-c1-l2a` replacing `sentinel-2-l2a`) and
known archive gaps, see [earthdaily.Sentinel2 (C1 L2A)](earthdaily_Sentinel2C1L2A.md#collection-status).

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
- SCL
- R
- G
- B

### Harmonization

Harmonization first tries to build the callback from Sentinel-2 metadata XML
(`product_metadata`) using `get_harmonize_callback(...)` when that asset is present.

If metadata is missing or cannot be read while `harmonize=true`, rslearn raises
that error instead of silently skipping harmonization.

If metadata XML is available but does not itself indicate whether an offset is
present, rslearn falls back to the processing baseline encoded in STAC
`properties["sentinel:product_id"]` when available, then to the item ID, and
otherwise to the acquisition date:

- scenes with baseline 04.00+ include a +1000 DN offset for reflectance bands,
- harmonization undoes this by subtracting 1000 DN (with clipping at zero),
- fallback applies this adjustment when the scene ID encodes baseline 04.00+,
- if the scene ID does not expose a processing baseline, fallback applies the
  adjustment for acquisitions on/after 2022-01-25,
- `SCL` and `visual` (TCI RGB) are not harmonized.
