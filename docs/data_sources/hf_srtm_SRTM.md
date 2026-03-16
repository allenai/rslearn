## rslearn.data_sources.hf_srtm.SRTM

Elevation data from the Shuttle Radar Topography Mission (SRTM), served from the Ai2
Hugging Face mirror at https://huggingface.co/datasets/allenai/srtm-global-void-filled.

The data is split into 1x1-degree tiles. SRTM1 (1 arc-second, ~30m resolution) is
available for some regions (primarily US territories), while SRTM3 (3 arc-second, ~90m
resolution) is available globally. By default, SRTM1 is preferred when available for
higher resolution.

No credentials are needed.

### Configuration

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

### Available Bands

The data source should be configured with a single band set containing a single band.
The band name can be set arbitrarily, but "dem" or "srtm" is suggested. The data type
should be int16 to match the source data.

Items from this data source do not come with a time range.
