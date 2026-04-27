## rslearn.data_sources.earthdaily.Biophysical

Biophysical variables on [EarthDaily](https://earthdaily.com/) platform (EDAgro layers).

See [EarthDaily Setup](earthdaily.md) for required dependency/credentials.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.earthdaily.Biophysical",
  "init_args": {
    // Required: which biophysical variable to fetch.
    // One of: "lai", "fapar", "fcover".
    "variable": "lai",
    // Optional: layer whose prepared item groups should be mirrored. When set, the
    // biophysical layer selects corresponding products by item ID instead of doing
    // an independent geometry/time STAC search.
    "match_source_layer": null,
    // Optional: template used with match_source_layer to derive biophysical item IDs.
    // Defaults to "{source_item_name}_{variable_upper}" when match_source_layer is set.
    "match_source_item_template": null,
    // Optional: STAC API `query` filter passed to searches.
    "query": null,
    // Optional: STAC item property to sort by before grouping/matching (default null).
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

### Available Bands

Band names correspond 1:1 with the selected `variable`:
- `variable: "lai"` → `lai`
- `variable: "fapar"` → `fapar`
- `variable: "fcover"` → `fcover`

### Matching a Sentinel-2 Layer

By default, `Biophysical` searches the selected LAI/FAPAR/FCOVER collection by the
window geometry, time range, and optional STAC query. That finds products in the same
space/time range, but it does not guarantee that the selected biophysical product was
derived from the same Sentinel-2 scene selected for another layer.

Set `match_source_layer` when you need source-scene alignment. During
`rslearn dataset prepare`, the source layer must appear earlier in the dataset config so
it is prepared first. The biophysical layer then reads the source layer's prepared item
groups for each window and mirrors those groups with corresponding EarthDaily products.

For EarthDaily products whose IDs are derived from Sentinel-2 scene IDs, the default
template is usually enough:

```jsonc
{
  "layers": {
    "sentinel2": {
      "type": "raster",
      "band_sets": [
        {
          "bands": ["B04", "B03", "B02"],
          "dtype": "float32"
        }
      ],
      "data_source": {
        "class_path": "rslearn.data_sources.earthdaily.Sentinel2",
        "init_args": {
          "assets": ["red", "green", "blue"]
        },
        "query_config": {
          "space_mode": "MOSAIC",
          "max_matches": 1
        }
      }
    },
    "lai": {
      "type": "raster",
      "band_sets": [
        {
          "bands": ["lai"],
          "dtype": "float32"
        }
      ],
      "data_source": {
        "class_path": "rslearn.data_sources.earthdaily.Biophysical",
        "init_args": {
          "variable": "lai",
          "match_source_layer": "sentinel2"
        },
        "query_config": {
          "min_matches": 1
        }
      }
    }
  }
}
```

With this configuration, if the `sentinel2` layer selected:

```text
S2C_31TEJ_20250424_0_L2A
```

then the `lai` layer selects:

```text
S2C_31TEJ_20250424_0_L2A_LAI
```

For FAPAR and FCOVER, set `variable` to `fapar` or `fcover`; the default template will
produce `_FAPAR` or `_FCOVER` suffixes.

### Match Template Fields

`match_source_item_template` is a Python format string. It supports these fields:

- `source_item_name`: the serialized `name` of the item selected by the source layer.
- `source_product_id`: the serialized `product_id` of the source item, if present.
- `variable`: the selected biophysical variable, e.g. `lai`.
- `variable_upper`: the selected biophysical variable uppercased, e.g. `LAI`.

The default is:

```text
{source_item_name}_{variable_upper}
```

Use `source_product_id` if the source layer item name is not the Sentinel-2 product ID
but the serialized item includes `product_id`:

```jsonc
{
  "class_path": "rslearn.data_sources.earthdaily.Biophysical",
  "init_args": {
    "variable": "fapar",
    "match_source_layer": "sentinel2",
    "match_source_item_template": "{source_product_id}_{variable_upper}"
  }
}
```

If a corresponding product is missing, the mirrored item group is skipped. Existing
missing-layer behavior is controlled by `query_config.min_matches`: with `min_matches`
greater than the number of successfully matched groups, the layer/window is rejected.
