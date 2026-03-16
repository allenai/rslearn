## rslearn.data_sources.soildb.SoilDB

This data source reads OpenLandMap-SoilDB rasters from the [OpenLandMap static STAC
catalog](https://stac.openlandmap.org/).

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

### Configuration

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
    // Optional request timeout (jsonargparse accepts strings like "30s").
    "timeout": "30s"
  }
}
```

### Available Bands

A single band is produced, and the name will match whatever is configured in the band
set.

### Example

Here is an example configuration for one soil type probability layer:

```jsonc
{
  "layers": {
    "soildb": {
      "type": "raster",
      "band_sets": [{
          "bands": ["udifluvents"],
          "dtype": "float32"
      }],
      "data_source": {
        "class_path": "rslearn.data_sources.soildb.SoilDB",
        "init_args": {
          "collection_id": "soil.types_ensemble_probabilities",
          "asset_key": "soil.types_ensemble.aquic.udifluvents_p_30m_s",
          "cache_dir": "cache/soildb",
          "timeout": "30s"
        }
      }
    }
  }
}
```
