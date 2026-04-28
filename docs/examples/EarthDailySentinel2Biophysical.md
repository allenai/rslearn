## EarthDaily Sentinel-2 + Biophysical Products

This example shows how to materialize EarthDaily Sentinel-2 imagery alongside derived
biophysical products such as LAI, FAPAR, and FCOVER.

The Sentinel-2 imagery and biophysical variables are separate EarthDaily STAC
collections. Rather than adding a special combined data source, prepare each layer
independently, then align the biophysical layer item groups to the Sentinel-2 item
groups before materialization.

### Layers

Here is a representative dataset configuration snippet. The target biophysical layers
use a larger `max_matches` during prepare so the alignment script has candidate items to
choose from.

```jsonc
{
  "layers": {
    "sentinel2": {
      "type": "raster",
      "band_sets": [
        {
          "bands": ["B02", "B03", "B04", "B08"],
          "dtype": "float32"
        }
      ],
      "data_source": {
        "class_path": "rslearn.data_sources.earthdaily.Sentinel2",
        "init_args": {
          "assets": ["blue", "green", "red", "nir"],
          "apply_scale_offset": true
        },
        "query_config": {
          "space_mode": "INTERSECTS",
          "max_matches": 4
        },
        "ingest": false
      }
    },
    "lai": {
      "type": "raster",
      "band_sets": [
        {
          "bands": ["lai"],
          "dtype": "float32",
          "nodata_value": 0
        }
      ],
      "data_source": {
        "class_path": "rslearn.data_sources.earthdaily.Biophysical",
        "init_args": {
          "variable": "lai"
        },
        "query_config": {
          "space_mode": "INTERSECTS",
          "max_matches": 20
        },
        "ingest": false
      }
    }
  }
}
```

Add similar layers for FAPAR and FCOVER by changing the layer name, band name, and
`variable` to `fapar` or `fcover`.

### Workflow

First prepare the dataset normally:

```bash
rslearn dataset prepare --root ./dataset
```

Then align the biophysical layer groups to the Sentinel-2 layer groups:

```bash
python docs/examples/align_item_groups_by_timestamp.py \
  --root ./dataset \
  --reference-layer sentinel2 \
  --target-layers lai fapar fcover \
  --match-mode nearest_timestamp \
  --max-delta 1d
```

The script rewrites the prepared item groups for `lai`, `fapar`, and `fcover` so they
have the same group order as `sentinel2`. Each target group is selected from the target
layer's already-prepared candidate groups by nearest item timestamp.

Finally materialize:

```bash
rslearn dataset materialize --root ./dataset
```

### Notes

- Run the alignment script after prepare and before materialize.
- Use `--dry-run` first to check how many windows would be updated.
- Increase the target layers' `query_config.max_matches` if no biophysical candidate is
  found within `--max-delta`.
- Rewritten target layer data is marked unmaterialized so materialize will read the
  aligned groups.
- If you need to change the alignment settings, rerun prepare for the target layers to
  restore their full candidate item groups before rerunning the script.
- The default `--group-time-range-source target` keeps the matched target layer's
  prepared request time ranges. This is usually the least surprising option for
  materialization.
- For data sources with predictable item name relationships, the script also supports
  `--match-mode name_regex` with `--reference-name-regex` and
  `--target-name-replacement`.
