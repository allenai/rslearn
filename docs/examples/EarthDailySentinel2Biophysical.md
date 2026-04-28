## EarthDaily Sentinel-2 + Biophysical Products

This example shows how to materialize EarthDaily Sentinel-2 imagery alongside derived
biophysical products such as LAI, FAPAR, and FCOVER.

The Sentinel-2 imagery and biophysical variables are separate EarthDaily STAC
collections. Rather than adding a special combined data source, prepare each layer
independently, then align the biophysical layer item groups to the Sentinel-2 item
groups by item name before materialization.

Use `earthdaily.Sentinel2L2A` for the Sentinel-2 layer in this workflow. Its
`sentinel-2-l2a` item IDs match the naming scheme used by the EarthDaily biophysical
products.

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
          "dtype": "uint16"
        }
      ],
      "data_source": {
        "class_path": "rslearn.data_sources.earthdaily.Sentinel2L2A",
        "init_args": {
          "assets": ["B02", "B03", "B04", "B08"],
          "harmonize": true
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

EarthDaily biophysical item IDs are derived from the Sentinel-2 scene IDs by appending
the uppercased variable name. For example, LAI products may be named:

```text
S2C_31TEJ_20250424_0_L2A -> S2C_31TEJ_20250424_0_L2A_LAI
```

Align the biophysical layer groups to the Sentinel-2 layer groups with that name
relationship:

```bash
python docs/examples/align_item_groups_by_name.py \
  --root ./dataset \
  --reference-layer sentinel2 \
  --target-layers lai fapar fcover \
  --target-name-template '{reference_item_name}_{target_layer_upper}'
```

The script rewrites the prepared item groups for `lai`, `fapar`, and `fcover` so they
have the same group order as `sentinel2`. For each reference group, the target item name
is formatted from the first Sentinel-2 item name and selected from the target layer's
already-prepared candidate groups.

Finally materialize:

```bash
rslearn dataset materialize --root ./dataset
```

### Notes

- Run the alignment script after prepare and before materialize.
- Use `--dry-run` first to check how many windows would be updated.
- Increase the target layers' `query_config.max_matches` if the expected biophysical
  item is not available in the already-prepared candidate groups.
- Rewritten target layer data is marked unmaterialized so materialize will read the
  aligned groups.
- If you need to change the alignment settings, rerun prepare for the target layers to
  restore their full candidate item groups before rerunning the script.
- The default `--group-time-range-source target` keeps the matched target layer's
  prepared request time ranges. This is usually the least surprising option for
  materialization.
- If target layer names differ from the product suffixes, run the script separately per
  target layer with a literal template such as
  `--target-name-template '{reference_item_name}_LAI'`.
