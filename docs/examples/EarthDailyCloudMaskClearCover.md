## EarthDaily Cloud-Mask Clear-Cover Selection

This example shows how to integrate EarthDaily EDA cloud-mask ranking into the normal
rslearn `prepare` → `ingest` → `materialize` workflow.

The Sentinel-2 imagery and EDA cloud masks are separate EarthDaily STAC collections.
Prepare both layers first, then run a small selection step that scores the prepared
cloud-mask candidates over each window AOI. The script rewrites the prepared
Sentinel-2 layer to the related L2A item with the highest clear cover. After that,
normal rslearn ingest/materialize commands operate on the selected item.

The workflow is:

1. Configure a Sentinel-2 layer and a cloud-mask layer.
2. Run `rslearn dataset prepare`.
3. Score the prepared cloud-mask candidates and rewrite the Sentinel-2 prepared item
   group to the clearest related image.
4. Run `rslearn dataset ingest` if the selected layers use `ingest: true`.
5. Run `rslearn dataset materialize`.
6. In the training config, convert the materialized `cloud_mask` layer to a binary
   mask and apply it to Sentinel-2 inputs.

The EDA cloud-mask class values are:

- `0`: nodata
- `1`: clear
- `2`: cloud
- `3`: cloud shadow
- `4`: thin cloud

### Layers

Here is a representative dataset configuration snippet. The cloud-mask layer uses
`INTERSECTS` and a higher `max_matches` so prepare keeps multiple candidate masks for
the post-prepare selection step.

```jsonc
{
  "layers": {
    "sentinel2": {
      "type": "raster",
      "band_sets": [
        {
          "bands": ["B02", "B03", "B04", "B08"],
          "dtype": "uint16",
          "nodata_value": 0
        }
      ],
      "resampling_method": "bilinear",
      "data_source": {
        "class_path": "rslearn.data_sources.earthdaily.Sentinel2L2A",
        "init_args": {
          "assets": ["B02", "B03", "B04", "B08"],
          "harmonize": true
        },
        "query_config": {
          "space_mode": "INTERSECTS",
          "max_matches": 1
        },
        "ingest": false
      }
    },
    "cloud_mask": {
      "type": "raster",
      "band_sets": [
        {
          "bands": ["cloud-mask"],
          "dtype": "uint8",
          "nodata_value": 0
        }
      ],
      "resampling_method": "nearest",
      "data_source": {
        "class_path": "rslearn.data_sources.earthdaily.Sentinel2EDACloudMask",
        "init_args": {
          "assets": ["cloud-mask"],
          "cloud_cover_max": 80,
          "sort_items_by": "datetime"
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

This example uses `Sentinel2L2A` because the EDA cloud-mask products point to
`sentinel-2-l2a` through `eda:derived_from_collection_id`.

### Workflow

First prepare the dataset normally:

```bash
rslearn dataset prepare --root ./dataset
```

At this point each window has prepared candidate groups for `cloud_mask`. Each
candidate is a cloud-mask item intersecting the window and time range. Run the
selection helper:

```bash
python docs/examples/select_earthdaily_sentinel2_by_cloud_mask.py \
  --root ./dataset \
  --cloud-mask-layer cloud_mask \
  --sentinel2-layer sentinel2 \
  --dry-run
```

The script scores each prepared cloud-mask candidate over the exact rslearn window
projection and bounds:

```text
clear_cover = count(cloud-mask == 1) / total_window_pixels
valid_cover = count(cloud-mask != 0) / total_window_pixels
```

Candidates are sorted by `clear_cover`, then `valid_cover`, then raw clear pixel count.
The script reads `eda:derived_from_collection_id` and `eda:derived_from_item_id` from
the selected cloud-mask STAC item, fetches that related Sentinel-2 item, and rewrites
the prepared `sentinel2` layer to contain only that selected item group.

Run it without `--dry-run` to update the prepared window metadata:

```bash
python docs/examples/select_earthdaily_sentinel2_by_cloud_mask.py \
  --root ./dataset \
  --cloud-mask-layer cloud_mask \
  --sentinel2-layer sentinel2
```

By default the helper also rewrites the `cloud_mask` layer to the selected cloud-mask
item, so materializing both layers yields the chosen Sentinel-2 image and the mask used
to choose it. Add `--keep-cloud-mask-candidates` if you only want to rewrite the
Sentinel-2 layer.

If your layers use `ingest: true`, ingest after the selection step so rslearn only
downloads the selected items:

```bash
rslearn dataset ingest --root ./dataset
```

Finally materialize:

```bash
rslearn dataset materialize --root ./dataset
```

### Training-Time Masking

The selected `cloud_mask` layer can be used at training time to set non-clear
Sentinel-2 pixels to the Sentinel-2 nodata value. This keeps the materialized
Sentinel-2 raster unchanged while making the masking explicit in the training
configuration.

Configure the training inputs to load both the Sentinel-2 image and the selected
cloud-mask raster:

```yaml
inputs:
  image:
    data_type: "raster"
    layers: ["sentinel2"]
    bands: ["B02", "B03", "B04", "B08"]
    passthrough: true
    dtype: FLOAT32
  cloud_mask:
    data_type: "raster"
    layers: ["cloud_mask"]
    bands: ["cloud-mask"]
    passthrough: true
    dtype: UINT8
```

Then convert EarthDaily EDA cloud-mask classes to a binary mask and apply it to the
Sentinel-2 input:

```yaml
transforms:
  - class_path: rslearn.train.transforms.earthdaily.EarthDailyCloudMaskToMask
    init_args:
      cloud_mask_selector: "cloud_mask"
      output_selector: "mask"
      # EarthDaily EDA value 1 is clear. Values 0, 2, 3, and 4 are masked out.
      clear_values: [1]
  - class_path: rslearn.train.transforms.mask.Mask
    init_args:
      selectors: ["image"]
      mask_selector: "mask"
      # Match the Sentinel-2 band-set nodata_value configured above.
      mask_value: 0
```

The resulting `image` tensor keeps pixels where `cloud-mask == 1` and sets all other
pixels (`0` nodata, `2` cloud, `3` cloud shadow, `4` thin cloud) to `mask_value`.

### Selection Helper

The helper script is available at
[`docs/examples/select_earthdaily_sentinel2_by_cloud_mask.py`](select_earthdaily_sentinel2_by_cloud_mask.py).

Useful options:

- `--min-clear-cover 0.8`: reject windows where the clearest candidate has less than
  80% clear cover over the AOI.
- `--groups` and `--windows`: restrict updates to specific prepared windows.
- `--keep-cloud-mask-candidates`: leave the cloud-mask layer's prepared candidates
  untouched and only rewrite the Sentinel-2 layer.
- `--env-file .env`: load EarthDaily credentials from a dotenv file.

The script expects each cloud-mask prepared group to contain one item, so configure the
cloud-mask layer with `query_config.space_mode: "INTERSECTS"`. It validates that the
selected cloud-mask item derives from the same EarthDaily collection used by the
configured Sentinel-2 layer.
