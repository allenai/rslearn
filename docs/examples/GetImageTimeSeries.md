## Get a Satellite Image Time Series

This example shows how to download a Sentinel-2 image time series for a list of
locations specified in a GeoJSON file.

Each location gets a window, and rslearn retrieves all available Sentinel-2 L2A images
within a user-specified time range, ordered chronologically.

### Dataset Configuration

Create a folder for the dataset and save the following as `config.json`:

```
export DATASET_PATH=./timeseries_dataset
mkdir -p $DATASET_PATH
```

```json
{
  "layers": {
    "sentinel2": {
      "type": "raster",
      "band_sets": [
        {
          "bands": ["R", "G", "B"],
          "dtype": "uint8"
        }
      ],
      "data_source": {
        "class_path": "rslearn.data_sources.planetary_computer.Sentinel2",
        "ingest": false,
        "init_args": {
          "harmonize": true,
          "query": {
            "eo:cloud_cover": {
              "lt": 50
            }
          },
          "sort_by": "datetime",
          "sort_ascending": true
        },
        "query_config": {
          "space_mode": "INTERSECTS",
          "max_matches": 99
        }
      }
    }
  }
}
```

Key settings:

- `sort_by: "datetime"` with `sort_ascending: true` orders images chronologically.
- `space_mode: "INTERSECTS"` creates a separate item group for each image that
  overlaps the window (as opposed to `MOSAIC` which merges images into coverage
  groups).
- `max_matches: 99` allows up to 99 images. Sentinel-2 revisits every ~5 days, so a
  3-month window yields roughly 18 images per orbit. Set this higher if needed.
- `eo:cloud_cover < 50` filters out heavily cloudy scenes. Adjust or remove as needed.
- Only the 8-bit true-color R, G, B bands are included to keep things simple. Add
  more bands as desired.

### Create Windows

Prepare a GeoJSON file `locations.geojson` with Point features for each location.
For example:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"name": "san_francisco"},
      "geometry": {"type": "Point", "coordinates": [-122.4194, 37.7749]}
    },
    {
      "type": "Feature",
      "properties": {"name": "new_york"},
      "geometry": {"type": "Point", "coordinates": [-73.9857, 40.7484]}
    },
    {
      "type": "Feature",
      "properties": {"name": "london"},
      "geometry": {"type": "Point", "coordinates": [-0.1276, 51.5074]}
    }
  ]
}
```

Create windows using `rslearn dataset add_windows`. Each window is 256x256 pixels at
10 m/pixel (2.56 km), centered at the point, in the appropriate UTM zone:

```
rslearn dataset add_windows \
  --root $DATASET_PATH \
  --group default \
  --fname locations.geojson \
  --utm \
  --resolution 10 \
  --window_size 256 \
  --start 2025-06-01T00:00:00+00:00 \
  --end 2025-09-01T00:00:00+00:00
```

### Retrieve Images

```
rslearn dataset prepare --root $DATASET_PATH
rslearn dataset materialize --root $DATASET_PATH
```

After materialization, each window will have one directory per Sentinel-2 image:

```
windows/default/san_francisco/layers/
  sentinel2/       # first image (earliest)
  sentinel2.1/     # second image
  sentinel2.2/     # third image
  ...
```

Each directory contains a GeoTIFF with the requested bands. The images are ordered
chronologically because of the `sort_by: "datetime"` setting.

### Inspect the Results

List how many images each window received:

```bash
for w in $DATASET_PATH/windows/default/*/; do
  name=$(basename $w)
  count=$(ls -d $w/layers/sentinel2* 2>/dev/null | wc -l)
  echo "$name: $count images"
done
```

Open a time series in QGIS:

```
qgis $DATASET_PATH/windows/default/san_francisco/layers/sentinel2*/R_G_B/geotiff.tif
```

### Tips

- **More bands**: Add bands to the `band_sets` array in `config.json`. Use multiple
  band sets with `zoom_offset` if you want bands at their native resolution (e.g.
  20 m bands like B05-B07 with `zoom_offset: -1`).
- **All bands at 10 m**: Put all 12 bands in a single band set with no `zoom_offset`.
  This resamples coarser bands to 10 m but simplifies downstream processing.
- **Cloud filtering**: Lower the `eo:cloud_cover` threshold for cleaner images, or
  remove it entirely to get every available image.
- **Larger areas**: Increase the `--window_size` value in the `add_windows` command.
