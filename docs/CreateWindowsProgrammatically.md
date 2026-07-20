# Create Windows Programmatically

Windows can be created programmatically when window creation needs logic that is not
available in `rslearn dataset add_windows`, such as a different time range for each
input feature.

The examples below assume the dataset `config.json` has already been written in
`./dataset`.

## Create One Window

The code below illustrates how to use the rslearn Python API to create a single window.
In this case, the window is 512x512 pixels centered at (-122.33, 47.62), and uses an
appropriate UTM projection for that location at 10 m/pixel.

```python
from datetime import UTC, datetime

import shapely
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection

lon, lat = -122.33, 47.62

# We first create a Dataset object, which represents the rslearn dataset. It
# provides references to the dataset's window metadata and window layer data
# storage that we need to pass when creating the Window.
dataset = Dataset(UPath("./dataset"))

# Get an appropriate UTM projection for the longitude/latitude, at 10 m/pixel.
# We generally use a negative resolution for the y_resolution so that pixel
# coordinates increase as we go down to lower latitudes, which matches typical
# treatment of image coordinates. The Projection specifies both a CRS and an
# x/y resolution.
projection: Projection = get_utm_ups_projection(lon, lat, 10, -10)

# Now re-project the point to pixel coordinates in the UTM Projection.
# STGeometry specifies the Projection, shapely geometry, and time range. Here,
# geometry is the point in its original WGS84 longitude/latitude coordinates,
# while projected_geometry is the point in pixel coordinates (UTM, 10 m/pixel).
geometry = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None)
projected_geometry = geometry.to_projection(projection)

# rslearn window coordinates need to be in integer pixel coordinates, so we
# cast.
center_x = int(projected_geometry.shp.x)
center_y = int(projected_geometry.shp.y)

# Now we create the window. Each window in an rslearn dataset sits in its own
# directory; in this case, it is `./dataset/default/seattle/` since the group
# name is "default" and the window name is "seattle". dataset.storage is a
# WindowStorage which stores metadata. The default is a file-based storage, so
# the metadata about the window's spatial and temporal bounds appears at
# `metadata.json` in the window directory.
window = Window(
  storage=dataset.storage,
  group="default",
  name="seattle",
  projection=projection,
  bounds=(
    center_x - 256,
    center_y - 256,
    center_x + 256,
    center_y + 256,
  ),
  time_range=(
      datetime(2024, 6, 1, tzinfo=UTC),
      datetime(2024, 9, 1, tzinfo=UTC),
  ),
  data_factory=dataset.window_data_storage_factory,
)
window.save()
print(window.bounds)
```

## Create Windows from Per-Feature Metadata

In this slightly more complicated example, we iterate through a GeoJSON and create one
window for each feature. Normally the `rslearn dataset add_windows` command can create
windows like this, but it applies the same time range to all windows; in this example,
we use the `start_time` and `end_time` attributes of each feature.

For example, save this as `regions.geojson`:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "name": "seattle_train",
        "start_time": "2024-06-01T00:00:00+00:00",
        "end_time": "2024-09-01T00:00:00+00:00",
        "split": "train"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[-122.36, 47.60], [-122.30, 47.60], [-122.30, 47.65], [-122.36, 47.65], [-122.36, 47.60]]]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "name": "portland_val",
        "start_time": "2023-07-01T00:00:00+00:00",
        "end_time": "2023-10-01T00:00:00+00:00",
        "split": "val"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[-122.70, 45.49], [-122.63, 45.49], [-122.63, 45.54], [-122.70, 45.54], [-122.70, 45.49]]]
      }
    }
  ]
}
```

The following script creates one bounds-matching window per polygon. It copies the
`split` property into `window.options`, where model split configurations can use it
later.

```python
import json
from datetime import datetime

import shapely
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection

with open("regions.geojson") as f:
    feature_collection = json.load(f)

dataset = Dataset(UPath("./dataset"))
for feature_data in feature_collection["features"]:
    properties = feature_data["properties"]
    shape = shapely.geometry.shape(feature_data["geometry"])
    geometry = STGeometry(WGS84_PROJECTION, shape, None)

    # Similar to the single window example, we use the UTM projection covering
    # this location at 10 m/pixel.
    centroid = shape.centroid
    projection = get_utm_ups_projection(centroid.x, centroid.y, 10, -10)

    # Then we project the polygon to pixel coordinates and get its bounds.
    projected_geometry = geometry.to_projection(projection)
    bounds = tuple(int(value) for value in projected_geometry.shp.bounds)
    time_range = (
        datetime.fromisoformat(properties["start_time"]),
        datetime.fromisoformat(properties["end_time"]),
    )

    window = Window(
      storage=dataset.storage,
      group="default",
      name=properties["name"],
      projection=projection,
      bounds=bounds,
      time_range=time_range,
      options={"split": properties["split"]},
      data_factory=dataset.window_data_storage_factory,
    )
    window.save()
```

Above, we tag the windows with a `split` key. Later, the model configuration can
select those windows through split tags:

```yaml
data:
  init_args:
    train_config:
      tags:
        split: train
    val_config:
      tags:
        split: val
```
