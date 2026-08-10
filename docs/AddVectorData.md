# Add Vector Data

Labels for tasks like window classification, window regression, and object detection are
stored as vector data. Here, we show how to import a folder of vector files into an
rslearn dataset using the LocalFiles data source, or how to write vector data programmatically.

For more complete tutorials, see:
- [Windows from GeoJSON](examples/WindowsFromGeojson.md): imports points and uses them as detection labels.
- [Find Stadiums](examples/FindStadiums.md) starts with point labels but turns them into per-window rasters to train a segmentation model.

## Import Vector Files with LocalFiles

Suppose `source_data/vectors/` contains vector data readable by Fiona, like GeoJSONs or Shapefiles.
Add a vector layer to `config.json`:

```json
{
  "layers": {
    "label": {
      "type": "vector",
      "data_source": {
        "class_path": "rslearn.data_sources.local_files.LocalFiles",
        "init_args": {
          "src_dir": "source_data/vectors"
        }
      }
    }
  }
}
```

After [creating windows](DatasetAddWindows.md), import only this layer:

```bash
rslearn dataset prepare --root ./dataset --enabled-layers label
rslearn dataset ingest --root ./dataset --enabled-layers label
rslearn dataset materialize --root ./dataset --enabled-layers label
```

`LocalFiles` matches each window against the source files; then, it reprojects and crops
the features from matching source files to the projection and bounds of each window.

## Write Vector Data Programmatically

For more flexibility, vector data can be programmatically written. Suppose we have a
`points.csv` file like this, and want to create a fixed-size window centered at each
point for training a window classification model:

```csv
lon,lat,start_time,end_time,class_id
-122.33,47.62,2024-06-01T00:00:00+00:00,2024-09-01T00:00:00+00:00,0
-122.67,45.52,2023-07-01T00:00:00+00:00,2023-10-01T00:00:00+00:00,1
```

Define a vector layer in `config.json`:

```json
{
  "layers": {
    "label": {
      "type": "vector"
    }
  }
}
```

We can then create the windows and write the vector data (it will be a GeoJSON with a
single feature per window):

```python
import csv
from datetime import datetime

import shapely
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.vector_format import GeojsonVectorFormat

WINDOW_SIZE = 256
dataset = Dataset(UPath("./dataset"))

with open("points.csv") as f:
  rows = list(csv.DictReader(f))

for index, row in enumerate(rows):
  # First, we find the UTM CRS that is appropriate for each point. The returned
  # Projection encodes both the CRS and an x/y resolution, which we set to
  # 10 m/pixel.
  lon = float(row["lon"])
  lat = float(row["lat"])
  projection = get_utm_ups_projection(lon, lat, 10, -10)

  # Now we re-project the point to pixel coordinates so we know how to set
  # the bounds of the Window.
  point = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None)
  projected_point = point.to_projection(projection)
  center_x = int(projected_point.shp.x)
  center_y = int(projected_point.shp.y)
  half_size = WINDOW_SIZE // 2
  bounds = (
    center_x - half_size,
    center_y - half_size,
    center_x + half_size,
    center_y + half_size,
  )

  # Create the window. The projection/bounds/time_range specify the
  # spatiotemporal box that the window corresponds to. Its metadata will be
  # written to `./dataset/windows/default/point_{index}/metadata.json`.
  window = Window(
    storage=dataset.storage,
    group="default",
    name=f"point_{index}",
    projection=projection,
    bounds=bounds,
    time_range=(
      datetime.fromisoformat(row["start_time"]),
      datetime.fromisoformat(row["end_time"]),
    ),
    data_factory=dataset.window_data_storage_factory,
  )
  window.save()

  # Now that the window is created, we can write the vector data. We turn the
  # point STGeometry into a Feature by adding the class_id as a property.
  feature = Feature(
    projected_point,
    {"class_id": int(row["class_id"])},
  )
  # Then we use `LayerWriter.write_vector` to write it to the window.
  with window.data.open_layer_writer("label") as writer:
    writer.write_vector(
      # The VectorFormat controls how the vector features are stored.
      # GeojsonVectorFormat just encodes them in a GeoJSON, which will be
      # at `{window_dir}/layers/label/data.geojson`.
      vector_format=GeojsonVectorFormat(),
      features=[feature],
    )
  window.mark_layer_completed("label")
```

For window classification, `ClassificationTask` reads the label from the first
vector feature that has the configured property name. We also set `read_class_id`
so it interprets the property as an integer class ID instead of a class name:

```yaml
data:
  init_args:
    task:
      class_path: rslearn.train.tasks.classification.ClassificationTask
      init_args:
        property_name: class_id
        classes: [class_0, class_1]
        read_class_id: true
```
