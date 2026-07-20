# Add Raster Data

Custom raster data can be added to an rslearn dataset in two ways:

- Use the `LocalFiles` data source when you have a georeferenced collection (e.g. a folder of GeoTIFFs) that you want to align with existing windows. It will spatially match the GeoTIFFs with the windows, and re-project and crop them to the window bounds.
- Write data programmatically using the Python API when more flexibility is needed.

## Import GeoTIFFs with LocalFiles

Suppose `source_data/geotiffs/` contains georeferenced GeoTIFFs and each file contains
three bands. Add a layer like this to the dataset's `config.json`:

```json
{
  "layers": {
    "image": {
      "type": "raster",
      "band_sets": [
        {
          "bands": ["B1", "B2", "B3"],
          "dtype": "uint8"
        }
      ],
      "data_source": {
        "class_path": "rslearn.data_sources.local_files.LocalFiles",
        "init_args": {
          "src_dir": "source_data/geotiffs"
        }
      }
    }
  }
}
```

Note that the `src_dir` is relative to the dataset root directory. To specify an
absolute path on the local filesystem, use e.g. `file:///path/to/source_data/geotiffs/`.

`LocalFiles` derives an item footprint from each raster's georeference and matches that
footprint to the dataset windows. Each automatically discovered file must contain the
complete band set. To combine bands stored in separate files, configure
`raster_item_specs` as described in the
[LocalFiles reference](data_sources/local_files_LocalFiles.md).

After [creating windows](DatasetAddWindows.md), run the normal data-source stages:

```bash
rslearn dataset prepare --root ./dataset --enabled-layers image
rslearn dataset ingest --root ./dataset --enabled-layers image
rslearn dataset materialize --root ./dataset --enabled-layers image
```

The resulting rasters are aligned to each window's projection and bounds.

## Write Rasters Programmatically

Write rasters programmatically when more customizability is needed. For example,
suppose `one_hot_labels.tif` contains one band per class. `SegmentationTask`
expects one integer class ID per pixel, so we can use a script to simultaneously
convert the raster format while also creating a window corresponding to the raster
and writing the raster alongside the window.

First define the output layer in `config.json` (here, we omit a `data_source` section for this layer,
which indicates that the data will be written by the user instead of handled via prepare/ingest/materialize):

```json
{
  "layers": {
    "label": {
      "type": "raster",
      "band_sets": [
        {
          "bands": ["class_id"],
          "dtype": "uint8",
          "nodata_value": 255
        }
      ]
    }
  }
}
```

Then create the window and write the converted labels:

```python
from datetime import UTC, datetime

import numpy as np
import rasterio
from upath import UPath

from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import Projection
from rslearn.utils.raster_array import RasterArray, RasterMetadata
from rslearn.utils.raster_format import (
  GeotiffRasterFormat,
  get_raster_projection_and_bounds,
)

# We first create a Dataset object, which represents the rslearn dataset. It
# provides references to the dataset's window metadata and window layer data
# storage that we need to pass when creating the Window.
dataset = Dataset(UPath("./dataset"))

# Read the source raster: we read the one-hot encoded array, but also process
# the georeference metadata.
with rasterio.open("one_hot_labels.tif") as source:
  # From the georeference metadata, we extract a Projection and bounds. The
  # Projection specifies both a CRS and an x/y resolution. The bounds specify
  # a box in integer pixel coordinates.
  projection: Projection
  bounds: tuple[int, int, int, int]
  projection, bounds = get_raster_projection_and_bounds(source)

  one_hot = source.read()

# Convert the one-hot array into class IDs that are compatible with
# SegmentationTask. We use 255 as a nodata value.
valid = np.any(one_hot != 0, axis=0)
class_ids = np.argmax(one_hot, axis=0).astype(np.uint8)[None, :, :]
class_ids[0, ~valid] = 255

# We create a window corresponding to the raster's extent. Its metadata will be
# written to `./dataset/windows/default/one_hot_labels/metadata.json`.
window = Window(
  storage=dataset.storage,
  group="default",
  name="one_hot_labels",
  projection=projection,
  bounds=bounds,
  time_range=(
    datetime(2024, 1, 1, tzinfo=UTC),
    datetime(2025, 1, 1, tzinfo=UTC),
  ),
  data_factory=dataset.window_data_storage_factory,
)
window.save()

# Now that the window is created, we can write the raster to the window. First
# we open a LayerWriter for the layer that we have in the dataset config.
with window.data.open_layer_writer("label") as writer:
  # When writing the raster, the projection and bounds specify the extent,
  # while the RasterArray contains the actual raster data. The raster_format
  # controls how the raster is stored; GeotiffRasterFormat writes a GeoTIFF
  # file. The list of bands should match one of the band sets configured in
  # the dataset config.
  writer.write_raster(
    bands=["class_id"],
    raster_format=GeotiffRasterFormat(),
    projection=projection,
    bounds=bounds,
    raster=RasterArray(
      chw_array=class_ids,
      metadata=RasterMetadata(nodata_value=255),
    ),
  )
# After the raster is written, we also need to mark the layer completed in the
# window's metadata. Most callers will check which layers are available
# (completed) at a window before reading them.
window.mark_layer_completed("label")
```

A matching task configuration looks like this, reusing the same 255 nodata value:

```yaml
data:
  init_args:
    task:
      class_path: rslearn.train.tasks.segmentation.SegmentationTask
      init_args:
        num_classes: 4
        nodata_value: 255
```
