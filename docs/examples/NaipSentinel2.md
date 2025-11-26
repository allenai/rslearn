## Creating a NAIP + Sentinel-2 Dataset

In this example, we create an rslearn dataset that pairs images from the
[National Agriculture Imagery Program (NAIP)](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-aerial-photography-national-agriculture-imagery-program-naip)
with spatially and temporally aligned Sentinel-2 images. This dataset could be used for
e.g. for super-resolution training.

The example highlights interacting with data sources programmatically, advanced
programmatic window creation, and how to use the NAIP and Sentinel-2 data sources.

We will proceed in two steps:

1. Setup the dataset configuration.
2. Create windows based on the timestamp of available NAIP images.
3. Materialize NAIP and Sentinel-2 images.

## Dataset Setup

We will use Microsoft Planetary Computer to obtain NAIP and Sentinel-2 L2A images. Here
is a suitable dataset configuration file:

```json
{
  "layers": {
    "sentinel2": {
      "band_sets": [{
          "bands": ["R", "G", "B"],
          "dtype": "uint8",
          "zoom_offset": -4
      }, {
          "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
          "dtype": "uint16",
          "zoom_offset": -4
      }],
      "data_source": {
        "class_path": "rslearn.data_sources.planetary_computer.Sentinel2",
        "init_args": {
          "cache_dir": "cache/planetary_computer",
          "harmonize": true,
          "sort_by": "eo:cloud_cover"
        },
        "ingest": false,
        "query_config": {
          "max_matches": 8
        }
      },
      "type": "raster"
    },
    "naip": {
      "band_sets": [{
          "bands": ["R", "G", "B"],
          "dtype": "uint8"
      }, {
          "bands": ["NIR"],
          "dtype": "uint8"
      }],
      "data_source": {
        "class_path": "rslearn.data_sources.planetary_computer.Naip",
        "init_args": {
          "cache_dir": "cache/planetary_computer"
        },
        "ingest": false
      },
      "type": "raster"
    }
  }
}
```

This will obtain eight Sentinel-2 images and one NAIP image matching the spatial bounds
and time range of each window that we create. We create an RGB GeoTIFF for
visualization, but also get all the 16-bit bands for Sentinel-2 and the NIR band for
NAIP for use with any models we may train on this data. The zoom offset for Sentinel-2
keeps the data close to the native Sentinel-2 resolution (10 m/pixel) instead of
resampling all the way up to the window resolution we will use (60 cm/pixel);
specifically, it will be stored at (60 * 2^4) cm/pixel.

Save this dataset configuration to a dataset folder like `./dataset/config.json`.

## Create Windows

NAIP images are captured over the continental US once every two to three years. Thus,
we would like to pick a few spatial locations to create windows, but center the time
range of the window based on the timestamps that NAIP images are available inside that
window. This means window creation will be more complicated than normal, because we
need to do a search to see what NAIP images are available before creating the window.

First, we select a few longitude/latitudes in the continental US, and define a function
to convert that to a 60 cm/pixel UTM window projection and bounds. 60 cm/pixel is the
resolution of most recent NAIP images.

```python
import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection

lon_lats = [
    (-122.32, 47.62), # Seattle
    (-106.29, 32.80), # White Sands National Park
    (-102.851, 30.072), # Amtrak route between Alpine and Sanderson
    (-122.233, 47.588), # Mercer Island Link station, to open in 2026
    (-119.334, 35.665), # Poso Creek Viaduct project for California HSR (viaduct completed in 2020)
    (-96.828, 32.959), # DART Addison Station, opened in October 2025
    (-71.256, 42.471), # Minuteman Bikeway
]

def get_utm_proj_and_bounds(lon: float, lat: float) -> tuple[Projection, PixelBounds]:
    # Convert the lon, lat to a UTM projection and bounds.
    # First create an STGeometry corresponding to the point.
    src_geom = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None)
    # Get appropriate UTM projection (60 cm/pixel), and transform it.
    utm_proj = get_utm_ups_projection(lon, lat, 0.6, -0.6)
    dst_geom = src_geom.to_projection(utm_proj)
    # Get a 1024x1024 image centered at the point.
    bounds = (
        int(dst_geom.shp.x) - 512,
        int(dst_geom.shp.y) - 512,
        int(dst_geom.shp.x) + 512,
        int(dst_geom.shp.y) + 512,
    )
    return utm_proj, bounds

proj_bounds = [get_utm_proj_and_bounds(lon, lat) for lon, lat in lon_lats]
```

Next, for each location, we use the `rslearn.data_sources.planetary_computer.Naip` data
source to get the timestamp of NAIP images:

```python
from datetime import datetime, UTC

from rslearn.data_sources.planetary_computer import Naip
from rslearn.config.dataset import QueryConfig, SpaceMode

data_source = Naip()
timestamps = []
for proj, bounds in proj_bounds:
    # We query the data source using a large time range to see all the available
    # images.
    geom = STGeometry(
        proj,
        shapely.box(*bounds),
        (datetime(2016, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC)),
    )
    items = data_source.get_items([geom], QueryConfig(space_mode=SpaceMode.INTERSECTS, max_matches=8))[0]
    # Then get the timestamp of the latest NAIP image.
    flat_items = [item for item_group in items for item in item_group]
    flat_items.sort(key=lambda item: item.geometry.time_range[0])
    timestamps.append(flat_items[-1].geometry.time_range[0])
```

Finally, we can create the rslearn windows:

```python
from datetime import timedelta

from rslearn.dataset import Window
from upath import UPath

# Replace with your dataset path.
ds_path = UPath("./dataset")

for (lon, lat), (proj, bounds), ts in zip(lon_lats, proj_bounds, timestamps):
    # Create the window.
    window_group = "default"
    window_name = f"window_{lon}_{lat}"
    window = Window(
        path=Window.get_window_root(ds_path, window_group, window_name),
        group=window_group,
        name=window_name,
        projection=proj,
        bounds=bounds,
        # We create a four-month time range centered at the timestamp of the NAIP image
        # so that we should be able to get enough Sentinel-2 scenes.
        time_range=(
            ts - timedelta(days=60),
            ts + timedelta(days=60),
        ),
    )
    window.save()
```

## Materialize Data

We use rslearn to materialize data. It identifies Sentinel-2 and NAIP satellite images
that match with the spatiotemporal windows we created based on the parameters we
specified in the dataset configuration file.

```
rslearn dataset prepare --root ./dataset --workers 32 --retry-max-attempts 5 --retry-backoff-seconds 5
rslearn dataset materialize --root ./dataset --workers 32 --retry-max-attempts 5 --retry-backoff-seconds 5
```

Now you can visualize these GeoTIFFs in qgis, and/or use the dataset for other
purposes.

```
qgis dataset/windows/default/window_-122.233_47.588/layers/*/R_G_B/geotiff.tif
```
