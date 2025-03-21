import json
import pathlib

import shapely
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.dataset.manage import (
    ingest_dataset_windows,
    materialize_dataset_windows,
    prepare_dataset_windows,
)
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.raster_format import GeotiffRasterFormat


def test_materialize_capitol_hill(tmp_path: pathlib.Path) -> None:
    """Test materializing data from SRTM data source.

    We materialize a portion of Capitol Hill neighborhood in Seattle and make sure
    that it is within the expected range.

    The test verifies that the ingestion pipeline is still working but also that the
    values are being written correctly.
    """

    ds_path = UPath(tmp_path)
    layer_name = "elevation"
    band_name = "elevation"
    dataset_config = {
        "layers": {
            layer_name: {
                "type": "raster",
                "band_sets": [
                    {
                        "dtype": "int32",
                        "bands": [band_name],
                    }
                ],
                "data_source": {
                    "name": "rslearn.data_sources.earthdata_srtm.SRTM",
                },
            },
        },
    }
    with (ds_path / "config.json").open("w") as f:
        json.dump(dataset_config, f)

    # Create a UTM window corresponding to Capitol Hill neighborhood.
    lon, lat = -122.3191, 47.6174
    utm_proj = get_utm_ups_projection(lon, lat, 10, -10)
    src_geometry = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None)
    dst_geometry = src_geometry.to_projection(utm_proj)
    dst_bounds = (
        int(dst_geometry.shp.x) - 4,
        int(dst_geometry.shp.y) - 4,
        int(dst_geometry.shp.x) + 4,
        int(dst_geometry.shp.y) + 4,
    )
    window = Window(
        path=Window.get_window_root(ds_path, "default", "default"),
        group="default",
        name="default",
        projection=utm_proj,
        bounds=dst_bounds,
        time_range=None,
    )
    window.save()

    # Now materialize the windows and verify the value range.
    dataset = Dataset(ds_path)
    windows = dataset.load_windows()
    prepare_dataset_windows(dataset, windows)
    ingest_dataset_windows(dataset, windows)
    materialize_dataset_windows(dataset, windows)

    raster_dir = window.get_raster_dir(layer_name, [band_name])
    array = GeotiffRasterFormat().decode_raster(
        raster_dir, window.projection, window.bounds
    )
    # Roughly 80-120 meters.
    print(f"min={array.min()} max={array.max()}")
    assert array.min() > 80
    assert array.max() < 120
