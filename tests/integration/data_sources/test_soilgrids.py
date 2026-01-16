import pathlib

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from upath import UPath

from rslearn.config import (
    BandSetConfig,
    DType,
    LayerConfig,
    LayerType,
    QueryConfig,
    SpaceMode,
)
from rslearn.data_sources.soilgrids import SOILGRIDS_NODATA_VALUE, SoilGrids
from rslearn.dataset import Window
from rslearn.dataset.storage.file import FileWindowStorage
from rslearn.utils.geometry import Projection


def test_soilgrids_clay_scale_offset_applied(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crs = "urn:ogc:def:crs:EPSG::4326"
    west, south, east, north = (-105.38, 39.45, -104.5, 40.07)
    width, height = (316, 275)

    # Patch the external `soilgrids` package to avoid real network calls, but still
    # validate that rslearn uses the package entrypoint.
    import soilgrids

    def fake_get_coverage_data(self, **kwargs):  # type: ignore[no-untyped-def]
        out = kwargs["output"]
        w = int(kwargs["width"])
        h = int(kwargs["height"])
        transform = from_bounds(
            float(kwargs["west"]),
            float(kwargs["south"]),
            float(kwargs["east"]),
            float(kwargs["north"]),
            w,
            h,
        )
        arr = np.ones((h, w), dtype=np.int16) * 10  # raw value
        profile = dict(
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype="int16",
            crs="EPSG:4326",
            transform=transform,
            nodata=int(SOILGRIDS_NODATA_VALUE),
        )
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(arr, 1)
            dst.update_tags(1, scale_factor=0.1, add_offset=5.0)
            dst.scales = (0.1,)
            dst.offsets = (5.0,)

        return None

    monkeypatch.setattr(
        soilgrids.SoilGrids, "get_coverage_data", fake_get_coverage_data
    )

    # Create a window whose pixel grid matches the requested bbox/width/height.
    xres = (east - west) / width
    yres = (south - north) / height
    projection = Projection(CRS.from_epsg(4326), xres, yres)
    bounds = (
        int(round(west / xres)),
        int(round(north / yres)),
        int(round(west / xres)) + width,
        int(round(north / yres)) + height,
    )
    window = Window(
        storage=FileWindowStorage(UPath(tmp_path)),
        group="default",
        name="bbox",
        projection=projection,
        bounds=bounds,
        time_range=None,
    )
    window.save()

    layer_cfg = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[
            BandSetConfig(
                dtype=DType.FLOAT32,
                bands=["B1"],
                nodata_vals=[SOILGRIDS_NODATA_VALUE],
            )
        ],
    )

    data_source = SoilGrids(
        service_id="clay",
        coverage_id="clay_0-5cm_mean",
        crs=crs,
        width=width,
        height=height,
    )
    item_groups = data_source.get_items(
        [window.get_geometry()], QueryConfig(space_mode=SpaceMode.INTERSECTS)
    )[0]
    data_source.materialize(window, item_groups, "clay", layer_cfg)

    out_path = window.get_raster_dir("clay", ["B1"]) / "geotiff.tif"
    with rasterio.open(out_path) as src:
        out = src.read(1)

    # Expected: 10 * 0.1 + 5.0 == 6.0
    assert float(out.max()) == pytest.approx(6.0)
