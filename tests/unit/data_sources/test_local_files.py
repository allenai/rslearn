import pathlib

import numpy as np
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from upath import UPath

from rslearn.data_sources.local_files import RasterImporter


class TestRasterImporterGCPGeometry:
    """Verify that the item geometry is computed correctly for rasters with GCPs."""

    def test_interior_gcps(self, tmp_path: pathlib.Path) -> None:
        """Test with GCPs in the interior of the image.

        Although the GCPs are in the image interior, the item geometry should still
        reflect the full image extent.
        """
        src_dir = UPath(tmp_path)
        width, height = 8, 8
        gcp_crs = CRS.from_epsg(4326)

        # GCPs at interior pixels — NOT image corners.
        # The underlying transform is x = col + 10, y = 28 - row, so the
        # full image extent should be box(10, 20, 18, 28).
        gcps = [
            GroundControlPoint(row=2, col=2, x=12.0, y=26.0),
            GroundControlPoint(row=2, col=6, x=16.0, y=26.0),
            GroundControlPoint(row=6, col=2, x=12.0, y=22.0),
            GroundControlPoint(row=6, col=6, x=16.0, y=22.0),
        ]
        data = np.ones((1, height, width), dtype=np.uint8)
        tif_path = src_dir / "image.tif"
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype="uint8",
        ) as dst:
            dst.gcps = (gcps, gcp_crs)
            dst.write(data)

        importer = RasterImporter()
        items = importer.list_items(src_dir)
        assert len(items) == 1

        bounds = items[0].geometry.shp.bounds
        assert bounds == (10.0, 20.0, 18.0, 28.0)

    def test_corner_gcps_produce_same_extent(self, tmp_path: pathlib.Path) -> None:
        """Test with GCPs at the image corners."""
        src_dir = UPath(tmp_path)
        width, height = 8, 8
        gcp_crs = CRS.from_epsg(4326)

        gcps = [
            GroundControlPoint(row=0, col=0, x=10.0, y=28.0),
            GroundControlPoint(row=0, col=width, x=18.0, y=28.0),
            GroundControlPoint(row=height, col=0, x=10.0, y=20.0),
            GroundControlPoint(row=height, col=width, x=18.0, y=20.0),
        ]
        data = np.ones((1, height, width), dtype=np.uint8)
        tif_path = src_dir / "image.tif"
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype="uint8",
        ) as dst:
            dst.gcps = (gcps, gcp_crs)
            dst.write(data)

        importer = RasterImporter()
        items = importer.list_items(src_dir)
        assert len(items) == 1

        bounds = items[0].geometry.shp.bounds
        assert bounds == (10.0, 20.0, 18.0, 28.0)
