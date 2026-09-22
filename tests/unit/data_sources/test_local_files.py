import json
import pathlib
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from upath import UPath

from rslearn.config import LayerType
from rslearn.data_sources import local_files
from rslearn.data_sources.local_files import (
    LocalFiles,
    RasterImporter,
    VectorImporter,
)


def _make_raster(src_dir: UPath, fname: str = "image.tif") -> None:
    """Write a small georeferenced GeoTIFF into the source directory."""
    with rasterio.open(
        src_dir / fname,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=1,
        dtype="uint8",
        crs=CRS.from_epsg(4326),
        transform=rasterio.transform.from_origin(10, 28, 1, 1),
    ) as dst:
        dst.write(np.ones((1, 8, 8), dtype=np.uint8))


def _make_data_source(src_dir: UPath) -> LocalFiles:
    """Create a raster LocalFiles data source over the source directory."""
    return LocalFiles(src_dir=str(src_dir), layer_type=LayerType.RASTER)


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


class TestItemListCache:
    """Verify the item list cache is robust to concurrent prepare workers.

    Several workers can call list_items on the same source directory at once, so a
    worker must never observe a cache file that another worker has created but not
    finished writing.
    """

    def test_cache_file_appears_only_once_complete(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cache file must not be visible while it is still being written."""
        src_dir = UPath(tmp_path)
        _make_raster(src_dir)

        cache_fname = src_dir / "summary.json"
        existed_during_write = None
        real_dump = json.dump

        def spy_dump(obj: Any, fp: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal existed_during_write
            existed_during_write = cache_fname.exists()
            return real_dump(obj, fp, *args, **kwargs)

        monkeypatch.setattr(local_files.json, "dump", spy_dump)

        items = _make_data_source(src_dir).list_items()

        assert len(items) == 1
        assert existed_during_write is False
        assert json.loads(cache_fname.read_text())

    def test_truncated_cache_is_relisted(self, tmp_path: pathlib.Path) -> None:
        """A partially written cache is discarded rather than raising."""
        src_dir = UPath(tmp_path)
        _make_raster(src_dir)
        cache_fname = src_dir / "summary.json"
        cache_fname.write_text('[{"name": "imag')

        items = _make_data_source(src_dir).list_items()

        assert [item.name for item in items] == ["image"]
        # The re-listed items should have replaced the truncated cache.
        assert len(json.loads(cache_fname.read_text())) == 1

    def test_empty_cache_is_relisted(self, tmp_path: pathlib.Path) -> None:
        """An empty cache, as left by an interrupted write, is discarded."""
        src_dir = UPath(tmp_path)
        _make_raster(src_dir)
        (src_dir / "summary.json").write_text("")

        items = _make_data_source(src_dir).list_items()

        assert [item.name for item in items] == ["image"]

    def test_valid_cache_is_used(self, tmp_path: pathlib.Path) -> None:
        """A readable cache is trusted and the source directory is not listed again."""
        src_dir = UPath(tmp_path)
        _make_raster(src_dir)
        data_source = _make_data_source(src_dir)
        cached_items = data_source.list_items()

        # A second data source over the same directory reads what the first one wrote.
        # Removing the raster proves the items came from the cache.
        (src_dir / "image.tif").unlink()
        items = _make_data_source(src_dir).list_items()

        assert [item.name for item in items] == [item.name for item in cached_items]


class TestImporterFileFiltering:
    """Verify which files in the source directory are treated as data files.

    Both importers must ignore the item list cache along with the temporary file
    another worker writes while caching its own item list, which is not readable.
    """

    def test_raster_importer_ignores_cache_files(self, tmp_path: pathlib.Path) -> None:
        """The raster importer must not try to read a partially written cache file."""
        src_dir = UPath(tmp_path)
        _make_raster(src_dir)
        (src_dir / "summary.json").write_text("[]")
        (src_dir / "summary.json.tmp.1234").write_text('[{"name": "imag')

        items = RasterImporter().list_items(src_dir)

        assert [item.name for item in items] == ["image"]

    def test_vector_importer_ignores_cache_files(self, tmp_path: pathlib.Path) -> None:
        """The vector importer must not try to read a partially written cache file."""
        src_dir = UPath(tmp_path)
        (src_dir / "data.geojson").write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [5, 5]},
                            "properties": {},
                        }
                    ],
                }
            )
        )
        (src_dir / "summary.json").write_text("[]")
        (src_dir / "summary.json.tmp.1234").write_text('[{"name": "dat')

        items = VectorImporter().list_items(src_dir)

        assert [item.name for item in items] == ["data"]
