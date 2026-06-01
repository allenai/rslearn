"""Unit tests for the Copernicus GLO-30 data source."""

import numpy as np
import pytest
import shapely

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import Item
from rslearn.data_sources.copernicus_glo30 import (
    CopernicusGLO30,
    _tile_name,
    _tile_url,
    compute_terrain,
)
from rslearn.data_sources.utils import MatchedItemGroup
from rslearn.utils.geometry import STGeometry


class TestTileNaming:
    def test_tile_name_north_east(self) -> None:
        assert _tile_name(47, 10) == "Copernicus_DSM_COG_10_N47_00_E010_00_DEM"

    def test_tile_name_south_west(self) -> None:
        assert _tile_name(-3, -123) == "Copernicus_DSM_COG_10_S03_00_W123_00_DEM"

    def test_tile_name_zero(self) -> None:
        assert _tile_name(0, 0) == "Copernicus_DSM_COG_10_N00_00_E000_00_DEM"

    def test_tile_url(self) -> None:
        url = _tile_url(47, -123)
        assert url.startswith(
            "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/"
        )
        assert url.endswith(
            "Copernicus_DSM_COG_10_N47_00_W123_00_DEM/"
            "Copernicus_DSM_COG_10_N47_00_W123_00_DEM.tif"
        )


class TestParseTileName:
    def test_roundtrip_north_east(self) -> None:
        name = _tile_name(47, 10)
        lat, lon = CopernicusGLO30._parse_tile_name(name)
        assert lat == 47
        assert lon == 10

    def test_roundtrip_south_west(self) -> None:
        name = _tile_name(-3, -123)
        lat, lon = CopernicusGLO30._parse_tile_name(name)
        assert lat == -3
        assert lon == -123

    def test_roundtrip_zero(self) -> None:
        name = _tile_name(0, 0)
        lat, lon = CopernicusGLO30._parse_tile_name(name)
        assert lat == 0
        assert lon == 0


class TestGetItems:
    def test_rejects_non_mosaic(self) -> None:
        ds = CopernicusGLO30()
        geom = STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 1, 1), None)
        with pytest.raises(ValueError, match="mosaic"):
            ds.get_items(
                [geom],
                QueryConfig(space_mode=SpaceMode.INTERSECTS, max_matches=1),
            )

    def test_rejects_min_matches(self) -> None:
        ds = CopernicusGLO30()
        geom = STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 1, 1), None)
        with pytest.raises(ValueError, match="min_matches"):
            ds.get_items(
                [geom],
                QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1, min_matches=1),
            )

    def test_single_tile(self) -> None:
        ds = CopernicusGLO30()
        geom = STGeometry(WGS84_PROJECTION, shapely.box(10.2, 47.3, 10.8, 47.7), None)
        groups = ds.get_items(
            [geom], QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1)
        )
        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert isinstance(groups[0][0], MatchedItemGroup)
        assert len(groups[0][0].items) == 1
        assert groups[0][0].items[0].name == _tile_name(47, 10)

    def test_multiple_tiles(self) -> None:
        """A bbox spanning 2x2 degree cells should return 4 items."""
        ds = CopernicusGLO30()
        geom = STGeometry(WGS84_PROJECTION, shapely.box(-0.5, -0.5, 0.5, 0.5), None)
        groups = ds.get_items(
            [geom], QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1)
        )
        items = groups[0][0].items
        assert len(items) == 4
        names = {item.name for item in items}
        expected = {
            _tile_name(-1, -1),
            _tile_name(-1, 0),
            _tile_name(0, -1),
            _tile_name(0, 0),
        }
        assert names == expected

    def test_no_time_range_on_items(self) -> None:
        ds = CopernicusGLO30()
        geom = STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 0.5, 0.5), None)
        groups = ds.get_items(
            [geom], QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1)
        )
        item = groups[0][0].items[0]
        assert item.geometry.time_range is None


class TestDeserializeItem:
    def test_roundtrip(self) -> None:
        ds = CopernicusGLO30()
        original = Item(
            _tile_name(47, -123),
            STGeometry(WGS84_PROJECTION, shapely.box(-123, 47, -122, 48), None),
        )
        serialized = original.serialize()
        restored = ds.deserialize_item(serialized)
        assert restored.name == original.name


class TestComputeTerrain:
    def test_flat_surface(self) -> None:
        """A constant-elevation surface should have zero slope and -1 aspect."""
        dem = np.full((10, 10), 100.0, dtype=np.float32)
        slope, aspect = compute_terrain(dem, pixel_size_deg=1 / 3600, lat_south=45.0)
        np.testing.assert_allclose(slope, 0.0, atol=1e-5)
        np.testing.assert_allclose(aspect, -1.0)

    def test_north_facing_slope(self) -> None:
        """Elevation increasing southward => slope faces north (aspect ~0 or ~360)."""
        dem = np.zeros((20, 20), dtype=np.float32)
        for row in range(20):
            dem[row, :] = row * 10.0  # increases toward south (row 0 = north)
        slope, aspect = compute_terrain(dem, pixel_size_deg=1 / 3600, lat_south=45.0)

        # Interior pixels should have nonzero slope.
        assert slope[10, 10] > 0.0

        # Aspect should be near 0° (north) for interior pixels, since steepest
        # descent goes from south (high) to north (low).
        interior_aspect = aspect[5:15, 5:15]
        valid = interior_aspect >= 0
        assert valid.any()
        # Allow wrapping around 0/360.
        mean_aspect = np.mean(interior_aspect[valid])
        assert mean_aspect < 30.0 or mean_aspect > 330.0

    def test_east_facing_slope(self) -> None:
        """Elevation increasing westward => slope faces east (aspect ~90)."""
        dem = np.zeros((20, 20), dtype=np.float32)
        for col in range(20):
            dem[:, col] = (19 - col) * 10.0  # decreases toward east
        slope, aspect = compute_terrain(dem, pixel_size_deg=1 / 3600, lat_south=45.0)
        assert slope[10, 10] > 0.0
        interior_aspect = aspect[5:15, 5:15]
        valid = interior_aspect >= 0
        mean_aspect = np.mean(interior_aspect[valid])
        assert 60.0 < mean_aspect < 120.0

    def test_nodata_propagation(self) -> None:
        """Nodata pixels should produce NaN in both outputs."""
        dem = np.full((10, 10), 100.0, dtype=np.float32)
        dem[5, 5] = -9999.0
        slope, aspect = compute_terrain(
            dem, pixel_size_deg=1 / 3600, lat_south=0.0, nodata=-9999.0
        )
        assert np.isnan(slope[5, 5])
        assert np.isnan(aspect[5, 5])

    def test_slope_range(self) -> None:
        """Slope should be in [0, 90)."""
        rng = np.random.RandomState(42)
        dem = rng.randn(50, 50).astype(np.float32) * 500 + 1000
        slope, _ = compute_terrain(dem, pixel_size_deg=1 / 3600, lat_south=30.0)
        assert np.all(slope[np.isfinite(slope)] >= 0)
        assert np.all(slope[np.isfinite(slope)] < 90)

    def test_aspect_range(self) -> None:
        """Aspect should be in [0, 360) or -1."""
        rng = np.random.RandomState(42)
        dem = rng.randn(50, 50).astype(np.float32) * 500 + 1000
        _, aspect = compute_terrain(dem, pixel_size_deg=1 / 3600, lat_south=30.0)
        valid = aspect[np.isfinite(aspect)]
        positive = valid[valid >= 0]
        assert np.all(positive >= 0)
        assert np.all(positive < 360)
        negative = valid[valid < 0]
        np.testing.assert_allclose(negative, -1.0)
