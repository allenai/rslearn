"""Unit tests for the Copernicus GLO-30 data source."""

import io
import pathlib
from unittest.mock import MagicMock

import boto3
import pytest
import shapely

from rslearn.config import (
    BandSetConfig,
    DType,
    LayerConfig,
    LayerType,
    QueryConfig,
    SpaceMode,
)
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSourceContext
from rslearn.data_sources.aws_glo30 import (
    DATA_ASSET,
    CopernicusGLO30,
    _tile_name,
    _tile_url,
)
from rslearn.data_sources.utils import MatchedItemGroup
from rslearn.utils.geometry import STGeometry

# Tiles used across the get_items tests.
LAND_TILES = [
    _tile_name(47, 10),
    _tile_name(0, 0),
    _tile_name(0, -1),
    _tile_name(-1, 0),
    _tile_name(-1, -1),
    _tile_name(47, -123),
]


def _mock_boto3(monkeypatch: pytest.MonkeyPatch, tile_names: list[str]) -> None:
    """Patch boto3 so the tileList.txt download returns the given tile names."""
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {
        "Body": io.BytesIO(("\n".join(tile_names) + "\n").encode())
    }
    monkeypatch.setattr(boto3, "client", lambda *a, **kw: mock_s3)


@pytest.fixture
def data_source(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> CopernicusGLO30:
    _mock_boto3(monkeypatch, LAND_TILES)
    return CopernicusGLO30(metadata_cache_dir=str(tmp_path / "cache"))


class TestTileNames:
    @pytest.mark.parametrize(
        "lat,lon,expected",
        [
            (47, 10, "Copernicus_DSM_COG_10_N47_00_E010_00_DEM"),
            (-3, -123, "Copernicus_DSM_COG_10_S03_00_W123_00_DEM"),
            (0, 0, "Copernicus_DSM_COG_10_N00_00_E000_00_DEM"),
        ],
    )
    def test_name_and_parse_roundtrip(self, lat: int, lon: int, expected: str) -> None:
        assert _tile_name(lat, lon) == expected
        assert CopernicusGLO30._parse_tile_name(expected) == (lat, lon)

    def test_tile_url(self) -> None:
        assert _tile_url(47, -123) == (
            "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/"
            "Copernicus_DSM_COG_10_N47_00_W123_00_DEM/"
            "Copernicus_DSM_COG_10_N47_00_W123_00_DEM.tif"
        )

    @pytest.mark.parametrize(
        "name",
        [
            "not_a_tile",
            "Copernicus_DSM_COG_10_X47_00_W123_00_DEM",
            "Copernicus_DSM_COG_10_N47_00_Z123_00_DEM",
            "Copernicus_DSM_COG_10_NXX_00_W123_00_DEM",
        ],
    )
    def test_parse_rejects_invalid_names(self, name: str) -> None:
        with pytest.raises(ValueError, match="invalid GLO-30 tile name"):
            CopernicusGLO30._parse_tile_name(name)


class TestInit:
    def test_rejects_multiple_band_sets(self, tmp_path: pathlib.Path) -> None:
        layer_config = LayerConfig(
            type=LayerType.RASTER,
            band_sets=[
                BandSetConfig(dtype=DType.FLOAT32, bands=["elevation"]),
                BandSetConfig(dtype=DType.FLOAT32, bands=["slope"]),
            ],
        )
        with pytest.raises(ValueError, match="single band set"):
            CopernicusGLO30(
                metadata_cache_dir=str(tmp_path / "cache"),
                context=DataSourceContext(layer_config=layer_config),
            )

    def test_rejects_multiple_bands(self, tmp_path: pathlib.Path) -> None:
        """Slope/aspect come from a transform, so only one band is allowed here."""
        layer_config = LayerConfig(
            type=LayerType.RASTER,
            band_sets=[
                BandSetConfig(dtype=DType.FLOAT32, bands=["elevation", "slope"]),
            ],
        )
        with pytest.raises(ValueError, match="single band"):
            CopernicusGLO30(
                metadata_cache_dir=str(tmp_path / "cache"),
                context=DataSourceContext(layer_config=layer_config),
            )

    def test_uses_band_name_from_layer_config(self, tmp_path: pathlib.Path) -> None:
        layer_config = LayerConfig(
            type=LayerType.RASTER,
            band_sets=[BandSetConfig(dtype=DType.FLOAT32, bands=["dem"])],
        )
        data_source = CopernicusGLO30(
            metadata_cache_dir=str(tmp_path / "cache"),
            context=DataSourceContext(layer_config=layer_config),
        )
        assert data_source.band_name == "dem"
        assert data_source.asset_bands == {DATA_ASSET: ["dem"]}


class TestGetItems:
    def test_rejects_non_mosaic(self, data_source: CopernicusGLO30) -> None:
        geom = STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 1, 1), None)
        with pytest.raises(ValueError, match="mosaic"):
            data_source.get_items(
                [geom],
                QueryConfig(space_mode=SpaceMode.INTERSECTS, max_matches=1),
            )

    def test_rejects_min_matches(self, data_source: CopernicusGLO30) -> None:
        geom = STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 1, 1), None)
        with pytest.raises(ValueError, match="min_matches"):
            data_source.get_items(
                [geom],
                QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1, min_matches=1),
            )

    def test_single_tile(self, data_source: CopernicusGLO30) -> None:
        geom = STGeometry(WGS84_PROJECTION, shapely.box(10.2, 47.3, 10.8, 47.7), None)
        groups = data_source.get_items(
            [geom], QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1)
        )
        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert isinstance(groups[0][0], MatchedItemGroup)
        item = groups[0][0].items[0]
        assert item.name == _tile_name(47, 10)
        # The DEM is static, so items carry no time range.
        assert item.geometry.time_range is None

    def test_multiple_tiles(self, data_source: CopernicusGLO30) -> None:
        """A bbox spanning 2x2 degree cells should return 4 items."""
        geom = STGeometry(WGS84_PROJECTION, shapely.box(-0.5, -0.5, 0.5, 0.5), None)
        groups = data_source.get_items(
            [geom], QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1)
        )
        items = groups[0][0].items
        assert {item.name for item in items} == {
            _tile_name(-1, -1),
            _tile_name(-1, 0),
            _tile_name(0, -1),
            _tile_name(0, 0),
        }

    def test_skips_tiles_missing_from_index(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ocean cells absent from tileList.txt should not produce items."""
        _mock_boto3(monkeypatch, [_tile_name(0, 0)])
        data_source = CopernicusGLO30(metadata_cache_dir=str(tmp_path / "cache"))
        geom = STGeometry(WGS84_PROJECTION, shapely.box(-0.5, -0.5, 0.5, 0.5), None)
        groups = data_source.get_items(
            [geom], QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1)
        )
        items = groups[0][0].items
        assert [item.name for item in items] == [_tile_name(0, 0)]


class TestAssetUrl:
    def test_get_asset_url(self, data_source: CopernicusGLO30) -> None:
        item = data_source.get_item_by_name(_tile_name(47, -123))
        assert data_source.get_asset_url(item, DATA_ASSET) == _tile_url(47, -123)

    def test_rejects_unknown_asset_key(self, data_source: CopernicusGLO30) -> None:
        item = data_source.get_item_by_name(_tile_name(47, -123))
        with pytest.raises(ValueError, match="Unknown asset key"):
            data_source.get_asset_url(item, "nope")
