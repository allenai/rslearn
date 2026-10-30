"""Integration tests for the BOKU European soil texture data source."""

import json
import pathlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
import shapely
from affine import Affine
from rasterio import CRS
from rasterio.enums import Resampling
from upath import UPath

from rslearn.config import (
    BandSetConfig,
    DType,
    LayerConfig,
    LayerType,
    QueryConfig,
    SpaceMode,
)
from rslearn.config.dataset import ResamplingMethod
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import DataSourceContext, Item
from rslearn.data_sources.european_soil_properties import (
    BASE_URL,
    TEXTURE_BANDS,
    TILE_COUNT,
    EuropeanSoilProperties,
)
from rslearn.utils.geometry import Projection, STGeometry


@pytest.fixture
def cache_dir(tmp_path: pathlib.Path) -> str:
    cache = tmp_path / "cache"
    cache.mkdir()
    item = Item(
        "Tile_69",
        STGeometry(
            WGS84_PROJECTION,
            shapely.box(13.5218010, 47.1270571, 16.5219513, 50.1272074),
            None,
        ),
    )
    (cache / "tile_index.json").write_text(json.dumps([item.serialize()]))
    return str(cache)


def _vienna_geometry() -> STGeometry:
    return STGeometry(
        WGS84_PROJECTION,
        shapely.box(16.0, 48.0, 16.1, 48.1),
        None,
    )


def _write_texture_geotiff(path: pathlib.Path) -> pathlib.Path:
    values = np.array(
        [
            [[51, 0], [0, 25]],
            [[101, 0], [0, 75]],
            [[102, 0], [255, 155]],
        ],
        dtype=np.uint8,
    )
    uncertainty = np.array(
        [
            [[1, 0], [1, 1]],
            [[1, 0], [1, 1]],
            [[1, 0], [1, 1]],
        ],
        dtype=np.uint8,
    )
    data = np.concatenate([values, uncertainty])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=6,
        dtype=np.uint8,
        crs=CRS.from_epsg(4326),
        transform=Affine(1, 0, 0, 0, -1, 2),
    ) as dst:
        dst.write(data)
    return path


def test_get_items_and_lookup(cache_dir: str) -> None:
    data_source = EuropeanSoilProperties(cache_dir=cache_dir)
    groups = data_source.get_items(
        [_vienna_geometry()], QueryConfig(space_mode=SpaceMode.INTERSECTS)
    )[0]

    assert len(groups) == 1
    assert [item.name for item in groups[0].items] == ["Tile_69"]
    assert data_source.get_item_by_name("Tile_69").name == "Tile_69"
    with pytest.raises(ValueError, match="not found"):
        data_source.get_item_by_name("Tile_123")


def test_get_asset_url(cache_dir: str) -> None:
    data_source = EuropeanSoilProperties(cache_dir=cache_dir)
    item = data_source.get_item_by_name("Tile_69")
    assert data_source.get_asset_url(item, "textures").endswith(
        "/9KQqb8pxPrArTta/Textures/Tile_69.tif"
    )
    with pytest.raises(ValueError, match="unknown asset key"):
        data_source.get_asset_url(item, "unknown")


def test_requires_float32_texture_band_set(cache_dir: str) -> None:
    wrong_dtype = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.UINT8, bands=TEXTURE_BANDS)],
    )
    with pytest.raises(ValueError, match="dtype=float32"):
        EuropeanSoilProperties(
            cache_dir=cache_dir, context=DataSourceContext(layer_config=wrong_dtype)
        )

    wrong_bands = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.FLOAT32, bands=["clay"])],
    )
    with pytest.raises(ValueError, match="requires one band set"):
        EuropeanSoilProperties(
            cache_dir=cache_dir, context=DataSourceContext(layer_config=wrong_bands)
        )


@pytest.mark.parametrize("resolution", [1, 0.5, 2])
def test_read_raster_preserves_source_values_and_masks_empty_pairs(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    cache_dir: str,
    resolution: float,
) -> None:
    geotiff_path = _write_texture_geotiff(tmp_path / "textures.tif")
    data_source = EuropeanSoilProperties(cache_dir=cache_dir)
    item = Item(
        "test",
        STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 2, 2), None),
    )
    monkeypatch.setattr(
        data_source,
        "get_asset_url",
        lambda item, asset_key: str(geotiff_path),
    )

    size = int(2 / resolution)
    raster = data_source.read_raster(
        layer_name="soil_texture",
        item=item,
        bands=TEXTURE_BANDS,
        projection=Projection(CRS.from_epsg(4326), resolution, -resolution),
        bounds=(0, -size, size, 0),
    )
    array = raster.get_chw_array()

    expected = np.array(
        [
            [[51, np.nan], [0, 25]],
            [[101, np.nan], [0, 75]],
            [[102, np.nan], [255, 155]],
        ],
        dtype=np.float32,
    )
    if resolution == 2:
        # The coarser pixel selects the bottom-right sample, without averaging
        # the empty top-right pixel or the other valid compositions into it.
        expected = expected[:, 1:, 1:]
    else:
        repeats = int(1 / resolution)
        expected = expected.repeat(repeats, axis=1).repeat(repeats, axis=2)
    assert array.shape == (3, size, size)
    # Resizing must preserve sampled compositions without blending.
    assert array.dtype == np.float32
    # The first valid pixel sums to 254: do not normalize it to 255 or 100.
    np.testing.assert_array_equal(array, expected)
    assert np.isnan(
        data_source.get_raster_metadata(
            "soil_texture", item, TEXTURE_BANDS
        ).nodata_value
    )
    assert np.isnan(raster.metadata.nodata_value)


def test_rejects_partial_band_request(cache_dir: str) -> None:
    data_source = EuropeanSoilProperties(cache_dir=cache_dir)
    item = data_source.get_item_by_name("Tile_69")
    with pytest.raises(ValueError, match="expected bands"):
        data_source.get_raster_metadata("soil_texture", item, ["clay"])


def test_discovers_exact_bounds_and_reuses_cache(tmp_path: pathlib.Path) -> None:
    cache = tmp_path / "cache"
    bounds = (-10.478219, 32.1270783, -7.4780687, 35.1272286)
    src = MagicMock()
    src.__enter__.return_value = src
    src.crs = CRS.from_epsg(4326)
    src.bounds = bounds
    with patch("rasterio.open", return_value=src) as open_cog:
        source = EuropeanSoilProperties(
            cache_dir="cache", context=DataSourceContext(ds_path=UPath(tmp_path))
        )
        assert open_cog.call_count == TILE_COUNT
        assert [call.args[0] for call in open_cog.call_args_list] == [
            f"{BASE_URL}/Tile_{i}.tif" for i in range(1, TILE_COUNT + 1)
        ]
        src.read.assert_not_called()
    assert source.get_item_by_name("Tile_1").geometry.shp.bounds == bounds
    # A query in the non-ideal edge strip must still find the tile.
    edge = STGeometry(WGS84_PROJECTION, shapely.box(-7.4781, 33, -7.47808, 33.01), None)
    groups = source.get_items([edge], QueryConfig(space_mode=SpaceMode.INTERSECTS))[0]
    assert groups
    assert groups[0].items[0].geometry.shp.bounds == bounds
    with patch(
        "rasterio.open", side_effect=AssertionError("unexpected network request")
    ):
        cached = EuropeanSoilProperties(cache_dir=str(cache))
    for i in range(1, TILE_COUNT + 1):
        name = f"Tile_{i}"
        assert (
            cached.get_item_by_name(name).serialize()
            == source.get_item_by_name(name).serialize()
        )


def test_failed_discovery_does_not_cache_partial_index(tmp_path: pathlib.Path) -> None:
    src = MagicMock()
    src.__enter__.return_value = src
    src.crs = CRS.from_epsg(4326)
    src.bounds = (0, 0, 3.0001, 3.0001)
    with (
        patch(
            "rasterio.open",
            side_effect=[src, rasterio.errors.RasterioIOError("unavailable")],
        ),
        pytest.raises(rasterio.errors.RasterioIOError, match="unavailable"),
    ):
        EuropeanSoilProperties(cache_dir=str(tmp_path))
    assert not (tmp_path / "tile_index.json").exists()
    with patch("rasterio.open", return_value=src) as open_cog:
        EuropeanSoilProperties(cache_dir=str(tmp_path))
        assert open_cog.call_count == TILE_COUNT


def test_instantiates_from_layer_config(cache_dir: str) -> None:
    config = LayerConfig.model_validate(
        {
            "type": "raster",
            "resampling_method": "nearest",
            "band_sets": [{"dtype": "float32", "bands": TEXTURE_BANDS}],
            "data_source": {
                "name": "rslearn.data_sources.european_soil_properties.EuropeanSoilProperties",
                "ingest": False,
                "cache_dir": cache_dir,
            },
        }
    )
    source = config.instantiate_data_source()
    assert isinstance(source, EuropeanSoilProperties)
    assert source.get_item_by_name("Tile_69").name == "Tile_69"


@pytest.mark.parametrize(
    "resampling",
    [
        None,
        ResamplingMethod.BILINEAR,
        ResamplingMethod.CUBIC,
        ResamplingMethod.CUBIC_SPLINE,
    ],
)
def test_rejects_non_nearest_layer_config(
    cache_dir: str, resampling: ResamplingMethod | None
) -> None:
    config = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.FLOAT32, bands=TEXTURE_BANDS)],
    )
    if resampling is not None:
        config = config.model_copy(update={"resampling_method": resampling})
    with (
        patch("rasterio.open") as open_cog,
        pytest.raises(ValueError, match="resampling_method=nearest"),
    ):
        EuropeanSoilProperties(
            cache_dir=cache_dir, context=DataSourceContext(layer_config=config)
        )
    open_cog.assert_not_called()


@pytest.mark.parametrize(
    "resampling", [Resampling.bilinear, Resampling.cubic, Resampling.average]
)
def test_rejects_non_nearest_read(cache_dir: str, resampling: Resampling) -> None:
    source = EuropeanSoilProperties(cache_dir=cache_dir)
    with (
        patch("rasterio.open") as open_cog,
        pytest.raises(ValueError, match="nearest-neighbour"),
    ):
        source.read_raster(
            layer_name="soil_texture",
            item=source.get_item_by_name("Tile_69"),
            bands=TEXTURE_BANDS,
            projection=Projection(CRS.from_epsg(4326), 1, -1),
            bounds=(0, -2, 2, 0),
            resampling=resampling,
        )
    open_cog.assert_not_called()
