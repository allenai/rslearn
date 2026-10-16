"""Integration tests for the BOKU European soil texture data source."""

import pathlib

import numpy as np
import pytest
import rasterio
import shapely
from affine import Affine
from rasterio import CRS
from rasterio.enums import Resampling

from rslearn.config import (
    BandSetConfig,
    DType,
    LayerConfig,
    LayerType,
    QueryConfig,
    SpaceMode,
)
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import DataSourceContext, Item
from rslearn.data_sources.european_soil_properties import (
    PERCENT_SCALE,
    TEXTURE_BANDS,
    EuropeanSoilProperties,
)
from rslearn.utils.geometry import Projection, STGeometry


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
            [[102, 0], [0, 75]],
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


def test_get_items_and_lookup() -> None:
    data_source = EuropeanSoilProperties()
    groups = data_source.get_items(
        [_vienna_geometry()], QueryConfig(space_mode=SpaceMode.INTERSECTS)
    )[0]

    assert len(groups) == 1
    assert [item.name for item in groups[0].items] == ["Tile_69"]
    assert data_source.get_item_by_name("Tile_69").name == "Tile_69"
    with pytest.raises(ValueError, match="not found"):
        data_source.get_item_by_name("Tile_123")


def test_get_asset_url() -> None:
    data_source = EuropeanSoilProperties()
    item = data_source.get_item_by_name("Tile_69")
    assert data_source.get_asset_url(item, "textures").endswith(
        "/9KQqb8pxPrArTta/Textures/Tile_69.tif"
    )
    with pytest.raises(ValueError, match="unknown asset key"):
        data_source.get_asset_url(item, "unknown")


def test_requires_float32_texture_band_set() -> None:
    wrong_dtype = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.UINT8, bands=TEXTURE_BANDS)],
    )
    with pytest.raises(ValueError, match="dtype=float32"):
        EuropeanSoilProperties(context=DataSourceContext(layer_config=wrong_dtype))

    wrong_bands = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.FLOAT32, bands=["clay"])],
    )
    with pytest.raises(ValueError, match="requires one band set"):
        EuropeanSoilProperties(context=DataSourceContext(layer_config=wrong_bands))


def test_read_raster_scales_and_masks_empty_pairs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geotiff_path = _write_texture_geotiff(tmp_path / "textures.tif")
    data_source = EuropeanSoilProperties()
    item = Item(
        "test",
        STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 2, 2), None),
    )
    monkeypatch.setattr(
        data_source,
        "get_asset_url",
        lambda item, asset_key: str(geotiff_path),
    )

    raster = data_source.read_raster(
        layer_name="soil_texture",
        item=item,
        bands=TEXTURE_BANDS,
        projection=Projection(CRS.from_epsg(4326), 1, -1),
        bounds=(0, -2, 2, 0),
        resampling=Resampling.nearest,
    )
    array = raster.get_chw_array()

    assert array.shape == (3, 2, 2)
    np.testing.assert_allclose(array[:, 0, 0], np.array([51, 102, 102]) * PERCENT_SCALE)
    assert np.isnan(array[:, 0, 1]).all()
    # A zero texture value with non-zero uncertainty is valid, not nodata.
    np.testing.assert_allclose(array[:, 1, 0], [0, 0, 100])
    np.testing.assert_allclose(array[:, 1, 1].sum(), 100)
    assert np.isnan(raster.metadata.nodata_value)


def test_rejects_partial_band_request() -> None:
    data_source = EuropeanSoilProperties()
    item = data_source.get_item_by_name("Tile_69")
    with pytest.raises(ValueError, match="expected bands"):
        data_source.get_raster_metadata("soil_texture", item, ["clay"])
