from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

pytest.importorskip("earthdaily")

from datetime import UTC, datetime

import shapely

from rslearn.config import BandSetConfig, DType, LayerConfig, LayerType
from rslearn.data_sources import DataSourceContext
from rslearn.data_sources.earthdaily import EarthDailyItem, Sentinel2
from rslearn.utils.geometry import Projection, STGeometry


class _FakeSentinel2(Sentinel2):
    def __init__(self, item: EarthDailyItem, *, apply_scale_offset: bool = True):
        super().__init__(
            apply_scale_offset=apply_scale_offset, assets=["red"], cache_dir=None
        )
        self._item = item

    def get_item_by_name(self, name: str) -> EarthDailyItem:
        return self._item


def test_read_raster_applies_asset_scale_offset(tmp_path: Path) -> None:
    tif_path = tmp_path / "red.tif"

    crs = CRS.from_epsg(3857)
    transform = Affine(1, 0, 0, 0, -1, 0)
    raw = np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16)
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype=str(raw.dtype),
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(raw, 1)

    geom = STGeometry(
        Projection(crs, 1, -1),
        shapely.box(0, 0, 2, 2),
        (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)),
    )
    item = EarthDailyItem(
        name="item1",
        geometry=geom,
        asset_urls={"red": str(tif_path)},
        asset_scale_offsets={"red": [{"scale": 0.001, "offset": 1.0, "nodata": 0.0}]},
    )
    ds = _FakeSentinel2(item)

    projection = Projection(crs, 1, -1)
    out = ds.read_raster(
        layer_name="layer",
        item=item,
        bands=["B04"],
        projection=projection,
        bounds=(0, 0, 2, 2),
    )

    arr = out.get_chw_array()
    assert arr.dtype == np.float32
    np.testing.assert_allclose(arr[0], raw.astype(np.float32) * 0.001 + 1.0)


def test_read_raster_no_apply_scale_offset_returns_raw(tmp_path: Path) -> None:
    tif_path = tmp_path / "red.tif"

    crs = CRS.from_epsg(3857)
    transform = Affine(1, 0, 0, 0, -1, 0)
    raw = np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16)
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype=str(raw.dtype),
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(raw, 1)

    geom = STGeometry(
        Projection(crs, 1, -1),
        shapely.box(0, 0, 2, 2),
        (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)),
    )
    item = EarthDailyItem(
        name="item1",
        geometry=geom,
        asset_urls={"red": str(tif_path)},
        asset_scale_offsets={"red": [{"scale": 0.001, "offset": 1.0, "nodata": 0.0}]},
    )
    ds = _FakeSentinel2(item, apply_scale_offset=False)

    projection = Projection(crs, 1, -1)
    out = ds.read_raster(
        layer_name="layer",
        item=item,
        bands=["B04"],
        projection=projection,
        bounds=(0, 0, 2, 2),
    )

    arr = out.get_chw_array()
    assert arr.dtype == raw.dtype
    np.testing.assert_array_equal(arr[0], raw)


def test_read_raster_preserves_nodata_from_metadata(tmp_path: Path) -> None:
    tif_path = tmp_path / "red.tif"

    crs = CRS.from_epsg(3857)
    transform = Affine(1, 0, 0, 0, -1, 0)
    raw = np.array([[0, 2000], [3000, 0]], dtype=np.uint16)
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype=str(raw.dtype),
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(raw, 1)

    geom = STGeometry(
        Projection(crs, 1, -1),
        shapely.box(0, 0, 2, 2),
        (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)),
    )
    item = EarthDailyItem(
        name="item1",
        geometry=geom,
        asset_urls={"red": str(tif_path)},
        asset_scale_offsets={"red": [{"scale": 0.001, "offset": 1.0, "nodata": 0.0}]},
    )
    ds = _FakeSentinel2(item)

    projection = Projection(crs, 1, -1)
    out = ds.read_raster(
        layer_name="layer",
        item=item,
        bands=["B04"],
        projection=projection,
        bounds=(0, 0, 2, 2),
    )

    arr = out.get_chw_array()
    assert arr.dtype == np.float32
    expected = np.array([[0.0, 3.0], [4.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(arr[0], expected)


def test_earthdaily_item_serialize_roundtrip_preserves_product_id() -> None:
    geom = STGeometry(
        Projection(CRS.from_epsg(3857), 1, -1),
        shapely.box(0, 0, 2, 2),
        (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)),
    )
    item = EarthDailyItem(
        name="item1",
        geometry=geom,
        asset_urls={"red": "/tmp/red.tif"},
        product_id="S2A_MSIL2A_20240101T000000_N0511_R080_T15CWM_20240101T150509",
    )

    restored = EarthDailyItem.deserialize(item.serialize())

    assert restored.product_id == item.product_id


def test_init_requires_float32_band_dtype_when_scale_offset_enabled() -> None:
    layer_cfg = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.UINT16, bands=["B04"])],
    )
    with pytest.raises(ValueError, match="requires band_sets dtype=float32"):
        Sentinel2(
            apply_scale_offset=True,
            assets=["red"],
            context=DataSourceContext(layer_config=layer_cfg),
        )


def test_init_allows_non_float32_when_scale_offset_disabled() -> None:
    layer_cfg = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.UINT16, bands=["B04"])],
    )
    Sentinel2(
        apply_scale_offset=False,
        assets=["red"],
        context=DataSourceContext(layer_config=layer_cfg),
    )
