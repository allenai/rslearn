import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio
import shapely
from rasterio.crs import CRS
from rasterio.transform import Affine

pytest.importorskip("earthdaily")

from rslearn.data_sources.earthdaily import EarthDailyItem, Sentinel2L2A
from rslearn.utils.geometry import Projection, STGeometry


def _make_item(asset_urls: dict[str, str]) -> EarthDailyItem:
    geom = STGeometry(
        Projection(CRS.from_epsg(3857), 1, -1),
        shapely.box(0, 0, 2, 2),
        (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)),
    )
    return EarthDailyItem(name="item1", geometry=geom, asset_urls=asset_urls)


def test_read_raster_harmonizes_non_visual_band(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tif_path = tmp_path / "B04.tif"
    raw = np.array([[[900, 1000], [1200, 2200]]], dtype=np.uint16)
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype=str(raw.dtype),
        crs=CRS.from_epsg(3857),
        transform=Affine(1, 0, 0, 0, -1, 0),
    ) as dst:
        dst.write(raw)

    item = _make_item(
        {"B04": str(tif_path), "product_metadata": "https://example.com/meta.xml"}
    )
    ds = Sentinel2L2A(harmonize=True, assets=["B04"], cache_dir=None)
    monkeypatch.setattr(ds, "get_item_by_name", lambda _name: item)
    monkeypatch.setattr(
        ds,
        "_get_product_xml",
        lambda _item: ET.fromstring(
            "<root><BOA_ADD_OFFSET>-1000</BOA_ADD_OFFSET></root>"
        ),
    )

    out = ds.read_raster(
        layer_name="layer",
        item_name=item.name,
        bands=["B04"],
        projection=Projection(CRS.from_epsg(3857), 1, -1),
        bounds=(0, 0, 2, 2),
    ).get_chw_array()

    expected = np.clip(raw, 1000, None) - 1000
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, expected)


def test_read_raster_does_not_harmonize_visual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tif_path = tmp_path / "visual.tif"
    raw = np.array(
        [
            [[10, 20], [30, 40]],
            [[11, 21], [31, 41]],
            [[12, 22], [32, 42]],
        ],
        dtype=np.uint8,
    )
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=3,
        dtype=str(raw.dtype),
        crs=CRS.from_epsg(3857),
        transform=Affine(1, 0, 0, 0, -1, 0),
    ) as dst:
        dst.write(raw)

    item = _make_item(
        {"visual": str(tif_path), "product_metadata": "https://example.com/meta.xml"}
    )
    ds = Sentinel2L2A(harmonize=True, assets=["visual"], cache_dir=None)
    monkeypatch.setattr(ds, "get_item_by_name", lambda _name: item)
    monkeypatch.setattr(
        ds,
        "_get_product_xml",
        lambda _item: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    out = ds.read_raster(
        layer_name="layer",
        item_name=item.name,
        bands=["R", "G", "B"],
        projection=Projection(CRS.from_epsg(3857), 1, -1),
        bounds=(0, 0, 2, 2),
    ).get_chw_array()

    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, raw)


def test_rejects_unknown_assets() -> None:
    with pytest.raises(ValueError, match="unknown EarthDaily Sentinel-2 L2A assets"):
        Sentinel2L2A(assets=["red"], cache_dir=None)


def test_sentinel2_l2a_disables_scale_offset_parsing() -> None:
    ds = Sentinel2L2A(cache_dir=None)
    assert ds.read_scale_offsets is False


def test_read_raster_harmonizes_with_date_fallback_when_metadata_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tif_path = tmp_path / "B04.tif"
    raw = np.array([[[900, 1000], [1200, 2200]]], dtype=np.uint16)
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype=str(raw.dtype),
        crs=CRS.from_epsg(3857),
        transform=Affine(1, 0, 0, 0, -1, 0),
    ) as dst:
        dst.write(raw)

    item = _make_item({"B04": str(tif_path)})
    ds = Sentinel2L2A(harmonize=True, assets=["B04"], cache_dir=None)
    monkeypatch.setattr(ds, "get_item_by_name", lambda _name: item)

    out = ds.read_raster(
        layer_name="layer",
        item_name=item.name,
        bands=["B04"],
        projection=Projection(CRS.from_epsg(3857), 1, -1),
        bounds=(0, 0, 2, 2),
    ).get_chw_array()

    expected = np.clip(raw, 1000, None) - 1000
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, expected)


def test_read_raster_no_date_fallback_before_cutoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tif_path = tmp_path / "B04.tif"
    raw = np.array([[[900, 1000], [1200, 2200]]], dtype=np.uint16)
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype=str(raw.dtype),
        crs=CRS.from_epsg(3857),
        transform=Affine(1, 0, 0, 0, -1, 0),
    ) as dst:
        dst.write(raw)

    geom = STGeometry(
        Projection(CRS.from_epsg(3857), 1, -1),
        shapely.box(0, 0, 2, 2),
        (datetime(2021, 1, 1, tzinfo=UTC), datetime(2021, 1, 2, tzinfo=UTC)),
    )
    item = EarthDailyItem(name="item1", geometry=geom, asset_urls={"B04": str(tif_path)})
    ds = Sentinel2L2A(harmonize=True, assets=["B04"], cache_dir=None)
    monkeypatch.setattr(ds, "get_item_by_name", lambda _name: item)

    out = ds.read_raster(
        layer_name="layer",
        item_name=item.name,
        bands=["B04"],
        projection=Projection(CRS.from_epsg(3857), 1, -1),
        bounds=(0, 0, 2, 2),
    ).get_chw_array()

    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, raw)
