import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pystac
import pytest
import rasterio
import shapely
from rasterio.crs import CRS
from rasterio.transform import Affine

pytest.importorskip("earthdaily")

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.earthdaily import EarthDailyItem, Sentinel2L2A
from rslearn.utils.geometry import Projection, STGeometry


def _make_item(
    asset_urls: dict[str, str],
    *,
    name: str = "item1",
    product_id: str | None = None,
    boa_offset_applied: bool | None = None,
    start_time: datetime = datetime(2024, 1, 1, tzinfo=UTC),
    end_time: datetime = datetime(2024, 1, 2, tzinfo=UTC),
) -> EarthDailyItem:
    geom = STGeometry(
        Projection(CRS.from_epsg(3857), 1, -1),
        shapely.box(0, 0, 2, 2),
        (start_time, end_time),
    )
    return EarthDailyItem(
        name=name,
        geometry=geom,
        asset_urls=asset_urls,
        product_id=product_id,
        boa_offset_applied=boa_offset_applied,
    )


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
        {"B04": str(tif_path), "product_metadata": "https://example.com/meta.xml"},
        boa_offset_applied=False,
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
        item=item,
        bands=["B04"],
        projection=Projection(CRS.from_epsg(3857), 1, -1),
        bounds=(0, 0, 2, 2),
    ).get_chw_array()

    expected = np.clip(raw, 1000, None) - 1000
    # Nodata is 0 so anything that wasn't 0 before should be 1.
    expected[(expected == 0) & (raw > 0)] = 1
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, expected)


def test_read_raster_skips_harmonization_when_offset_already_applied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Earth Search COGs marked as corrected must not lose another 1000 DN."""
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
        {"B04": str(tif_path), "product_metadata": "https://example.com/meta.xml"},
        boa_offset_applied=True,
    )
    ds = Sentinel2L2A(harmonize=True, assets=["B04"], cache_dir=None)
    monkeypatch.setattr(ds, "get_item_by_name", lambda _name: item)
    monkeypatch.setattr(
        ds,
        "_get_product_xml",
        lambda _item: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    out = ds.read_raster(
        layer_name="layer",
        item=item,
        bands=["B04"],
        projection=Projection(CRS.from_epsg(3857), 1, -1),
        bounds=(0, 0, 2, 2),
    ).get_chw_array()

    np.testing.assert_array_equal(out, raw)


@pytest.mark.parametrize("boa_offset_applied", [True, False])
def test_stac_item_preserves_boa_offset_applied(boa_offset_applied: bool) -> None:
    """Keep the Earth Search harmonization flag when converting STAC items."""
    stac_item = pystac.Item(
        id="S2A_TEST",
        geometry=shapely.geometry.mapping(shapely.box(0, 0, 1, 1)),
        bbox=[0, 0, 1, 1],
        datetime=datetime(2024, 1, 1, tzinfo=UTC),
        properties={"earthsearch:boa_offset_applied": boa_offset_applied},
    )
    stac_item.add_asset(
        "B04",
        pystac.Asset(
            href="s3://source/B04.tif",
            extra_fields={
                "alternate": {"download": {"href": "https://example.com/B04.tif"}}
            },
        ),
    )
    ds = Sentinel2L2A(harmonize=True, assets=["B04"], cache_dir=None)

    item = ds._stac_item_to_item(stac_item)

    assert item.boa_offset_applied is boa_offset_applied


@pytest.mark.parametrize(
    ("common_name", "canonical_name"),
    [
        ("blue", "B02"),
        ("nir", "B08"),
        ("scl", "SCL"),
    ],
)
def test_stac_item_normalizes_common_name_asset_keys(
    common_name: str, canonical_name: str
) -> None:
    stac_item = pystac.Item(
        id="S2A_TEST",
        geometry=shapely.geometry.mapping(shapely.box(0, 0, 1, 1)),
        bbox=[0, 0, 1, 1],
        datetime=datetime(2024, 1, 1, tzinfo=UTC),
        properties={},
    )
    stac_item.add_asset(
        common_name,
        pystac.Asset(
            href=f"s3://source/{common_name}.tif",
            extra_fields={
                "alternate": {
                    "download": {"href": f"https://example.com/{common_name}.tif"}
                }
            },
        ),
    )
    ds = Sentinel2L2A(assets=[canonical_name], cache_dir=None)

    item = ds._stac_item_to_item(stac_item)

    assert item.asset_urls == {canonical_name: f"https://example.com/{common_name}.tif"}


def test_stac_item_prefers_canonical_asset_key_over_common_name() -> None:
    stac_item = pystac.Item(
        id="S2A_TEST",
        geometry=shapely.geometry.mapping(shapely.box(0, 0, 1, 1)),
        bbox=[0, 0, 1, 1],
        datetime=datetime(2024, 1, 1, tzinfo=UTC),
        properties={},
    )
    for asset_key in ("B02", "blue"):
        stac_item.add_asset(
            asset_key,
            pystac.Asset(
                href=f"s3://source/{asset_key}.tif",
                extra_fields={
                    "alternate": {
                        "download": {"href": f"https://example.com/{asset_key}.tif"}
                    }
                },
            ),
        )
    ds = Sentinel2L2A(assets=["B02"], cache_dir=None)

    item = ds._stac_item_to_item(stac_item)

    assert item.asset_urls == {"B02": "https://example.com/B02.tif"}


def test_get_items_keeps_items_with_common_name_asset_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stac_item = pystac.Item(
        id="S2A_TEST",
        geometry=shapely.geometry.mapping(shapely.box(0, 0, 1, 1)),
        bbox=[0, 0, 1, 1],
        datetime=datetime(2024, 1, 1, tzinfo=UTC),
        properties={},
    )
    for asset_key in ("blue", "nir", "scl"):
        stac_item.add_asset(
            asset_key,
            pystac.Asset(
                href=f"s3://source/{asset_key}.tif",
                extra_fields={
                    "alternate": {
                        "download": {"href": f"https://example.com/{asset_key}.tif"}
                    }
                },
            ),
        )

    class SearchResult:
        def item_collection(self) -> list[pystac.Item]:
            return [stac_item]

    class Client:
        def search(self, **kwargs: object) -> SearchResult:
            return SearchResult()

    ds = Sentinel2L2A(assets=["B02", "B08", "SCL"], cache_dir=None)
    monkeypatch.setattr(ds, "_load_client", lambda: (None, Client(), None))
    monkeypatch.setattr(
        ds, "get_item_by_name", lambda _name: ds._stac_item_to_item(stac_item)
    )
    geometry = STGeometry(
        WGS84_PROJECTION,
        shapely.box(0.25, 0.25, 0.75, 0.75),
        (datetime(2023, 12, 31, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)),
    )

    groups = ds.get_items([geometry], QueryConfig(space_mode=SpaceMode.INTERSECTS))[0]

    assert len(groups) == 1
    assert groups[0].items[0].asset_urls == {
        "B02": "https://example.com/blue.tif",
        "B08": "https://example.com/nir.tif",
        "SCL": "https://example.com/scl.tif",
    }


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
        item=item,
        bands=["R", "G", "B"],
        projection=Projection(CRS.from_epsg(3857), 1, -1),
        bounds=(0, 0, 2, 2),
    ).get_chw_array()

    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, raw)


def test_read_raster_does_not_harmonize_scl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tif_path = tmp_path / "SCL.tif"
    raw = np.array([[[0, 4], [8, 11]]], dtype=np.uint8)
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
        {"SCL": str(tif_path), "product_metadata": "https://example.com/meta.xml"}
    )
    ds = Sentinel2L2A(harmonize=True, assets=["SCL"], cache_dir=None)
    monkeypatch.setattr(ds, "get_item_by_name", lambda _name: item)
    monkeypatch.setattr(
        ds,
        "_get_product_xml",
        lambda _item: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    out = ds.read_raster(
        layer_name="layer",
        item=item,
        bands=["SCL"],
        projection=Projection(CRS.from_epsg(3857), 1, -1),
        bounds=(0, 0, 2, 2),
    ).get_chw_array()

    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, raw)


def test_rejects_unknown_assets() -> None:
    with pytest.raises(ValueError, match="unknown EarthDaily Sentinel-2 L2A assets"):
        Sentinel2L2A(assets=["red"], cache_dir=None)


def test_sentinel2_l2a_exposes_scl() -> None:
    ds = Sentinel2L2A(assets=["SCL"], cache_dir=None)
    assert ds.asset_bands == {"SCL": ["SCL"]}


def test_sentinel2_l2a_disables_scale_offset_parsing() -> None:
    ds = Sentinel2L2A(cache_dir=None)
    assert ds.read_scale_offsets is False


def test_read_raster_raises_when_metadata_missing(
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

    with pytest.raises(
        KeyError,
        match=(
            "missing metadata asset URL \\(expected one of: "
            "product_metadata, granule_metadata\\)"
        ),
    ):
        ds.read_raster(
            layer_name="layer",
            item=item,
            bands=["B04"],
            projection=Projection(CRS.from_epsg(3857), 1, -1),
            bounds=(0, 0, 2, 2),
        )


def test_resolve_metadata_url_uses_granule_metadata() -> None:
    item = _make_item(
        {
            "B04": "https://example.com/B04.tif",
            "granule_metadata": "https://example.com/granule_metadata.xml",
        }
    )
    ds = Sentinel2L2A(harmonize=True, assets=["B04"], cache_dir=None)

    assert ds._resolve_metadata_url(item) == "https://example.com/granule_metadata.xml"


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
    item = EarthDailyItem(
        name="item1",
        geometry=geom,
        asset_urls={
            "B04": str(tif_path),
            "product_metadata": "https://example.com/meta.xml",
        },
    )
    ds = Sentinel2L2A(harmonize=True, assets=["B04"], cache_dir=None)
    monkeypatch.setattr(ds, "get_item_by_name", lambda _name: item)
    monkeypatch.setattr(ds, "_get_product_xml", lambda _item: ET.fromstring("<root />"))

    out = ds.read_raster(
        layer_name="layer",
        item=item,
        bands=["B04"],
        projection=Projection(CRS.from_epsg(3857), 1, -1),
        bounds=(0, 0, 2, 2),
    ).get_chw_array()

    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, raw)


def test_read_raster_harmonizes_with_processing_baseline_fallback_before_cutoff(
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
        {"B04": str(tif_path), "product_metadata": "https://example.com/meta.xml"},
        name="S2A_MSIL2A_20210101T000000_N0400_R080_T15CWM_20210101T150509",
        start_time=datetime(2021, 1, 1, tzinfo=UTC),
        end_time=datetime(2021, 1, 2, tzinfo=UTC),
    )
    ds = Sentinel2L2A(harmonize=True, assets=["B04"], cache_dir=None)
    monkeypatch.setattr(ds, "get_item_by_name", lambda _name: item)
    monkeypatch.setattr(ds, "_get_product_xml", lambda _item: ET.fromstring("<root />"))

    out = ds.read_raster(
        layer_name="layer",
        item=item,
        bands=["B04"],
        projection=Projection(CRS.from_epsg(3857), 1, -1),
        bounds=(0, 0, 2, 2),
    ).get_chw_array()

    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, raw)


def test_read_raster_processing_baseline_fallback_overrides_geometry_date(
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
        {"B04": str(tif_path), "product_metadata": "https://example.com/meta.xml"},
        name="S2A_MSIL2A_20240101T000000_N0399_R080_T15CWM_20240101T150509",
    )
    ds = Sentinel2L2A(harmonize=True, assets=["B04"], cache_dir=None)
    monkeypatch.setattr(ds, "get_item_by_name", lambda _name: item)
    monkeypatch.setattr(ds, "_get_product_xml", lambda _item: ET.fromstring("<root />"))

    out = ds.read_raster(
        layer_name="layer",
        item=item,
        bands=["B04"],
        projection=Projection(CRS.from_epsg(3857), 1, -1),
        bounds=(0, 0, 2, 2),
    ).get_chw_array()

    expected = np.clip(raw, 1000, None) - 1000
    expected[(expected == 0) & (raw > 0)] = 1
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, expected)


def test_fallback_harmonize_preserves_nodata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nodata pixels (0) must stay 0; valid pixels that would become 0 get clamped to 1."""
    tif_path = tmp_path / "B04.tif"
    raw = np.array([[[0, 500], [1000, 2000]]], dtype=np.uint16)
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
        {"B04": str(tif_path), "product_metadata": "https://example.com/meta.xml"},
        name="S2A_MSIL2A_20240101T000000_N0400_R080_T15CWM_20240101T150509",
    )
    ds = Sentinel2L2A(harmonize=True, assets=["B04"], cache_dir=None)
    monkeypatch.setattr(ds, "get_item_by_name", lambda _name: item)
    monkeypatch.setattr(ds, "_get_product_xml", lambda _item: ET.fromstring("<root />"))

    out = ds.read_raster(
        layer_name="layer",
        item=item,
        bands=["B04"],
        projection=Projection(CRS.from_epsg(3857), 1, -1),
        bounds=(0, 0, 2, 2),
    ).get_chw_array()

    expected = np.array([[[0, 1], [1, 1000]]], dtype=np.uint16)
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, expected)


def test_read_raster_uses_product_id_processing_baseline_before_item_name(
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
        {"B04": str(tif_path), "product_metadata": "https://example.com/meta.xml"},
        name="earthdaily-item-id-without-baseline",
        product_id="S2A_MSIL2A_20210101T000000_N0400_R080_T15CWM_20210101T150509",
        start_time=datetime(2021, 1, 1, tzinfo=UTC),
        end_time=datetime(2021, 1, 2, tzinfo=UTC),
    )
    ds = Sentinel2L2A(harmonize=True, assets=["B04"], cache_dir=None)
    monkeypatch.setattr(ds, "get_item_by_name", lambda _name: item)
    monkeypatch.setattr(ds, "_get_product_xml", lambda _item: ET.fromstring("<root />"))

    out = ds.read_raster(
        layer_name="layer",
        item=item,
        bands=["B04"],
        projection=Projection(CRS.from_epsg(3857), 1, -1),
        bounds=(0, 0, 2, 2),
    ).get_chw_array()

    expected = np.clip(raw, 1000, None) - 1000
    expected[(expected == 0) & (raw > 0)] = 1
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, expected)
