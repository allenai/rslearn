from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio
import shapely
from rasterio.crs import CRS
from rasterio.transform import Affine
from upath import UPath

pytest.importorskip("earthdaily")

from rslearn.config import BandSetConfig, DType, LayerConfig, LayerType
from rslearn.data_sources import DataSourceContext
from rslearn.data_sources.earthdaily import EarthDailyItem, Sentinel2EDACloudMask
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import Projection, STGeometry


def test_sentinel2_eda_cloud_mask_defaults() -> None:
    ds = Sentinel2EDACloudMask(cache_dir=None)

    assert ds.collection_name == "sentinel-2-eda-cloud-mask"
    assert ds.asset_bands == {"cloud-mask": ["cloud-mask", "cirrus-mask"]}
    assert ds.read_scale_offsets is False


def test_sentinel2_eda_cloud_mask_rejects_unknown_assets() -> None:
    with pytest.raises(
        ValueError, match="unknown EarthDaily Sentinel-2 EDA cloud mask assets"
    ):
        Sentinel2EDACloudMask(assets=["cirrus"], cache_dir=None)


def test_sentinel2_eda_cloud_mask_accepts_additional_ordered_bands() -> None:
    ds = Sentinel2EDACloudMask(
        band_names=["cloud-mask", "custom-cirrus"],
        cache_dir=None,
    )

    assert ds.asset_bands == {"cloud-mask": ["cloud-mask", "custom-cirrus"]}


def test_sentinel2_eda_cloud_mask_infers_asset_from_layer_config() -> None:
    layer_cfg = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.UINT8, bands=["cloud-mask"])],
    )

    ds = Sentinel2EDACloudMask(
        context=DataSourceContext(layer_config=layer_cfg),
        cache_dir=None,
    )

    assert ds.asset_bands == {"cloud-mask": ["cloud-mask"]}


def test_sentinel2_eda_cloud_mask_infers_cirrus_asset_from_layer_config() -> None:
    layer_cfg = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.UINT8, bands=["cirrus-mask"])],
    )

    ds = Sentinel2EDACloudMask(
        context=DataSourceContext(layer_config=layer_cfg),
        cache_dir=None,
    )

    assert ds.asset_bands == {"cloud-mask": ["cirrus-mask"]}


def test_sentinel2_eda_cloud_mask_read_raster(tmp_path: Path) -> None:
    tif_path = tmp_path / "cloud-mask.tif"

    crs = CRS.from_epsg(3857)
    transform = Affine(1, 0, 0, 0, -1, 0)
    raw = np.array(
        [
            [[0, 1], [2, 4]],
            [[0, 1], [1, 2]],
        ],
        dtype=np.uint8,
    )
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=2,
        dtype=str(raw.dtype),
        crs=crs,
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(raw)

    geom = STGeometry(
        Projection(crs, 1, -1),
        shapely.box(0, 0, 2, 2),
        (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)),
    )
    item = EarthDailyItem(
        name="item1",
        geometry=geom,
        asset_urls={"cloud-mask": str(tif_path)},
    )
    ds = Sentinel2EDACloudMask(cache_dir=None)

    out = ds.read_raster(
        layer_name="layer",
        item=item,
        bands=["cloud-mask"],
        projection=Projection(crs, 1, -1),
        bounds=(0, 0, 2, 2),
    )

    assert out.metadata is not None
    assert out.metadata.nodata_value == 0
    np.testing.assert_array_equal(out.get_chw_array(), raw[:1])


def test_sentinel2_eda_cloud_mask_read_raster_keeps_configured_bands(
    tmp_path: Path,
) -> None:
    tif_path = tmp_path / "cloud-mask.tif"

    crs = CRS.from_epsg(3857)
    transform = Affine(1, 0, 0, 0, -1, 0)
    raw = np.array(
        [
            [[0, 1], [2, 4]],
            [[9, 8], [7, 6]],
        ],
        dtype=np.uint8,
    )
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=2,
        dtype=str(raw.dtype),
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(raw)

    geom = STGeometry(
        Projection(crs, 1, -1),
        shapely.box(0, 0, 2, 2),
        (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)),
    )
    item = EarthDailyItem(
        name="item1",
        geometry=geom,
        asset_urls={"cloud-mask": str(tif_path)},
    )
    ds = Sentinel2EDACloudMask(
        cache_dir=None,
    )

    out = ds.read_raster(
        layer_name="layer",
        item=item,
        bands=["cloud-mask", "cirrus-mask"],
        projection=Projection(crs, 1, -1),
        bounds=(0, 0, 2, 2),
    )

    np.testing.assert_array_equal(out.get_chw_array(), raw)


def test_sentinel2_eda_cloud_mask_read_raster_keeps_cirrus_band_only(
    tmp_path: Path,
) -> None:
    tif_path = tmp_path / "cloud-mask.tif"

    crs = CRS.from_epsg(3857)
    transform = Affine(1, 0, 0, 0, -1, 0)
    raw = np.array(
        [
            [[0, 1], [2, 4]],
            [[0, 1], [1, 2]],
        ],
        dtype=np.uint8,
    )
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=2,
        dtype=str(raw.dtype),
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(raw)

    geom = STGeometry(
        Projection(crs, 1, -1),
        shapely.box(0, 0, 2, 2),
        (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)),
    )
    item = EarthDailyItem(
        name="item1",
        geometry=geom,
        asset_urls={"cloud-mask": str(tif_path)},
    )
    layer_cfg = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.UINT8, bands=["cirrus-mask"])],
    )
    ds = Sentinel2EDACloudMask(
        context=DataSourceContext(layer_config=layer_cfg),
        cache_dir=None,
    )

    out = ds.read_raster(
        layer_name="layer",
        item=item,
        bands=["cirrus-mask"],
        projection=Projection(crs, 1, -1),
        bounds=(0, 0, 2, 2),
    )

    np.testing.assert_array_equal(out.get_chw_array(), raw[1:2])


def test_sentinel2_eda_cloud_mask_ingest_writes_first_band_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tif_path = tmp_path / "cloud-mask.tif"

    crs = CRS.from_epsg(3857)
    transform = Affine(1, 0, 0, 0, -1, 0)
    raw = np.array(
        [
            [[0, 1], [2, 4]],
            [[0, 1], [1, 2]],
        ],
        dtype=np.uint8,
    )
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=2,
        dtype=str(raw.dtype),
        crs=crs,
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(raw)

    geom = STGeometry(
        Projection(crs, 1, -1),
        shapely.box(0, 0, 2, 2),
        (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)),
    )
    item = EarthDailyItem(
        name="item1",
        geometry=geom,
        asset_urls={"cloud-mask": "https://example.com/cloud-mask.tif"},
    )
    ds = Sentinel2EDACloudMask(band_names=["cloud-mask"], cache_dir=None)
    monkeypatch.setattr(
        ds,
        "_download_asset_to_tmp",
        lambda _asset_url, _tmp_dir, _asset_key, _item_name: str(tif_path),
    )

    tile_store = DefaultTileStore(convert_rasters_to_cogs=False)
    tile_store.set_dataset_path(UPath(tmp_path / "ds"))
    layer_tile_store = TileStoreWithLayer(tile_store, "cloud")

    ds.ingest(layer_tile_store, [item], [[geom]])

    out = tile_store.read_raster(
        "cloud",
        item,
        ["cloud-mask"],
        Projection(crs, 1, -1),
        (0, 0, 2, 2),
    )
    np.testing.assert_array_equal(out.get_chw_array(), raw[:1])


def test_sentinel2_eda_cloud_mask_ingest_writes_configured_bands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tif_path = tmp_path / "cloud-mask.tif"

    crs = CRS.from_epsg(3857)
    transform = Affine(1, 0, 0, 0, -1, 0)
    raw = np.array(
        [
            [[0, 1], [2, 4]],
            [[9, 8], [7, 6]],
        ],
        dtype=np.uint8,
    )
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=2,
        dtype=str(raw.dtype),
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(raw)

    geom = STGeometry(
        Projection(crs, 1, -1),
        shapely.box(0, 0, 2, 2),
        (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)),
    )
    item = EarthDailyItem(
        name="item1",
        geometry=geom,
        asset_urls={"cloud-mask": "https://example.com/cloud-mask.tif"},
    )
    ds = Sentinel2EDACloudMask(
        cache_dir=None,
    )
    monkeypatch.setattr(
        ds,
        "_download_asset_to_tmp",
        lambda _asset_url, _tmp_dir, _asset_key, _item_name: str(tif_path),
    )

    tile_store = DefaultTileStore(convert_rasters_to_cogs=False)
    tile_store.set_dataset_path(UPath(tmp_path / "ds"))
    layer_tile_store = TileStoreWithLayer(tile_store, "cloud")

    ds.ingest(layer_tile_store, [item], [[geom]])

    out = tile_store.read_raster(
        "cloud",
        item,
        ["cloud-mask", "cirrus-mask"],
        Projection(crs, 1, -1),
        (0, 0, 2, 2),
    )
    np.testing.assert_array_equal(out.get_chw_array(), raw)
