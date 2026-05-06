import inspect
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

import numpy as np
import pytest
import rasterio
import shapely
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from upath import UPath

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.chelsa import CHELSADaily, CHELSADailyItem
from rslearn.dataset.compositing import TemporalMaxCompositor, TemporalMeanCompositor
from rslearn.dataset.tile_utils import read_raster_window_from_tiles
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import Projection, STGeometry, get_global_raster_bounds
from rslearn.utils.raster_array import RasterArray


def _window(start: datetime, end: datetime) -> STGeometry:
    return STGeometry(
        WGS84_PROJECTION,
        shapely.box(-1.0, -1.0, 1.0, 1.0),
        (start, end),
    )


def test_chelsa_daily_get_asset_url() -> None:
    data_source = CHELSADaily(band_names=["tas"])
    item = data_source.get_item_by_name("chelsa_daily_20230616")

    assert data_source.get_asset_url(item, "tas") == (
        "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/daily/tas/2023/"
        "CHELSA_tas_16_06_2023_V.2.1.tif"
    )


def test_chelsa_daily_has_no_extent_init_arg() -> None:
    assert "extent" not in inspect.signature(CHELSADaily.__init__).parameters


def test_chelsa_daily_precipitation_alias_switches_for_pr_band() -> None:
    data_source = CHELSADaily(band_names=["pr"])

    pre_overlap = data_source.get_item_by_name("chelsa_daily_20191231")
    overlap = data_source.get_item_by_name("chelsa_daily_20200601")
    post_overlap = data_source.get_item_by_name("chelsa_daily_20210101")

    assert data_source.get_asset_url(pre_overlap, "pr") == (
        "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/daily/pr/2019/"
        "CHELSA_pr_31_12_2019_V.2.1.tif"
    )
    assert data_source.get_asset_url(overlap, "pr") == (
        "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/daily/pr/2020/"
        "CHELSA_pr_01_06_2020_V.2.1.tif"
    )
    assert data_source.get_asset_url(post_overlap, "pr") == (
        "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/daily/prec/2021/"
        "CHELSA_prec_01_01_2021_V.2.1.tif"
    )


def test_chelsa_daily_precipitation_alias_switches_for_prec_band() -> None:
    data_source = CHELSADaily(band_names=["prec"])

    pre_overlap = data_source.get_item_by_name("chelsa_daily_20191231")
    overlap = data_source.get_item_by_name("chelsa_daily_20200601")
    post_overlap = data_source.get_item_by_name("chelsa_daily_20210101")

    assert data_source.get_asset_url(pre_overlap, "prec") == (
        "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/daily/pr/2019/"
        "CHELSA_pr_31_12_2019_V.2.1.tif"
    )
    assert data_source.get_asset_url(overlap, "prec") == (
        "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/daily/prec/2020/"
        "CHELSA_prec_01_06_2020_V.2.1.tif"
    )
    assert data_source.get_asset_url(post_overlap, "prec") == (
        "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/daily/prec/2021/"
        "CHELSA_prec_01_01_2021_V.2.1.tif"
    )


def test_chelsa_daily_get_items_returns_daily_items() -> None:
    data_source = CHELSADaily(band_names=["tas"])

    groups = data_source.get_items(
        [
            _window(
                datetime(2023, 6, 16, 12, tzinfo=UTC),
                datetime(2023, 6, 18, 0, tzinfo=UTC),
            )
        ],
        query_config=QueryConfig(space_mode=SpaceMode.SINGLE_COMPOSITE),
    )

    assert len(groups) == 1
    assert len(groups[0]) == 1
    items = groups[0][0].items
    assert [item.name for item in items] == [
        "chelsa_daily_20230616",
        "chelsa_daily_20230617",
    ]


def test_chelsa_daily_get_items_clamps_to_configured_range() -> None:
    data_source = CHELSADaily(
        band_names=["tas"],
        start_date="2023-06-10",
        end_date="2023-06-12",
    )

    groups = data_source.get_items(
        [
            _window(
                datetime(2023, 6, 9, 0, tzinfo=UTC),
                datetime(2023, 6, 14, 0, tzinfo=UTC),
            )
        ],
        query_config=QueryConfig(space_mode=SpaceMode.SINGLE_COMPOSITE),
    )

    assert len(groups) == 1
    assert len(groups[0]) == 1
    items = groups[0][0].items
    assert [item.name for item in items] == [
        "chelsa_daily_20230610",
        "chelsa_daily_20230611",
        "chelsa_daily_20230612",
    ]


def test_chelsa_daily_get_items_outside_range_returns_empty_group() -> None:
    data_source = CHELSADaily(
        band_names=["tas"],
        start_date="2023-06-10",
        end_date="2023-06-12",
    )

    groups = data_source.get_items(
        [
            _window(
                datetime(2023, 6, 13, 0, tzinfo=UTC),
                datetime(2023, 6, 14, 0, tzinfo=UTC),
            )
        ],
        query_config=QueryConfig(space_mode=SpaceMode.SINGLE_COMPOSITE),
    )

    assert len(groups) == 1
    assert len(groups[0]) == 1
    assert len(groups[0][0].items) == 0


def test_chelsa_daily_requires_single_composite() -> None:
    data_source = CHELSADaily(band_names=["tas"])

    with pytest.raises(ValueError, match="SINGLE_COMPOSITE"):
        data_source.get_items(
            [
                _window(
                    datetime(2023, 6, 16, 0, tzinfo=UTC),
                    datetime(2023, 6, 17, 0, tzinfo=UTC),
                )
            ],
            query_config=QueryConfig(space_mode=SpaceMode.MOSAIC),
        )


def test_chelsa_daily_item_serialization_roundtrip() -> None:
    data_source = CHELSADaily(band_names=["tas"])
    item = data_source.get_item_by_name("chelsa_daily_20230616")

    restored = CHELSADailyItem.deserialize(item.serialize())

    assert restored.name == item.name
    assert restored.item_date == item.item_date
    assert restored.geometry.serialize() == item.geometry.serialize()


def test_chelsa_daily_bounds_limit_item_geometry() -> None:
    data_source = CHELSADaily(
        band_names=["tas"],
        bounds=[-8.0, 49.0, 2.0, 61.0],
    )
    item = data_source.get_item_by_name("chelsa_daily_20190907")

    assert item.geometry.to_projection(WGS84_PROJECTION).shp.bounds == pytest.approx(
        (-8.0, 49.0, 2.0, 61.0)
    )


def test_chelsa_daily_get_items_outside_bounds_returns_empty_group() -> None:
    data_source = CHELSADaily(
        band_names=["tas"],
        bounds=[-8.0, 49.0, 2.0, 61.0],
    )
    time_range = (
        datetime(2023, 6, 16, 0, tzinfo=UTC),
        datetime(2023, 6, 17, 0, tzinfo=UTC),
    )
    geometry = STGeometry(
        WGS84_PROJECTION,
        shapely.box(10.0, 49.0, 11.0, 50.0),
        time_range,
    )

    groups = data_source.get_items(
        [geometry],
        query_config=QueryConfig(space_mode=SpaceMode.SINGLE_COMPOSITE),
    )

    assert len(groups) == 1
    assert len(groups[0]) == 1
    assert groups[0][0].items == []
    assert groups[0][0].request_time_range == geometry.time_range


def test_chelsa_daily_get_raster_bounds_global_item_local_crs() -> None:
    data_source = CHELSADaily(band_names=["tas"])
    item = data_source.get_item_by_name("chelsa_daily_20190907")
    projection = Projection(CRS.from_epsg(27700), 10.0, -10.0)

    bounds = data_source.get_raster_bounds("chelsa_daily", item, ["tas"], projection)

    assert bounds == get_global_raster_bounds(projection)
    assert bounds[2] > bounds[0]
    assert bounds[3] > bounds[1]


def test_chelsa_daily_read_raster_window_from_tiles_local_crs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_source = CHELSADaily(band_names=["tas"])
    item = data_source.get_item_by_name("chelsa_daily_20190907")
    projection = Projection(CRS.from_epsg(27700), 10.0, -10.0)
    bounds = (62072, -28746, 62168, -28650)

    def _fake_read_raster(
        layer_name: str,
        read_item: CHELSADailyItem,
        bands: list[str],
        read_projection: Projection,
        read_bounds: tuple[int, int, int, int],
        resampling: Resampling = Resampling.bilinear,
    ) -> RasterArray:
        assert layer_name == "chelsa_daily"
        assert read_item == item
        assert bands == ["tas"]
        assert read_projection == projection
        assert read_bounds == bounds
        assert resampling == Resampling.bilinear
        width = read_bounds[2] - read_bounds[0]
        height = read_bounds[3] - read_bounds[1]
        return RasterArray(
            chw_array=np.ones((1, height, width), dtype=np.float32),
            time_range=read_item.geometry.time_range,
        )

    monkeypatch.setattr(data_source, "read_raster", _fake_read_raster)

    raster = read_raster_window_from_tiles(
        tile_store=TileStoreWithLayer(data_source, "chelsa_daily"),
        item=item,
        bands=["tas"],
        projection=projection,
        bounds=bounds,
        nodata_val=0.0,
        band_dtype=np.float32,
        resampling=Resampling.bilinear,
    )

    assert raster is not None
    assert raster.array.shape == (1, 1, 96, 96)
    assert np.all(raster.array == 1.0)


def test_chelsa_daily_ingest_with_bounds_writes_spatial_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src_transform = from_origin(0.0, 2.0, 0.5, 0.5)
    src_crs = CRS.from_epsg(4326)
    arr = np.arange(16, dtype=np.uint16).reshape(1, 4, 4)
    src_path = tmp_path / "chelsa_source.tif"
    with rasterio.open(
        src_path,
        "w",
        driver="GTiff",
        width=arr.shape[2],
        height=arr.shape[1],
        count=1,
        dtype="uint16",
        crs=src_crs,
        transform=src_transform,
        nodata=65535,
    ) as dst:
        dst.write(arr)

    data_source = CHELSADaily(
        band_names=["tas"],
        start_date="2020-03-28",
        end_date="2020-03-28",
        bounds=[0.5, 0.5, 1.5, 1.5],
    )
    item = data_source.get_item_by_name("chelsa_daily_20200328")
    monkeypatch.setattr(
        data_source, "get_asset_url", lambda item, asset_key: str(src_path)
    )
    monkeypatch.setattr(
        "rslearn.data_sources.chelsa.requests.get",
        lambda *args, **kwargs: pytest.fail("bounded ingest should not download file"),
    )

    tile_store = DefaultTileStore(convert_rasters_to_cogs=True)
    tile_store.set_dataset_path(UPath(tmp_path / "ds"))
    layer_store = TileStoreWithLayer(tile_store, "chelsa")
    data_source.ingest(layer_store, [item], geometries=[[]])

    stored_path = tile_store._get_raster_fname("chelsa", item.name, ["tas"])
    with rasterio.open(stored_path) as stored:
        assert stored.width == 2
        assert stored.height == 2
        assert stored.transform == from_origin(0.5, 1.5, 0.5, 0.5)
        assert stored.nodata == 65535
        np.testing.assert_array_equal(stored.read(), arr[:, 1:3, 1:3])


def test_chelsa_daily_ingest_preserves_nodata_and_temporal_reducers_ignore_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _LocalResponse:
        def __init__(self, path: UPath) -> None:
            self.path = path
            self._f: BinaryIO | None = None

        def __enter__(self) -> "_LocalResponse":
            self._f = self.path.open("rb")
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            assert self._f is not None
            self._f.close()

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int) -> Iterator[bytes]:
            assert self._f is not None
            while True:
                chunk = self._f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    src_transform = from_origin(0.0, 2.0, 0.5, 0.5)
    src_crs = CRS.from_epsg(4326)

    day1 = np.full((1, 4, 4), 3000, dtype=np.uint16)
    day2 = np.full((1, 4, 4), 3100, dtype=np.uint16)
    day2[0, 0:2, 0:2] = 65535

    source_paths = {
        "2020-03-28": UPath(tmp_path / "chelsa_20200328.tif"),
        "2020-03-29": UPath(tmp_path / "chelsa_20200329.tif"),
    }
    for d, arr in [("2020-03-28", day1), ("2020-03-29", day2)]:
        with rasterio.open(
            source_paths[d],
            "w",
            driver="GTiff",
            width=arr.shape[2],
            height=arr.shape[1],
            count=1,
            dtype="uint16",
            crs=src_crs,
            transform=src_transform,
            nodata=65535,
        ) as dst:
            dst.write(arr)

    data_source = CHELSADaily(
        band_names=["tas"],
        start_date="2020-03-28",
        end_date="2020-03-29",
    )
    item1 = data_source.get_item_by_name("chelsa_daily_20200328")
    item2 = data_source.get_item_by_name("chelsa_daily_20200329")
    items = [item1, item2]

    url_to_path = {
        "mock://chelsa_daily_20200328": source_paths["2020-03-28"],
        "mock://chelsa_daily_20200329": source_paths["2020-03-29"],
    }

    monkeypatch.setattr(
        data_source,
        "get_asset_url",
        lambda item, asset_key: f"mock://{item.name}",
    )
    monkeypatch.setattr(
        "rslearn.data_sources.chelsa.requests.get",
        lambda url, stream, timeout: _LocalResponse(url_to_path[url]),
    )

    tile_store = DefaultTileStore(convert_rasters_to_cogs=True)
    tile_store.set_dataset_path(UPath(tmp_path / "ds"))
    layer_store = TileStoreWithLayer(tile_store, "chelsa")
    data_source.ingest(layer_store, items, geometries=[[], []])

    stored_path = tile_store._get_raster_fname("chelsa", item2.name, ["tas"])
    with rasterio.open(stored_path) as stored:
        assert stored.nodata == 65535

    # Read with a misaligned target grid so bilinear resampling is exercised.
    # With nodata metadata preserved, nodata pixels are excluded and do not
    # contaminate temporal reducers.
    projection = Projection(src_crs, 0.27, -0.27)
    bounds = (0, -7, 7, 0)
    temporal_max = TemporalMaxCompositor().build_composite(
        group=items,
        nodata_val=65535,
        bands=["tas"],
        bounds=bounds,
        band_dtype=np.uint16,
        tile_store=layer_store,
        projection=projection,
        resampling_method=Resampling.bilinear,
        remapper=None,
    )
    temporal_mean = TemporalMeanCompositor().build_composite(
        group=items,
        nodata_val=65535,
        bands=["tas"],
        bounds=bounds,
        band_dtype=np.uint16,
        tile_store=layer_store,
        projection=projection,
        resampling_method=Resampling.bilinear,
        remapper=None,
    )

    assert set(np.unique(temporal_max.get_chw_array()).tolist()) == {3000, 3100}
    assert set(np.unique(temporal_mean.get_chw_array()).tolist()) == {3000, 3050}
