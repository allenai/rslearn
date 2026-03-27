from datetime import datetime
from unittest.mock import Mock, patch

import shapely

from rslearn.config import (
    BandSetConfig,
    DataSourceConfig,
    DType,
    LayerConfig,
    LayerType,
    QueryConfig,
    SpaceMode,
)
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSourceContext
from rslearn.data_sources.aws_sentinel2_element84 import Sentinel2
from rslearn.data_sources.stac import SourceItem
from rslearn.utils.geometry import STGeometry
from rslearn.utils.stac import StacAsset, StacItem


def test_sentinel2_get_item_by_name_delegates_to_stac_data_source() -> None:
    """Ensure get_item_by_name works and doesn't raise NotImplementedError."""
    data_source = Sentinel2(assets=["visual"])

    mock_stac_item = StacItem(
        id="test-item-id",
        properties={
            "datetime": "2024-01-01T00:00:00Z",
            "earthsearch:boa_offset_applied": False,
        },
        collection="sentinel-2-l2a",
        bbox=(-122.0, 37.0, -121.0, 38.0),
        geometry={
            "type": "Polygon",
            "coordinates": [
                [[-122, 37], [-121, 37], [-121, 38], [-122, 38], [-122, 37]]
            ],
        },
        assets={
            "visual": StacAsset(
                href="https://example.com/visual.tif",
                title="Visual",
                type="image/tiff",
                roles=["data"],
            )
        },
        time_range=(datetime(2024, 1, 1), datetime(2024, 1, 1)),
    )

    with patch.object(data_source.client, "search", return_value=[mock_stac_item]):
        item = data_source.get_item_by_name("test-item-id")

    assert item.name == "test-item-id"
    assert "visual" in item.asset_urls


def _make_aws_sentinel2_context(
    *,
    ingest: bool,
    space_mode: SpaceMode = SpaceMode.MOSAIC,
) -> DataSourceContext:
    layer_cfg = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.UINT16, bands=["B04", "B03", "B8A"])],
        data_source=DataSourceConfig(
            class_path="rslearn.data_sources.aws_sentinel2_element84.Sentinel2",
            init_args={},
            query_config=QueryConfig(space_mode=space_mode),
            ingest=ingest,
        ),
    )
    return DataSourceContext(layer_config=layer_cfg)


def _make_source_item(name: str) -> SourceItem:
    geometry = STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 1, 1), None)
    return SourceItem(
        name=name,
        geometry=geometry,
        asset_urls={
            "red": f"https://example.com/{name}_red.tif",
            "green": f"https://example.com/{name}_green.tif",
            "nir08": f"https://example.com/{name}_nir08.tif",
        },
        properties={},
    )


def test_aws_sentinel2_omnicloudmask_skips_prepare_ranking_when_ingest_disabled() -> (
    None
):
    data_source = Sentinel2(
        context=_make_aws_sentinel2_context(
            ingest=False, space_mode=SpaceMode.SINGLE_COMPOSITE
        ),
        sort_by_omnicloudmask=True,
    )
    geometry = STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 1, 1), None)
    items = [_make_source_item("item0"), _make_source_item("item1")]

    with patch(
        "rslearn.data_sources.omnicloudmask_utils.sort_items_by_omnicloudmask"
    ) as sort_mock:
        result = data_source._post_filter_items(geometry, items)

    assert result == items
    sort_mock.assert_not_called()


def test_aws_sentinel2_omnicloudmask_prepare_ranking_when_ingest_disabled_non_single_composite() -> (
    None
):
    data_source = Sentinel2(
        context=_make_aws_sentinel2_context(ingest=False, space_mode=SpaceMode.MOSAIC),
        sort_by_omnicloudmask=True,
    )
    geometry = STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 1, 1), None)
    items = [_make_source_item("item0"), _make_source_item("item1")]

    with patch(
        "rslearn.data_sources.omnicloudmask_utils.sort_items_by_omnicloudmask",
        return_value=[items[1], items[0]],
    ) as sort_mock:
        result = data_source._post_filter_items(geometry, items)

    assert result == [items[1], items[0]]
    sort_mock.assert_called_once()


def test_aws_sentinel2_omnicloudmask_reranks_during_materialize_when_deferred() -> None:
    data_source = Sentinel2(
        context=_make_aws_sentinel2_context(
            ingest=False, space_mode=SpaceMode.SINGLE_COMPOSITE
        ),
        sort_by_omnicloudmask=True,
    )
    item0 = _make_source_item("item0")
    item1 = _make_source_item("item1")
    item_groups = [[item0, item1], [item1]]
    window = Mock()
    window.get_geometry.return_value = STGeometry(
        WGS84_PROJECTION, shapely.box(0, 0, 1, 1), None
    )
    layer_cfg = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.UINT16, bands=["B04"])],
    )
    captured_groups: dict[str, list[list[SourceItem]]] = {}

    def fake_materialize(
        _self: object,
        _tile_store: object,
        _window: object,
        _layer_name: str,
        _layer_cfg: LayerConfig,
        materialize_item_groups: list[list[SourceItem]],
    ) -> None:
        captured_groups["item_groups"] = materialize_item_groups

    with (
        patch(
            "rslearn.data_sources.omnicloudmask_utils.sort_items_by_omnicloudmask",
            side_effect=lambda items, *args, **kwargs: list(reversed(items)),
        ) as sort_mock,
        patch(
            "rslearn.data_sources.direct_materialize_data_source.RasterMaterializer.materialize",
            new=fake_materialize,
        ),
    ):
        data_source.materialize(window, item_groups, "sentinel2", layer_cfg)

    assert captured_groups["item_groups"] == [[item1, item0], [item1]]
    assert sort_mock.call_count == 1
