"""Unit tests for rslearn.data_sources.stac."""

from datetime import UTC, datetime
from typing import Any

import pytest
import shapely
from pyproj import CRS

from rslearn.config import QueryConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.stac import StacDataSource
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.stac import StacAsset, StacItem


def _make_antimeridian_utm_geometry() -> STGeometry:
    """Build a small UTM box that straddles the antimeridian."""
    utm_proj = Projection(CRS.from_epsg(32701), 10, -10)
    p1 = STGeometry(WGS84_PROJECTION, shapely.Point(-179.99, -16), None).to_projection(
        utm_proj
    )
    p2 = STGeometry(WGS84_PROJECTION, shapely.Point(179.99, -15), None).to_projection(
        utm_proj
    )
    return STGeometry(
        utm_proj, shapely.box(p1.shp.x, p1.shp.y, p2.shp.x, p2.shp.y), None
    )


def test_antimeridian_geometry_produces_multipolygon_intersects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The EPSG:32701 should be split across the antimeridian for the STAC search."""
    ds = StacDataSource(
        endpoint="https://example.com/stac",
        collection_name="test-collection",
    )

    captured_kwargs: dict[str, Any] = {}

    def fake_search(**kwargs: Any) -> list:
        captured_kwargs.update(kwargs)
        return []

    monkeypatch.setattr(ds.client, "search", fake_search)

    geom = _make_antimeridian_utm_geometry()
    ds.get_items([geom], QueryConfig())

    intersects_geojson = captured_kwargs.get("intersects")
    assert intersects_geojson is not None

    # We should get MultiPolygon with two small components.
    assert intersects_geojson["type"] == "MultiPolygon", (
        f"Expected MultiPolygon for antimeridian-crossing geometry, got {intersects_geojson['type']}"
    )

    shp = shapely.geometry.shape(intersects_geojson)
    for part in shp.geoms:
        lon_extent = part.bounds[2] - part.bounds[0]
        assert lon_extent < 1.0, (
            f"MultiPolygon component spans {lon_extent}° longitude, expected < 1°"
        )


def test_multi_collection_get_item_by_name_searches_all_collections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ds = StacDataSource(
        endpoint="https://example.com/stac",
        collection_name=["collection-a", "collection-b"],
    )

    captured_kwargs: dict[str, Any] = {}
    stac_item = StacItem(
        id="item-id",
        properties={},
        collection="collection-b",
        bbox=(-1, -1, 1, 1),
        geometry=shapely.geometry.mapping(shapely.box(-1, -1, 1, 1)),
        assets={
            "image": StacAsset(
                href="https://example.com/image.tif",
                title=None,
                type=None,
                roles=None,
            )
        },
        time_range=(
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2020, 1, 2, tzinfo=UTC),
        ),
    )

    def fake_search(**kwargs: Any) -> list[StacItem]:
        captured_kwargs.update(kwargs)
        return [stac_item]

    monkeypatch.setattr(ds.client, "search", fake_search)

    item = ds.get_item_by_name("item-id")

    assert item.name == "item-id"
    assert captured_kwargs["ids"] == ["item-id"]
    assert captured_kwargs["collections"] == ["collection-a", "collection-b"]


def test_multi_collection_get_items_searches_all_collections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ds = StacDataSource(
        endpoint="https://example.com/stac",
        collection_name=["collection-a", "collection-b"],
    )
    captured_kwargs: dict[str, Any] = {}

    def fake_search(**kwargs: Any) -> list[StacItem]:
        captured_kwargs.update(kwargs)
        return []

    monkeypatch.setattr(ds.client, "search", fake_search)

    geom = STGeometry(WGS84_PROJECTION, shapely.box(-1, -1, 1, 1), None)
    ds.get_items([geom], QueryConfig())

    assert captured_kwargs["collections"] == ["collection-a", "collection-b"]
