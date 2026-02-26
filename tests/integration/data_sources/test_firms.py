"""Mocked integration tests for the FIRMS data source."""

import json
import pathlib
from datetime import UTC, datetime

import pytest
import requests
import shapely
from upath import UPath

from rslearn.config import QueryConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.firms import FIRMS
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import STGeometry


class _MockResponse:
    """Simple response stub for mocked FIRMS requests."""

    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")


class _MockSession:
    """Simple session stub for mocking retry-enabled sessions."""

    def __init__(self, get_fn):
        self._get_fn = get_fn
        self.closed = False

    def get(self, url: str, timeout: float) -> _MockResponse:
        return self._get_fn(url, timeout)

    def close(self) -> None:
        self.closed = True


def _make_geometry() -> STGeometry:
    return STGeometry(
        WGS84_PROJECTION,
        shapely.box(-7.7, 33.5, -7.5, 33.7),
        (
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 5, tzinfo=UTC),
        ),
    )


def test_ingest(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test FIRMS ingest with chunked requests and vector output."""
    geometry = _make_geometry()
    data_source = FIRMS(
        map_key="dummy",
        max_days_per_request=2,
        spatial_bin_degrees=1.0,
    )

    item_groups = data_source.get_items([geometry], QueryConfig())[0]
    assert len(item_groups) == 1
    assert len(item_groups[0]) == 1
    item = item_groups[0][0]

    request_urls: list[str] = []

    def fake_get(url: str, timeout: float) -> _MockResponse:
        del timeout
        request_urls.append(url)
        if url.endswith("/2/2026-01-01"):
            return _MockResponse(
                "latitude,longitude,acq_date,confidence\n"
                "33.61,-7.61,2026-01-01,h\n"
                "33.61,-6.61,2026-01-01,h\n"
            )
        if url.endswith("/2/2026-01-03"):
            return _MockResponse(
                "latitude,longitude,acq_date,confidence\n"
                "33.62,-7.62,2026-01-03,n\n"
            )
        pytest.fail(f"unexpected FIRMS URL: {url}")

    session = _MockSession(fake_get)
    monkeypatch.setattr("rslearn.data_sources.firms.create_retry_session", lambda: session)

    tile_store_dir = UPath(tmp_path / "tiles")
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)
    layer_tile_store = TileStoreWithLayer(tile_store, "firms")

    data_source.ingest(
        tile_store=layer_tile_store,
        items=[item],
        geometries=[[geometry]],
    )
    assert session.closed

    # Verify requests were chunked by max_days_per_request.
    assert len(request_urls) == 2
    assert request_urls[0].endswith("/2/2026-01-01")
    assert request_urls[1].endswith("/2/2026-01-03")

    assert layer_tile_store.is_vector_ready(item.name)
    out_path = tile_store_dir / "firms" / item.name / "data.geojson"
    assert out_path.exists()
    with out_path.open() as f:
        geojson = json.load(f)

    # One point from the first chunk is outside the item bbox and should be filtered.
    assert len(geojson["features"]) == 2

    # Verify re-ingest skips already-ready item and does not make new requests.
    data_source.ingest(
        tile_store=layer_tile_store,
        items=[item],
        geometries=[[geometry]],
    )
    assert len(request_urls) == 2


def test_ingest_raises_on_http_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FIRMS ingest should propagate HTTP errors from the API call."""
    geometry = _make_geometry()
    data_source = FIRMS(
        map_key="dummy",
        max_days_per_request=2,
        spatial_bin_degrees=1.0,
    )
    item = data_source.get_items([geometry], QueryConfig())[0][0][0]

    def fake_get(url: str, timeout: float) -> _MockResponse:
        del url, timeout
        return _MockResponse("error", status_code=500)

    session = _MockSession(fake_get)
    monkeypatch.setattr("rslearn.data_sources.firms.create_retry_session", lambda: session)

    tile_store_dir = UPath(tmp_path / "tiles")
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)
    layer_tile_store = TileStoreWithLayer(tile_store, "firms")

    with pytest.raises(requests.HTTPError):
        data_source.ingest(
            tile_store=layer_tile_store,
            items=[item],
            geometries=[[geometry]],
        )
    assert session.closed
