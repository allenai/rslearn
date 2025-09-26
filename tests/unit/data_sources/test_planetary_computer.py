"""Tests for Planetary Computer ingest edge cases."""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pystac
import pytest
import rasterio
from rasterio.transform import Affine
from shapely.geometry import box, mapping
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.planetary_computer import (
    PlanetaryComputerItem,
    Sentinel1,
    Sentinel2,
)
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import STGeometry


class _FakeResponse:
    """Minimal streaming response returning fixed payload."""

    def __init__(self, payload: bytes, headers: dict[str, str] | None = None) -> None:
        self._stream = io.BytesIO(payload)
        self.headers = headers or {}

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def raise_for_status(self) -> None:  # pragma: no cover - nothing to do
        return None

    def iter_content(self, chunk_size: int = 8192):
        while True:
            chunk = self._stream.read(chunk_size)
            if not chunk:
                break
            yield chunk


def test_ingest_truncated_download_leads_to_rasterio_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A partially downloaded asset causes RasterioIOError during materialization."""
    # Prepare a valid GeoTIFF and keep its metadata for later reads.
    src_path = tmp_path / "source.tif"
    data = np.arange(25, dtype=np.uint8).reshape((1, 5, 5))
    transform = Affine.translation(0, 5) * Affine.scale(1, -1)
    with rasterio.open(
        src_path,
        "w",
        driver="GTiff",
        width=5,
        height=5,
        count=1,
        dtype=data.dtype,
        transform=transform,
        crs="EPSG:4326",
    ) as dst:
        dst.write(data)

    with src_path.open("rb") as f:
        full_payload = f.read()
    truncated_payload = full_payload[:-50]

    # Stream only the truncated payload when ingest downloads the asset.
    def fake_get(url: str, stream: bool, timeout: float):  # pragma: no cover - patched
        assert stream is True
        return _FakeResponse(
            truncated_payload,
            headers={"Content-Length": str(len(full_payload))},
        )

    monkeypatch.setattr(
        "rslearn.data_sources.planetary_computer.requests.get", fake_get
    )
    monkeypatch.setattr(
        "rslearn.data_sources.planetary_computer.planetary_computer.sign",
        lambda url: url,
    )

    tile_store = DefaultTileStore(convert_rasters_to_cogs=False)
    tile_store.set_dataset_path(UPath(tmp_path / "dataset"))
    layer_name = "layer"
    wrapped_tile_store = TileStoreWithLayer(tile_store, layer_name)

    now = datetime.utcnow()
    geometry = STGeometry(WGS84_PROJECTION, box(0, 0, 1, 1), (now, now))
    item = PlanetaryComputerItem(
        name="fake-item",
        geometry=geometry,
        asset_urls={"B01": "https://example.test/fake"},
    )

    data_source = Sentinel2(assets=["B01"])

    with pytest.raises(IOError, match="Incomplete download"):
        data_source.ingest(wrapped_tile_store, [item], [[geometry]])

    assert not tile_store.is_raster_ready(layer_name, item.name, ["B01"])


def test_read_raster_refreshes_stale_asset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Detect stale cache entries causing missing blobs during materialization."""

    cache_dir = UPath(tmp_path / "pc_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_source = Sentinel1(band_names=["vh"], cache_dir=cache_dir)
    item_name = "S1_fake_item"

    geometry = STGeometry(
        projection=WGS84_PROJECTION,
        shp=box(0, 0, 1, 1),
        time_range=(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )

    old_href = "https://example.test/old/iw-vh.rtc.tiff"
    new_href = "https://example.test/new/iw-vh.rtc.tiff"

    cached_item = PlanetaryComputerItem(item_name, geometry, {"vh": old_href})
    with (cache_dir / f"{item_name}.json").open("w") as f:
        json.dump(cached_item.serialize(), f)

    stac_item = pystac.Item(
        id=item_name,
        geometry=mapping(box(0, 0, 1, 1)),
        bbox=box(0, 0, 1, 1).bounds,
        datetime=datetime(2024, 1, 1, tzinfo=timezone.utc),
        properties={},
    )
    stac_item.add_asset("vh", pystac.Asset(href=new_href))

    collection = SimpleNamespace(get_item=lambda name: stac_item)
    monkeypatch.setattr(data_source, "_load_client", lambda: (None, collection))

    monkeypatch.setattr(
        "rslearn.data_sources.planetary_computer.planetary_computer.sign",
        lambda url: url,
    )

    open_calls: list[str] = []

    class _Dataset:
        def __enter__(self) -> "_Dataset":
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
            return False

    class _WarpedVRT:
        def __init__(self, src, **_: object) -> None:
            self.src = src

        def __enter__(self) -> "_WarpedVRT":
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
            return False

        def read(self) -> np.ndarray:
            return np.zeros((1, 1, 1), dtype=np.float32)

    def fake_open(url: str, *args: object, **kwargs: object) -> _Dataset:
        open_calls.append(url)
        if url == old_href:
            raise rasterio.errors.RasterioIOError("not found")
        if url == new_href:
            return _Dataset()
        raise AssertionError(url)

    monkeypatch.setattr(
        "rslearn.data_sources.planetary_computer.rasterio.open", fake_open
    )
    monkeypatch.setattr(
        "rslearn.data_sources.planetary_computer.rasterio.vrt.WarpedVRT",
        _WarpedVRT,
    )

    result = data_source.read_raster(
        layer_name="layer",
        item_name=item_name,
        bands=["vh"],
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 1, 1),
    )

    assert result.shape == (1, 1, 1)
    assert open_calls == [old_href, new_href]

    with (cache_dir / f"{item_name}.json").open() as f:
        refreshed_payload = json.load(f)
    assert refreshed_payload["asset_urls"]["vh"] == new_href
