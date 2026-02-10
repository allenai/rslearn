from datetime import datetime
from unittest.mock import patch

from rslearn.data_sources.aws_sentinel2_element84 import Sentinel2
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
