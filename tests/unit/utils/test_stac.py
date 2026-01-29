"""Unit tests for the STAC client."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from rslearn.utils.stac import StacClient


def test_search_request_formatting() -> None:
    """Test that the search request is correctly formatted."""
    client = StacClient("https://example.com/stac")

    mock_response = MagicMock()
    mock_response.json.return_value = {"features": [], "links": []}
    mock_response.raise_for_status = MagicMock()

    with patch.object(client.session, "post", return_value=mock_response) as mock_post:
        # Tell STAC client to make a search request.
        # It will submit it using the STAC JSON POST option.
        client.search(
            collections=["sentinel-2-l2a"],
            bbox=(-122.5, 47.5, -122.0, 48.0),
            date_time=(
                datetime(2025, 1, 1, tzinfo=UTC),
                datetime(2025, 1, 2, tzinfo=UTC),
            ),
            query={"eo:cloud_cover": {"lt": 50}},
        )

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        request_data = call_kwargs.kwargs["json"]

        # So make sure each of the four arguments we passed to search end up formatted
        # correctly in the JSON request data.
        assert request_data["collections"] == ["sentinel-2-l2a"]
        assert request_data["bbox"] == (-122.5, 47.5, -122.0, 48.0)
        assert request_data["datetime"] == "2025-01-01T00:00:00Z/2025-01-02T00:00:00Z"
        assert request_data["query"] == {"eo:cloud_cover": {"lt": 50}}


def test_search_response_parsing() -> None:
    """Test that STAC items are correctly parsed from the response."""
    client = StacClient("https://example.com/stac")

    # Simplified response based on real Element84 STAC API data
    response_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "S2A_54LWP_20250101_0_L2A",
                "properties": {
                    "platform": "sentinel-2a",
                    "eo:cloud_cover": 26.876989,
                    "datetime": "2025-01-01T00:59:58.336000Z",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [140.999826, -9.950874],
                            [140.999826, -10.943923],
                            [142.004655, -10.942270],
                            [142.001471, -9.949375],
                            [140.999826, -9.950874],
                        ]
                    ],
                },
                "bbox": [140.999826, -10.943923, 142.004655, -9.949375],
                "assets": {
                    "blue": {
                        "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/54/L/WP/2025/1/S2A_54LWP_20250101_0_L2A/B02.tif",
                        "title": "Blue (band 2) - 10m",
                        "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                        "roles": ["data", "reflectance"],
                    },
                    "red": {
                        "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/54/L/WP/2025/1/S2A_54LWP_20250101_0_L2A/B04.tif",
                        "title": "Red (band 4) - 10m",
                        "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                        "roles": ["data", "reflectance"],
                    },
                },
                "collection": "sentinel-2-l2a",
            }
        ],
        "links": [],
    }

    mock_response = MagicMock()
    mock_response.json.return_value = response_data
    mock_response.raise_for_status = MagicMock()

    with patch.object(client.session, "post", return_value=mock_response):
        items = client.search()

        assert len(items) == 1
        item = items[0]

        # Verify item ID
        assert item.id == "S2A_54LWP_20250101_0_L2A"

        # Verify collection
        assert item.collection == "sentinel-2-l2a"

        # Verify properties
        assert item.properties["platform"] == "sentinel-2a"
        assert item.properties["eo:cloud_cover"] == 26.876989

        # Verify bbox
        assert item.bbox == (140.999826, -10.943923, 142.004655, -9.949375)

        # Verify geometry
        assert item.geometry is not None
        assert item.geometry["type"] == "Polygon"
        assert len(item.geometry["coordinates"][0]) == 5

        # Verify assets
        assert item.assets is not None
        assert "blue" in item.assets
        assert "red" in item.assets
        assert (
            item.assets["blue"].href
            == "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/54/L/WP/2025/1/S2A_54LWP_20250101_0_L2A/B02.tif"
        )
        assert item.assets["blue"].title == "Blue (band 2) - 10m"
        assert (
            item.assets["blue"].type
            == "image/tiff; application=geotiff; profile=cloud-optimized"
        )
        assert item.assets["blue"].roles == ["data", "reflectance"]

        # Verify time range (should be same start/end since only datetime is set)
        assert item.time_range is not None
        assert item.time_range[0] == datetime(2025, 1, 1, 0, 59, 58, 336000, tzinfo=UTC)
        assert item.time_range[1] == datetime(2025, 1, 1, 0, 59, 58, 336000, tzinfo=UTC)
