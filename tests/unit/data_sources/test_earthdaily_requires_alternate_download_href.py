import pytest

pytest.importorskip("earthdaily")

from datetime import UTC, datetime

import pystac
import shapely

from rslearn.data_sources.earthdaily import EarthDaily


def test_stac_item_to_item_raises_without_alternate_download_href() -> None:
    ds = EarthDaily(collection_name="dummy", asset_bands={"lai": ["lai"]})

    geom = shapely.box(-1, -1, 1, 1).__geo_interface__
    item = pystac.Item(
        id="item1",
        geometry=geom,
        bbox=[-1, -1, 1, 1],
        datetime=datetime(2024, 1, 1, tzinfo=UTC),
        properties={},
    )
    # href present but alternate.download.href is missing -> should raise.
    item.add_asset("lai", pystac.Asset(href="https://example.com/lai.tif"))

    with pytest.raises(ValueError, match=r"alternate\\.download\\.href"):
        ds._stac_item_to_item(item)

