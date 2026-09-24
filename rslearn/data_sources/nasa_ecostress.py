"""NASA ECOSTRESS L2T LSTE data source backed by LP DAAC / CMR STAC.

This provides the ECOSTRESS Tiled Land Surface Temperature and Emissivity
Instantaneous L2 Global 70 m product (``ECO_L2T_LSTE``), served as Cloud-Optimized
GeoTIFFs from NASA's Earthdata Cloud (LP DAAC). Discovery uses the same CMR STAC
LPCLOUD endpoint as :mod:`rslearn.data_sources.nasa_hls`, so an Earthdata bearer
token (``EARTHDATA_TOKEN``) is required to read the protected assets.

By default this data source exposes only the ``LST`` band (land surface / skin
temperature), which the tiled COGs provide as float32 Kelvin with a ``NaN`` fill
value.

Cloud masking: the LST retrieval is run on all pixels regardless of cloud cover, so
cloudy pixels are *not* reliably set to ``NaN`` in the LST layer -- only pixels where
no retrieval was produced are filled with ``NaN``. For authoritative cloud screening,
NASA recommends the separate ``cloud`` layer (0 = clear, 1 = cloudy); add ``"cloud"``
to ``band_names`` and keep only pixels where it equals 0.

See https://www.earthdata.nasa.gov/data/catalog/lpcloud-eco-l2t-lste-002 and
https://ecostress.jpl.nasa.gov/ for details.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import shapely
from typing_extensions import override

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSourceContext
from rslearn.data_sources.nasa_hls import (
    _HTTP_URL_PROPERTY_PREFIX,
    _NasaHlsBase,
)
from rslearn.data_sources.stac import SourceItem
from rslearn.log_utils import get_logger
from rslearn.utils.geometry import STGeometry
from rslearn.utils.stac import StacItem

logger = get_logger(__name__)


class EcostressLSTE(_NasaHlsBase):
    """NASA ECOSTRESS Tiled L2 LST&E (ECO_L2T_LSTE) data source.

    The product is a tiled (MGRS-based) 70 m instantaneous land surface temperature
    and emissivity dataset served as Cloud-Optimized GeoTIFFs from LP DAAC. Each layer
    (LST, QC, cloud, ...) is a separate COG.

    By default only the ``LST`` band (skin temperature, float32 Kelvin, ``NaN`` fill)
    is retrieved. An Earthdata bearer token is required (see ``EARTHDATA_TOKEN``).

    ECOSTRESS STAC assets are keyed by their full granule-relative path rather than a
    short band name, so we map each requested band to its asset by matching the
    ``_<band>`` filename suffix (e.g. ``..._LST`` for ``LST``).
    """

    COLLECTION_NAME = "ECO_L2T_LSTE_002"
    DEFAULT_BANDS = ["LST"]
    SUPPORTED_BANDS = [
        "LST",
        "LST_err",
        "EmisWB",
        "QC",
        "cloud",
        "water",
        "height",
        "view_zenith",
    ]

    def __init__(
        self,
        band_names: list[str] | None = None,
        query: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_ascending: bool = True,
        timeout: timedelta = timedelta(seconds=30),
        earthdata_token: str | None = None,
        s3_credentials_url: str = _NasaHlsBase.S3_CREDENTIALS_URL,
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Create an ECOSTRESS L2T LSTE data source.

        Args:
            band_names: optional bands to expose. Defaults to ``["LST"]``.
            query: optional STAC query dict to include in searches.
            sort_by: sort STAC results by this property.
            sort_ascending: whether the sort should be ascending.
            timeout: timeout for auth and asset requests.
            earthdata_token: optional Earthdata bearer token override.
            s3_credentials_url: LP DAAC temporary credentials endpoint.
            context: optional datasource context from rslearn.
        """
        super().__init__(
            band_names=band_names,
            query=query,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
            timeout=timeout,
            earthdata_token=earthdata_token,
            s3_credentials_url=s3_credentials_url,
            context=context,
            # ECOSTRESS asset keys are path-like, so we cannot filter by band-named
            # asset keys in the STAC search; _should_include_item handles it instead.
            require_asset_filter=False,
        )

    @override
    def _stac_item_to_item(self, stac_item: StacItem) -> SourceItem:
        if stac_item.geometry is None:
            raise ValueError("got unexpected item with no geometry")
        if stac_item.time_range is None:
            raise ValueError("got unexpected item with no time range")
        if stac_item.assets is None:
            raise ValueError("got unexpected item with no assets")

        shp = shapely.geometry.shape(stac_item.geometry)
        geometry = STGeometry(WGS84_PROJECTION, shp, stac_item.time_range)

        asset_urls: dict[str, str] = {}
        properties: dict[str, Any] = {}
        for band in self.asset_bands:
            suffix = f"_{band}"
            for asset_key, asset in stac_item.assets.items():
                if asset_key.startswith("s3_"):
                    # Prefer direct S3 access when available.
                    if asset_key.endswith(suffix):
                        asset_urls[band] = asset.href
                elif asset_key.endswith(suffix):
                    properties[f"{_HTTP_URL_PROPERTY_PREFIX}{band}"] = asset.href
                    asset_urls.setdefault(band, asset.href)

        for prop_name in self.properties_to_record:
            if prop_name in stac_item.properties:
                properties[prop_name] = stac_item.properties[prop_name]

        return SourceItem(stac_item.id, geometry, asset_urls, properties)

    @override
    def _should_include_item(self, item: SourceItem) -> bool:
        return all(band in item.asset_urls for band in self.asset_bands)
