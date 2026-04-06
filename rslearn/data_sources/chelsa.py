"""Data source for CHELSA-daily climate rasters."""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, date, datetime, timedelta
from typing import Any

import requests
from typing_extensions import override
from upath import UPath

from rslearn.config import QueryConfig, SpaceMode
from rslearn.data_sources.data_source import (
    DataSourceContext,
    Item,
    ItemLookupDataSource,
)
from rslearn.data_sources.direct_materialize_data_source import (
    DirectMaterializeDataSource,
)
from rslearn.data_sources.utils import MatchedItemGroup, match_candidate_items_to_window
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.geometry import STGeometry, get_global_geometry


class CHELSADailyItem(Item):
    """An item representing one CHELSA daily timestep."""

    def __init__(self, name: str, geometry: STGeometry, item_date: date) -> None:
        """Create a new CHELSADaily item.

        Args:
            name: unique item name.
            geometry: item geometry.
            item_date: UTC calendar date represented by this item.
        """
        super().__init__(name=name, geometry=geometry)
        self.item_date = item_date

    @override
    def serialize(self) -> dict:
        d = super().serialize()
        d["item_date"] = self.item_date.isoformat()
        return d

    @staticmethod
    def deserialize(d: dict[str, Any]) -> CHELSADailyItem:
        """Deserialize from JSON-decoded dictionary."""
        item = Item.deserialize(d)
        return CHELSADailyItem(
            name=item.name,
            geometry=item.geometry,
            item_date=date.fromisoformat(d["item_date"]),
        )


class CHELSADaily(
    DirectMaterializeDataSource[CHELSADailyItem], ItemLookupDataSource[CHELSADailyItem]
):
    """Data source for CHELSA-daily global climate rasters.

    URL pattern:
    ``https://os.unil.cloud.switch.ch/chelsa02/chelsa/{extent}/daily/{variable}/{year}/CHELSA_{variable}_{day}_{month}_{year}_{version}.tif``
    """

    BASE_URL = "https://os.unil.cloud.switch.ch/chelsa02/chelsa"
    DEFAULT_START_DATE = date(1979, 1, 1)
    DEFAULT_END_DATE = date(2025, 8, 29)
    DEFAULT_VERSION = "V.2.1"
    # CHELSA daily uses both "pr" and "prec" for precipitation:
    # - "pr" is used before 2020 and overlaps in 2020.
    # - "prec" is used after 2020 and overlaps in 2020.
    PRECIPITATION_ALIASES = {"pr", "prec"}
    PRECIPITATION_OVERLAP_YEAR = 2020
    ALLOWED_VARIABLES = {
        "clt",
        "hurs",
        "pr",
        "prec",
        "ps",
        "rsds",
        "sfcWind",
        "tas",
        "tasmax",
        "tasmin",
        "tz",
    }

    def __init__(
        self,
        band_names: list[str] | None = None,
        extent: str = "global",
        start_date: date | str = DEFAULT_START_DATE,
        end_date: date | str = DEFAULT_END_DATE,
        base_url: str = BASE_URL,
        version: str = DEFAULT_VERSION,
        timeout: timedelta = timedelta(seconds=60),
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Create a new CHELSADaily data source.

        Args:
            band_names: CHELSA variable names (e.g. "tas", "pr"). If omitted and
                context.layer_config is present, uses the unique bands from the layer.
            extent: CHELSA extent in URL path, usually "global".
            start_date: earliest available date (inclusive).
            end_date: latest available date (inclusive).
            base_url: CHELSA base URL.
            version: filename version segment, e.g. "V.2.1".
            timeout: HTTP timeout for ingest downloads.
            context: data source context.
        """
        self.extent = extent
        self.base_url = base_url.rstrip("/")
        self.version = version
        self.timeout = timeout

        self.start_date = self._parse_date(start_date)
        self.end_date = self._parse_date(end_date)
        if self.end_date < self.start_date:
            raise ValueError("end_date must be >= start_date")

        self.band_names: list[str]
        if context.layer_config is not None:
            self.band_names = []
            for band_set in context.layer_config.band_sets:
                for band in band_set.bands:
                    if band not in self.band_names:
                        self.band_names.append(band)
        elif band_names is not None:
            self.band_names = list(band_names)
        else:
            raise ValueError(
                "band_names must be set if layer_config is not in the context"
            )

        invalid = [b for b in self.band_names if b not in self.ALLOWED_VARIABLES]
        if invalid:
            raise ValueError(
                f"unsupported CHELSA daily variable(s): {invalid}; "
                f"supported: {sorted(self.ALLOWED_VARIABLES)}"
            )

        super().__init__(asset_bands={band: [band] for band in self.band_names})

    @staticmethod
    def _parse_date(value: date | str) -> date:
        if isinstance(value, date):
            return value
        return date.fromisoformat(value)

    @staticmethod
    def _to_utc(t: datetime) -> datetime:
        if t.tzinfo is None:
            return t.replace(tzinfo=UTC)
        return t.astimezone(UTC)

    @classmethod
    def _item_name_for_date(cls, d: date) -> str:
        return f"chelsa_daily_{d.strftime('%Y%m%d')}"

    @staticmethod
    def _date_from_item_name(name: str) -> date:
        prefix = "chelsa_daily_"
        if not name.startswith(prefix):
            raise ValueError(f"invalid CHELSA item name {name!r}")
        suffix = name[len(prefix) :]
        if len(suffix) != 8 or not suffix.isdigit():
            raise ValueError(f"invalid CHELSA item name {name!r}")
        return date.fromisoformat(f"{suffix[0:4]}-{suffix[4:6]}-{suffix[6:8]}")

    def _build_item(self, item_date: date) -> CHELSADailyItem:
        start = datetime(item_date.year, item_date.month, item_date.day, tzinfo=UTC)
        end = start + timedelta(days=1)
        return CHELSADailyItem(
            name=self._item_name_for_date(item_date),
            geometry=get_global_geometry((start, end)),
            item_date=item_date,
        )

    @classmethod
    def _resolve_variable_for_date(
        cls, requested_variable: str, item_date: date
    ) -> str:
        """Resolve CHELSA variable alias for a specific date.

        CHELSA precipitation changed variable naming from ``pr`` to ``prec`` with
        overlap in 2020. Users may configure either alias once, and this method picks
        the correct URL variable automatically by year.
        """
        if requested_variable not in cls.PRECIPITATION_ALIASES:
            return requested_variable

        if item_date.year < cls.PRECIPITATION_OVERLAP_YEAR:
            return "pr"
        if item_date.year > cls.PRECIPITATION_OVERLAP_YEAR:
            return "prec"
        return requested_variable

    @override
    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[MatchedItemGroup[CHELSADailyItem]]]:
        if query_config.space_mode != SpaceMode.SINGLE_COMPOSITE:
            raise ValueError(
                "CHELSADaily expects SINGLE_COMPOSITE space mode in the query configuration"
            )

        dataset_start = datetime(
            self.start_date.year,
            self.start_date.month,
            self.start_date.day,
            tzinfo=UTC,
        )
        dataset_end_exclusive = datetime(
            self.end_date.year,
            self.end_date.month,
            self.end_date.day,
            tzinfo=UTC,
        ) + timedelta(days=1)

        groups: list[list[MatchedItemGroup[CHELSADailyItem]]] = []
        for geometry in geometries:
            if geometry.time_range is None:
                raise ValueError("expected all geometries to have a time range")

            request_start = self._to_utc(geometry.time_range[0])
            request_end = self._to_utc(geometry.time_range[1])
            clipped_start = max(request_start, dataset_start)
            clipped_end = min(request_end, dataset_end_exclusive)

            if clipped_start >= clipped_end:
                groups.append([MatchedItemGroup(items=[], request_time_range=None)])
                continue

            day_cursor = datetime(
                clipped_start.year,
                clipped_start.month,
                clipped_start.day,
                tzinfo=UTC,
            )

            items: list[CHELSADailyItem] = []
            while day_cursor < clipped_end:
                items.append(self._build_item(day_cursor.date()))
                day_cursor += timedelta(days=1)

            groups.append(
                match_candidate_items_to_window(geometry, items, query_config)
            )

        return groups

    @override
    def deserialize_item(self, serialized_item: dict) -> CHELSADailyItem:
        return CHELSADailyItem.deserialize(serialized_item)

    @override
    def get_item_by_name(self, name: str) -> CHELSADailyItem:
        d = self._date_from_item_name(name)
        if d < self.start_date or d > self.end_date:
            raise ValueError(
                f"item {name!r} is outside configured dataset range "
                f"{self.start_date.isoformat()}..{self.end_date.isoformat()}"
            )
        return self._build_item(d)

    @override
    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[CHELSADailyItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        for item in items:
            for asset_key, band_names in self.asset_bands.items():
                if tile_store.is_raster_ready(item, band_names):
                    continue

                asset_url = self.get_asset_url(item, asset_key)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    local_fname = os.path.join(
                        tmp_dir,
                        f"{item.name}_{asset_key}.tif",
                    )
                    with requests.get(
                        asset_url, stream=True, timeout=self.timeout.total_seconds()
                    ) as r:
                        r.raise_for_status()
                        with open(local_fname, "wb") as f:
                            for chunk in r.iter_content(chunk_size=1024 * 1024):
                                f.write(chunk)

                    tile_store.write_raster_file(
                        item,
                        band_names,
                        UPath(local_fname),
                        time_range=item.geometry.time_range,
                    )

    @override
    def get_asset_url(self, item: CHELSADailyItem, asset_key: str) -> str:
        if asset_key not in self.asset_bands:
            raise KeyError(
                f"unknown CHELSA asset {asset_key!r}; "
                f"known={sorted(self.asset_bands.keys())}"
            )

        d = item.item_date
        yyyy = f"{d.year:04d}"
        mm = f"{d.month:02d}"
        dd = f"{d.day:02d}"
        resolved_variable = self._resolve_variable_for_date(asset_key, d)

        return (
            f"{self.base_url}/{self.extent}/daily/{resolved_variable}/{yyyy}/"
            f"CHELSA_{resolved_variable}_{dd}_{mm}_{yyyy}_{self.version}.tif"
        )
