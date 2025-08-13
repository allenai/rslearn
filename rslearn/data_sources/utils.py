"""Utilities shared by data sources."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TypeVar

import shapely

from rslearn.config import QueryConfig, SpaceMode, TimeMode
from rslearn.data_sources import Item
from rslearn.utils import STGeometry, shp_intersects

MOSAIC_MIN_ITEM_COVERAGE = 0.1
"""Minimum fraction of area that item should cover when adding it to a mosaic group."""

MOSAIC_REMAINDER_EPSILON = 0.01
"""Fraction of original geometry area below which mosaic is considered to contain the
entire geometry."""

ItemType = TypeVar("ItemType", bound=Item)


@dataclass
class PendingMosaic:
    """A mosaic being created by match_candidate_items_to_window.

    Args:
        items: the list of items in the mosaic.
        remainder: the remainder of the geometry that is not covered by any of the
            items.
        completed: whether the mosaic is done (sufficient coverage of the geometry).
    """

    # Cannot use list[ItemType] here.
    items: list
    remainder: shapely.Polygon
    completed: bool = False


def mosaic_matching(
    window_geometry: STGeometry,
    items: list[ItemType],
    item_shps: list[shapely.Geometry],
    max_matches: int,
) -> list[list[ItemType]]:
    """Spatial item matching for mosaic space mode.

    This attempts to piece together items into mosaics that fully cover the window
    geometry. If there are items leftover that only partially cover the window
    geometry, they are returned as partial mosaics.

    Args:
        window_geometry: the geometry of the window.
        items: list of items.
        item_shps: the item shapes projected to the window's projection.
        max_matches: the maximum number of matches (mosaics) to create.

    Returns:
        list of item groups, each one corresponding to a different mosaic.
    """
    # To create mosaics, we iterate over the items in order, and add each item to
    # the first mosaic that the new item adds coverage to.

    # max_matches could be very high if the user just wants us to create as many
    # mosaics as possible, so we initialize the list here as empty and just add
    # more pending mosaics when it is necessary.
    pending_mosaics: list[PendingMosaic] = []

    for item, item_shp in zip(items, item_shps):
        # See if the item can match with any existing mosaic.
        item_matched = False

        for pending_mosaic in pending_mosaics:
            if pending_mosaic.completed:
                continue
            if not shp_intersects(item_shp, pending_mosaic.remainder):
                continue

            # Check if the intersection area is too small.
            # If it is a sizable part of the item or of the geometry, then proceed.
            intersect_area = item_shp.intersection(pending_mosaic.remainder).area
            if (
                intersect_area / item_shp.area < MOSAIC_MIN_ITEM_COVERAGE
                and intersect_area / pending_mosaic.remainder.area
                < MOSAIC_MIN_ITEM_COVERAGE
            ):
                continue

            pending_mosaic.remainder = pending_mosaic.remainder - item_shp
            pending_mosaic.items.append(item)
            item_matched = True

            # Mark the mosaic completed if it has sufficient coverage of the
            # geometry.
            if (
                pending_mosaic.remainder.area / window_geometry.shp.area
                < MOSAIC_REMAINDER_EPSILON
            ):
                pending_mosaic.completed = True

            break

        if item_matched:
            continue

        # See if we can add a new mosaic based on this item. There must be room for
        # more mosaics, but the item must also intersect the requested geometry.
        if len(pending_mosaics) >= max_matches:
            continue
        intersect_area = item_shp.intersection(window_geometry.shp).area
        if (
            intersect_area / item_shp.area < MOSAIC_MIN_ITEM_COVERAGE
            and intersect_area / window_geometry.shp.area < MOSAIC_MIN_ITEM_COVERAGE
        ):
            continue

        pending_mosaics.append(
            PendingMosaic(
                items=[item],
                remainder=window_geometry.shp - item_shp,
            )
        )

    return [pending_mosaic.items for pending_mosaic in pending_mosaics]


def per_period_mosaic_matching(
    window_geometry: STGeometry,
    item_list: list[ItemType],
    period_duration: timedelta,
    max_matches: int,
) -> list[list[ItemType]]:
    """Match items to the geometry with one mosaic per period.

    We divide the time range of the geometry into shorter periods. Within each period,
    we use the items corresponding to that period to create a mosaic. The returned item
    groups include one group per period, starting from the most recent periods, up to
    the provided max_matches.

    The periods are also bounded to the window's time range, and aligned with the end
    of that time range, i.e. the most recent window is
    (end_time - period_duration, end_time), the next is
    (end_time - 2*period_duration, end_time - period_duration), and so on. Note that
    this means that if the window duration is shorter than the period_duration, there
    will be zero matches.

    This is used e.g. when a model should process three mosaics, where each mosaic
    should come from a different month. This gives more diversity of images, since
    simply searching for the least cloudy images could result in selecting all of the
    images from the same month.

    max_matches may be smaller than the total number of periods in the given time
    range. In this case, we prefer to use mosaics of the most recent periods. However,
    sometimes there may be no items in a period; in that case, the older periods are
    used as a fallback. This means that reducing the window duration down to match
    max_matches*period_duration is not equivalent to a longer window duration.

    Args:
        window_geometry: the window geometry to match items to.
        item_list: the list of items.
        period_duration: the duration of one period.
        max_matches: the number of per-period mosaics to create.

    Returns:
        the matched item groups, where each group contains items that yield a
            per-period mosaic.
    """
    if window_geometry.time_range is None:
        raise ValueError(
            "all windows must have time range for per period mosaic matching"
        )

    # For each period, we create an STGeometry with modified time range matching that
    # period, and use it with match_candidate_items_to_window to get a mosaic.
    cur_groups: list[list[ItemType]] = []
    period_start = window_geometry.time_range[1] - period_duration
    while (
        period_start >= window_geometry.time_range[0] and len(cur_groups) < max_matches
    ):
        period_time_range = (
            period_start,
            period_start + period_duration,
        )
        period_start -= period_duration
        period_geom = STGeometry(
            window_geometry.projection, window_geometry.shp, period_time_range
        )

        # We modify the QueryConfig here since caller should be asking for
        # multiple mosaics, but we just want one mosaic per period.
        period_groups = match_candidate_items_to_window(
            period_geom,
            item_list,
            QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1),
        )

        # There should be zero or one group depending on whether there were
        # any items that matched. We keep the group if it is there.
        if len(period_groups) == 0 or len(period_groups[0]) == 0:
            # No matches for this period.
            continue
        cur_groups.append(period_groups[0])

    return cur_groups


def match_candidate_items_to_window(
    geometry: STGeometry, items: list[ItemType], query_config: QueryConfig
) -> list[list[ItemType]]:
    """Match candidate items to a window based on the query configuration.

    Candidate items should be collected that intersect with the window's spatial
    extent.

    Args:
        geometry: the window's geometry
        items: all items from the data source that intersect spatially with the geometry
        query_config: the query configuration to use for matching

    Returns:
        list of matched item groups.
    """
    # Use time mode to filter and order the items.
    if geometry.time_range:
        items = [
            item
            for item in items
            if geometry.intersects_time_range(item.geometry.time_range)
        ]

        placeholder_datetime = datetime.now(timezone.utc)
        if query_config.time_mode == TimeMode.BEFORE:
            items.sort(
                key=lambda item: item.geometry.time_range[0]
                if item.geometry.time_range
                else placeholder_datetime,
                reverse=True,
            )
        elif query_config.time_mode == TimeMode.AFTER:
            items.sort(
                key=lambda item: item.geometry.time_range[0]
                if item.geometry.time_range
                else placeholder_datetime,
                reverse=False,
            )

    # Now apply space mode.
    item_shps = []
    for item in items:
        item_geom = item.geometry
        # We need to re-project items to the geometry projection for the spatial checks
        # below. Unless the item's geometry indicates global coverage, in which case we
        # set it to match the geometry to show that it should cover the entire
        # geometry.
        if item_geom.projection != geometry.projection:
            if item_geom.is_global():
                item_geom = geometry
            else:
                item_geom = item_geom.to_projection(geometry.projection)
        item_shps.append(item_geom.shp)

    if query_config.space_mode == SpaceMode.CONTAINS:
        groups = []
        for item, item_shp in zip(items, item_shps):
            if not item_shp.contains(geometry.shp):
                continue
            groups.append([item])
            if len(groups) >= query_config.max_matches:
                break

    elif query_config.space_mode == SpaceMode.INTERSECTS:
        groups = []
        for item, item_shp in zip(items, item_shps):
            if not shp_intersects(item_shp, geometry.shp):
                continue
            groups.append([item])
            if len(groups) >= query_config.max_matches:
                break

    elif query_config.space_mode == SpaceMode.MOSAIC:
        groups = mosaic_matching(geometry, items, item_shps, query_config.max_matches)

    elif query_config.space_mode == SpaceMode.PER_PERIOD_MOSAIC:
        groups = per_period_mosaic_matching(
            geometry, items, query_config.period_duration, query_config.max_matches
        )

    elif query_config.space_mode == SpaceMode.COMPOSITE:
        group = []
        for item, item_shp in zip(items, item_shps):
            if not shp_intersects(item_shp, geometry.shp):
                continue
            group.append(item)
        groups = [group]

    else:
        raise ValueError(f"invalid space mode {query_config.space_mode}")

    # Enforce minimum matches if set.
    if len(groups) < query_config.min_matches:
        return []

    return groups
