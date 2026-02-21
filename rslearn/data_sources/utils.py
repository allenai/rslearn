"""Utilities shared by data sources."""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TypeVar

import shapely

from rslearn.config import QueryConfig, SpaceMode, TimeMode
from rslearn.data_sources import Item
from rslearn.log_utils import get_logger
from rslearn.utils import STGeometry, shp_intersects

logger = get_logger(__name__)

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


def _create_single_coverage_mosaics(
    window_geometry: STGeometry,
    items: list[ItemType],
    item_shps: list[shapely.Geometry],
    max_mosaics: int,
) -> list[list[ItemType]]:
    """Create mosaics where each mosaic covers the window geometry once.

    This attempts to piece together items into mosaics that fully cover the window
    geometry. If there are items leftover that only partially cover the window
    geometry, they are returned as partial mosaics.

    Args:
        window_geometry: the geometry of the window.
        items: list of items.
        item_shps: the item shapes projected to the window's projection.
        max_mosaics: the maximum number of mosaics to create.

    Returns:
        list of item groups, each one corresponding to a different single-coverage
        mosaic.
    """
    # To create mosaics, we iterate over the items in order, and add each item to
    # the first mosaic that the new item adds coverage to.

    # max_mosaics could be very high if the user just wants us to create as many
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
        if len(pending_mosaics) >= max_mosaics:
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


def _consolidate_mosaics_by_overlaps(
    mosaics: list[list[ItemType]],
    overlaps: int,
    max_groups: int,
) -> list[list[ItemType]]:
    """Consolidate single-coverage mosaics into groups based on desired overlaps.

    Args:
        mosaics: list of single-coverage mosaics (each mosaic is a list of items).
        overlaps: the number of overlapping coverages wanted per group.
        max_groups: the maximum number of groups to return.

    Returns:
        list of item groups, where each group contains items from multiple mosaics
        to achieve the desired number of overlapping coverages.
    """
    if overlaps <= 0:
        overlaps = 1

    groups: list[list[ItemType]] = []
    for i in range(0, len(mosaics), overlaps):
        if len(groups) >= max_groups:
            break
        # Combine overlaps consecutive mosaics into one group
        combined_items: list[ItemType] = []
        for mosaic in mosaics[i : i + overlaps]:
            combined_items.extend(mosaic)
        if combined_items:
            groups.append(combined_items)

    return groups


def match_with_space_mode_contains(
    geometry: STGeometry,
    items: list[ItemType],
    item_shps: list[shapely.Geometry],
    query_config: QueryConfig,
) -> list[list[ItemType]]:
    """Match items that fully contain the window geometry.

    Args:
        geometry: the window's geometry.
        items: list of items.
        item_shps: the item shapes projected to the window's projection.
        query_config: the query configuration.

    Returns:
        list of matched item groups, where each group contains a single item.
    """
    groups: list[list[ItemType]] = []
    for item, item_shp in zip(items, item_shps):
        if not item_shp.contains(geometry.shp):
            continue
        groups.append([item])
        if len(groups) >= query_config.max_matches:
            break
    return groups


def match_with_space_mode_intersects(
    geometry: STGeometry,
    items: list[ItemType],
    item_shps: list[shapely.Geometry],
    query_config: QueryConfig,
) -> list[list[ItemType]]:
    """Match items that intersect any portion of the window geometry.

    Args:
        geometry: the window's geometry.
        items: list of items.
        item_shps: the item shapes projected to the window's projection.
        query_config: the query configuration.

    Returns:
        list of matched item groups, where each group contains a single item.
    """
    groups: list[list[ItemType]] = []
    for item, item_shp in zip(items, item_shps):
        if not shp_intersects(item_shp, geometry.shp):
            continue
        groups.append([item])
        if len(groups) >= query_config.max_matches:
            break
    return groups


def match_with_space_mode_mosaic(
    geometry: STGeometry,
    items: list[ItemType],
    item_shps: list[shapely.Geometry],
    query_config: QueryConfig,
) -> list[list[ItemType]]:
    """Match items into mosaic groups that cover the window geometry.

    Creates groups of items that together cover the window geometry. The number of
    overlapping coverages in each group is controlled by mosaic_compositing_overlaps.

    Args:
        geometry: the window's geometry.
        items: list of items.
        item_shps: the item shapes projected to the window's projection.
        query_config: the query configuration.

    Returns:
        list of matched item groups, where each group forms a mosaic covering the
        window.
    """
    overlaps = query_config.mosaic_compositing_overlaps

    # Calculate how many single-coverage mosaics we need to create.
    # We need enough mosaics to consolidate into max_matches groups with the
    # desired number of overlaps per group.
    max_single_mosaics = query_config.max_matches * overlaps

    # Create single-coverage mosaics
    single_mosaics = _create_single_coverage_mosaics(
        geometry, items, item_shps, max_single_mosaics
    )

    # Consolidate into groups based on overlaps
    return _consolidate_mosaics_by_overlaps(
        single_mosaics, overlaps, query_config.max_matches
    )


def match_with_space_mode_single_composite(
    geometry: STGeometry,
    items: list[ItemType],
    item_shps: list[shapely.Geometry],
    query_config: QueryConfig,
) -> list[list[ItemType]]:
    """Match items for SINGLE_COMPOSITE.

    All spatially-intersecting items go into a single group so that one composite can
    be created from all matching items.

    Args:
        geometry: the window's geometry.
        items: list of items.
        item_shps: the item shapes projected to the window's projection.
        query_config: the query configuration.

    Returns:
        list containing a single item group of all spatially-intersecting items,
        or empty list if no items intersect.
    """
    group_items: list[ItemType] = []
    for item, item_shp in zip(items, item_shps):
        if shp_intersects(item_shp, geometry.shp):
            group_items.append(item)
    return [group_items] if group_items else []


# Type alias for space mode handler functions
SpaceModeHandler = Callable[
    [STGeometry, list[ItemType], list[shapely.Geometry], QueryConfig],
    list[list[ItemType]],
]

# Dict mapping SpaceMode values to their handler functions.
# PER_PERIOD_MOSAIC is deprecated; it reuses the MOSAIC handler and period splitting
# is handled in match_candidate_items_to_window.
space_mode_handlers: dict[SpaceMode, SpaceModeHandler] = {
    SpaceMode.CONTAINS: match_with_space_mode_contains,
    SpaceMode.INTERSECTS: match_with_space_mode_intersects,
    SpaceMode.MOSAIC: match_with_space_mode_mosaic,
    SpaceMode.PER_PERIOD_MOSAIC: match_with_space_mode_mosaic,
    SpaceMode.SINGLE_COMPOSITE: match_with_space_mode_single_composite,
}


def _filter_and_project_items(
    geometry: STGeometry, items: list[ItemType], query_config: QueryConfig
) -> tuple[list[ItemType], list[shapely.Geometry]]:
    """Filter items by time and project to geometry's projection.

    Returns:
        tuple of (acceptable_items, acceptable_item_shps)
    """
    # Use time mode to filter and order the items.
    if geometry.time_range:
        items = [
            item
            for item in items
            if geometry.intersects_time_range(item.geometry.time_range)
        ]

        placeholder_datetime = datetime.now(UTC)
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

    # Project items to geometry's projection.
    acceptable_items: list[ItemType] = []
    acceptable_item_shps: list[shapely.Geometry] = []
    for item in items:
        item_geom = item.geometry
        # We need to re-project items to the geometry projection for the spatial checks
        # later. Unless the item's geometry indicates global coverage, in which case we
        # set it to match the geometry to show that it should cover the entire
        # geometry.
        if item_geom.projection != geometry.projection:
            if item_geom.is_global():
                item_geom = geometry
            else:
                item_geom = item_geom.to_projection(geometry.projection)

        if item_geom.shp.area == 0:
            # Must have been an item that didn't quite match the window's spatial extent.
            continue

        acceptable_items.append(item)
        acceptable_item_shps.append(item_geom.shp)

    return acceptable_items, acceptable_item_shps


def match_candidate_items_to_window(
    geometry: STGeometry, items: list[ItemType], query_config: QueryConfig
) -> list[list[ItemType]]:
    """Match candidate items to a window based on the query configuration.

    If ``period_duration`` is set, the window time range is split into sub-periods
    and the handler is applied per-period with effective max_matches=1.

    When ``period_duration`` is set and ``per_period_mosaic_reverse_time_order``
    is True (the current default), the resulting groups are reversed so that the
    most recent period comes first. This default will change to False after
    2026-04-01.

    Args:
        geometry: the window's geometry
        items: all items from the data source that intersect spatially with the geometry
        query_config: the query configuration to use for matching

    Returns:
        list of matched item groups.
    """
    # PER_PERIOD_MOSAIC should default to 30-day periods in case period_duration is not
    # set, since period_duration previously applied only for PER_PERIOD_MOSAIC with
    # default 30 day duration.
    period_duration = query_config.period_duration
    if (
        query_config.space_mode == SpaceMode.PER_PERIOD_MOSAIC
        and period_duration is None
    ):
        period_duration = timedelta(days=30)

    # Filter items by time and project them into the geometry's projection.
    acceptable_items, acceptable_item_shps = _filter_and_project_items(
        geometry, items, query_config
    )

    handler = space_mode_handlers.get(query_config.space_mode)
    if handler is None:
        raise ValueError(f"invalid space mode {query_config.space_mode}")

    # Handle period_duration if set. This causes the space_mode_handler to be called
    # once for each period within the window time range. In this case max_matches
    # controls the number of periods, while each handler creates at most one item
    # group.
    if period_duration is not None:
        if geometry.time_range is None:
            raise ValueError("period_duration is set but geometry has no time_range")
        per_period_query_config = QueryConfig(
            space_mode=query_config.space_mode,
            max_matches=1,
            mosaic_compositing_overlaps=query_config.mosaic_compositing_overlaps,
        )

        # Iterate from most recent period backwards so that when max_matches
        # truncates, we keep the most recent periods.
        groups: list[list[ItemType]] = []
        period_end = geometry.time_range[1]
        while (
            period_end - period_duration >= geometry.time_range[0]
            and len(groups) < query_config.max_matches
        ):
            period_start = period_end - period_duration
            period_geom = STGeometry(
                geometry.projection, geometry.shp, (period_start, period_end)
            )
            period_end = period_start

            # Re-filter items to this period.
            period_items, period_shps = _filter_and_project_items(
                period_geom, items, per_period_query_config
            )
            period_groups = handler(
                period_geom, period_items, period_shps, per_period_query_config
            )
            if period_groups:
                groups.append(period_groups[0])

        # Groups are in reverse chronological order. Reverse to chronological
        # unless the deprecated per_period_mosaic_reverse_time_order is True.
        if query_config.per_period_mosaic_reverse_time_order:
            warnings.warn(
                "QueryConfig.per_period_mosaic_reverse_time_order defaults to True, "
                "which returns item groups in reverse temporal order (most recent "
                "first) when period_duration is set. This default will change to "
                "False (chronological order) after 2026-04-01. To silence this "
                "warning, explicitly set per_period_mosaic_reverse_time_order=False.",
                FutureWarning,
                stacklevel=3,
            )
        else:
            groups.reverse()
    else:
        groups = handler(geometry, acceptable_items, acceptable_item_shps, query_config)

    # Enforce minimum matches if set.
    if len(groups) < query_config.min_matches:
        logger.warning(
            "Window rejected: found %d matches (required: %d) for time range %s",
            len(groups),
            query_config.min_matches,
            geometry.time_range if geometry.time_range else "unlimited",
        )
        return []

    return groups
