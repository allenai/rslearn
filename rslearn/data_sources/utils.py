"""Utilities shared by data sources."""

from dataclasses import dataclass
from datetime import UTC, datetime
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


def match_candidate_items_to_window(
    geometry: STGeometry, items: list[ItemType], query_config: QueryConfig
) -> list[list[ItemType]]:
    """Match candidate items to a window based on the query configuration.

    Candidate items should be collected that intersect with the window's spatial
    extent.

    Args:
        geometry: the window projected to the same projection as the items
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

    groups = []

    if query_config.space_mode == SpaceMode.CONTAINS:
        for item, item_shp in zip(items, item_shps):
            if not item_shp.contains(geometry.shp):
                continue
            groups.append([item])
            if len(groups) >= query_config.max_matches:
                break

    elif query_config.space_mode == SpaceMode.INTERSECTS:
        for item, item_shp in zip(items, item_shps):
            if not shp_intersects(item_shp, geometry.shp):
                continue
            groups.append([item])
            if len(groups) >= query_config.max_matches:
                break

    elif query_config.space_mode == SpaceMode.MOSAIC:
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
                    pending_mosaic.remainder.area / geometry.shp.area
                    < MOSAIC_REMAINDER_EPSILON
                ):
                    pending_mosaic.completed = True

                break

            if item_matched:
                continue

            # See if we can add a new mosaic based on this item. There must be room for
            # more mosaics, but the item must also intersect the requested geometry.
            if len(pending_mosaics) >= query_config.max_matches:
                continue
            intersect_area = item_shp.intersection(geometry.shp).area
            if (
                intersect_area / item_shp.area < MOSAIC_MIN_ITEM_COVERAGE
                and intersect_area / geometry.shp.area < MOSAIC_MIN_ITEM_COVERAGE
            ):
                continue

            pending_mosaics.append(
                PendingMosaic(
                    items=[item],
                    remainder=geometry.shp - item_shp,
                )
            )

        for pending_mosaic in pending_mosaics:
            groups.append(pending_mosaic.items)

    return groups
