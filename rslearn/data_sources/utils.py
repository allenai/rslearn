from rslearn.config import QueryConfig, SpaceMode, TimeMode
from rslearn.data_sources import Item
from rslearn.utils import STGeometry

MOSAIC_MIN_ITEM_COVERAGE = 0.1
"""Minimum fraction of area that item should cover when adding it to a mosaic group."""


def match_candidate_items_to_window(
    geometry: STGeometry, items: list[Item], query_config: QueryConfig
) -> list[list[Item]]:
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
    item_shps = []
    for item in items:
        item_geom = item.geometry
        if item_geom.projection != geometry.projection:
            item_geom = item_geom.to_projection(geometry.projection)
        item_shps.append(item_geom.shp)

    # Use time mode to filter and order the items.
    if geometry.time_range:
        if query_config.time_mode == TimeMode.WITHIN:
            items = [
                item
                for item in items
                if geometry.intersects_time_range(item.geometry.time_range)
            ]
        elif query_config.time_mode in [
            TimeMode.NEAREST,
            TimeMode.BEFORE,
            TimeMode.AFTER,
        ]:
            if query_config.time_mode == TimeMode.BEFORE:
                items = [
                    item
                    for item in items
                    if not item.time_range or item.range[1] <= geometry.time_range[0]
                ]
            elif query_config.time_mode == TimeMode.AFTER:
                items = [
                    item
                    for item in items
                    if not item.time_range
                    or item.time_range[0] >= geometry.time_range[1]
                ]
            items.sort(
                key=lambda item: geometry.distance_to_time_range(item.time_range)
            )

    # Now apply space mode.
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
            if not item_shp.intersects(geometry.shp):
                continue
            groups.append([item])
            if len(groups) >= query_config.max_matches:
                break

    elif query_config.space_mode == SpaceMode.MOSAIC:
        # To create mosaic groups, we repeatedly try to fill the geometry.
        # Each time the geometry is fully covered, we start another group.
        # We terminate when there are no more items or we have exceeded max_matches.
        cur_remainder = None
        cur_group = []

        for item, item_shp in zip(items, item_shps):
            if cur_remainder is None:
                cur_remainder = geometry.shp

            if not item_shp.intersects(cur_remainder):
                continue

            # Check if the intersection area is too small.
            # If it is a sizable part of the item or of the geometry, then continue.
            intersect_area = item_shp.intersection(cur_remainder).area
            if (
                intersect_area / item_shp.area < MOSAIC_MIN_ITEM_COVERAGE
                and intersect_area / cur_remainder.area < MOSAIC_MIN_ITEM_COVERAGE
            ):
                continue

            cur_remainder = cur_remainder - item_shp
            cur_group.append(item)

            if cur_remainder.area <= 0:
                cur_remainder = None
                groups.append(cur_group)
                cur_group = []

            if len(groups) >= query_config.max_matches:
                break

        if len(cur_group) > 0:
            groups.append(cur_group)

    return groups
