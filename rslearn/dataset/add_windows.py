"""Functions to add windows to an rslearn dataset."""

from datetime import datetime

import fiona
import shapely
import shapely.geometry
import tqdm
from rasterio.crs import CRS

from rslearn.const import WGS84_PROJECTION
from rslearn.utils import PixelBounds, Projection, STGeometry, get_utm_ups_crs

from .dataset import Dataset
from .window import Window


def add_windows_from_geometries(
    dataset: Dataset,
    group: str,
    geometries: list[STGeometry],
    projection: Projection,
    name: str | None = None,
    grid_size: int | None = None,
    window_size: int | None = None,
    time_range: tuple[datetime, datetime] | None = None,
    use_utm: bool = False,
) -> list[Window]:
    """Create windows based on a list of STGeometry.

    Args:
        dataset: the dataset to add the windows to.
        group: the group to add the windows to.
        geometries: list of STGeometry.
        projection: the projection of the output windows.
        name: optional name of the output window (or prefix if grid_size is set).
        grid_size: if None (default), create one window corresponding to the box
            (re-projected if needed). Otherwise, create windows corresponding to cells
            of a grid of this grid size, with one window at each cell that intersects
            the specified box.
        window_size: use box only to define center of the window, but make the window
            this size. Only one of grid_size and window_size can be specified.
        time_range: optional time range for the output windows, in case a geometry does
            not have a time range. If not specified, then the output window will not
            have a time range.
        use_utm: override output projection with an appropriate UTM projection

    Returns:
        list of newly created windows
    """
    # Get list of axis-aligned boxes and associated projection to create.
    out_box_list: list[tuple[PixelBounds, Projection]] = []
    for geometry in tqdm.tqdm(geometries):
        cur_projection = projection
        if use_utm:
            # Override the CRS in the specified projection with appropriate UTM CRS.
            wgs84_geom = geometry.to_projection(WGS84_PROJECTION)
            wgs84_point = wgs84_geom.shp.centroid
            utm_crs = get_utm_ups_crs(wgs84_point.x, wgs84_point.y)
            cur_projection = Projection(
                utm_crs, cur_projection.x_resolution, cur_projection.y_resolution
            )

        geometry = geometry.to_projection(cur_projection)
        shp = geometry.shp

        if grid_size:
            start_tile = (
                int(shp.bounds[0]) // grid_size,
                int(shp.bounds[1]) // grid_size,
            )
            end_tile = (
                int(shp.bounds[2] + grid_size - 1) // grid_size,
                int(shp.bounds[3] + grid_size - 1) // grid_size,
            )
            for col in range(start_tile[0], end_tile[0]):
                for row in range(start_tile[1], end_tile[1]):
                    out_box_list.append(
                        (
                            (
                                col * grid_size,
                                row * grid_size,
                                (col + 1) * grid_size,
                                (row + 1) * grid_size,
                            ),
                            cur_projection,
                        )
                    )
        elif window_size:
            centroid = shp.centroid
            out_box_list.append(
                (
                    (
                        int(centroid.x) - window_size // 2,
                        int(centroid.y) - window_size // 2,
                        int(centroid.x) + window_size // 2,
                        int(centroid.y) + window_size // 2,
                    ),
                    cur_projection,
                )
            )
        else:
            out_box_list.append(
                (
                    (
                        int(shp.bounds[0]),
                        int(shp.bounds[1]),
                        int(shp.bounds[2]),
                        int(shp.bounds[3]),
                    ),
                    cur_projection,
                )
            )

    # Create window for each computed box.
    windows: list[Window] = []
    for out_box, cur_projection in out_box_list:
        # Use name provided by user if possible.
        # If there are multiple boxes, we need to suffix by something,
        # so we use the topleft box coordinate.
        # If name is not specified, the default name includes both the spatial
        # coordinates, and the specified time range if any.
        if name and len(out_box_list) == 1:
            cur_window_name = name
        elif name:
            cur_window_name = f"{name}_{out_box[0]}_{out_box[1]}"
        else:
            cur_window_name = f"{out_box[0]}_{out_box[1]}_{out_box[2]}_{out_box[3]}"
            if time_range:
                cur_window_name += (
                    f"_{time_range[0].isoformat()}_{time_range[1].isoformat()}"
                )
        window = Window(
            path=dataset.path / "windows" / group / cur_window_name,
            group=group,
            name=cur_window_name,
            projection=cur_projection,
            bounds=out_box,
            time_range=time_range,
        )
        window.save()
        windows.append(window)

    return windows


def add_windows_from_box(
    dataset: Dataset,
    group: str,
    box: tuple[float, float, float, float],
    projection: Projection,
    src_projection: Projection | None = None,
    name: str | None = None,
    grid_size: int | None = None,
    window_size: int | None = None,
    time_range: tuple[datetime, datetime] | None = None,
    use_utm: bool = False,
) -> list[Window]:
    """Create windows based on the specified box.

    Args:
        dataset: the dataset to add the windows to.
        group: the group to add the windows to.
        box: an axis-aligned rectangle (x1, y1, x2, y2).
        projection: the projection of the output windows.
        src_projection: the projection of the specified box (defaults to projection)
        name: see add_windows_from_geometries
        grid_size: see add_windows_from_geometries
        window_size: see add_windows_from_geometries
        time_range: see add_windows_from_geometries
        use_utm: see add_windows_from_geometries

    Returns:
        list of newly created windows
    """
    # Get box in target projection (re-projecting if src_projection is set).
    if not src_projection:
        src_projection = projection
    geometry = STGeometry(src_projection, shapely.box(*box), None)

    return add_windows_from_geometries(
        dataset=dataset,
        group=group,
        projection=projection,
        name=name,
        grid_size=grid_size,
        window_size=window_size,
        time_range=time_range,
        use_utm=use_utm,
        geometries=[geometry],
    )


def add_windows_from_file(
    dataset: Dataset,
    group: str,
    fname: str,
    projection: Projection,
    name: str | None = None,
    grid_size: int | None = None,
    window_size: int | None = None,
    time_range: tuple[datetime, datetime] | None = None,
    use_utm: bool = False,
) -> list[Window]:
    """Create windows based on the specified vector file.

    Args:
        dataset: the dataset to add the windows to.
        group: the group to add the windows to.
        fname: vector filename.
        projection: the projection of the output windows.
        name: see add_windows_from_geometries
        grid_size: see add_windows_from_geometries
        window_size: see add_windows_from_geometries
        time_range: see add_windows_from_geometries
        use_utm: see add_windows_from_geometries

    Returns:
        list of newly created windows
    """
    # Create geometries for every feature in the file.
    geometries = []
    with fiona.open(fname) as src:
        for feat in src:
            shp = shapely.geometry.shape(feat.geometry)
            crs = CRS.from_wkt(src.crs_wkt)
            geometries.append(STGeometry(Projection(crs, 1, 1), shp, None))

    return add_windows_from_geometries(
        dataset=dataset,
        group=group,
        projection=projection,
        name=name,
        grid_size=grid_size,
        window_size=window_size,
        time_range=time_range,
        use_utm=use_utm,
        geometries=geometries,
    )
