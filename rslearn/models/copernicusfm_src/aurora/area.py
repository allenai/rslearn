"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import torch

__all__ = ["area", "radius_earth"]


# float: Radius of the earth in kilometers.
radius_earth = 6378137 / 1000


def area(polygon: torch.Tensor) -> torch.Tensor:
    """Compute the area of a polygon specified by latitudes and longitudes in degrees.

    This function is a PyTorch port of the PyPI package `area`. In particular, it is heavily
    inspired by the following file:

        https://github.com/scisco/area/blob/9d9549d6ebffcbe4bffe11b71efa2d406d1c9fe9/area/__init__.py

    Args:
        polygon (:class:`torch.Tensor`): Polygon of the shape `(*b, n, 2)` where `b` is an optional
            multidimensional batch size, `n` is the number of points of the polygon, and 2
            concatenates first latitudes and then longitudes. The polygon does not have be closed.

    Returns:
        :class:`torch.Tensor`: Area in square kilometers.
    """
    # Be sure to close the loop.
    polygon = torch.cat((polygon, polygon[..., -1:, :]), axis=-2)

    area = torch.zeros(polygon.shape[:-2], dtype=polygon.dtype, device=polygon.device)
    n = polygon.shape[-2]  # Number of points of the polygon

    rad = torch.deg2rad  # Convert degrees to radians.

    if n > 2:
        for i in range(n):
            i_lower = i
            i_middle = (i + 1) % n
            i_upper = (i + 2) % n

            lon_lower = polygon[..., i_lower, 1]
            lat_middle = polygon[..., i_middle, 0]
            lon_upper = polygon[..., i_upper, 1]

            area = area + (rad(lon_upper) - rad(lon_lower)) * torch.sin(rad(lat_middle))

    area = area * radius_earth * radius_earth / 2

    return torch.abs(area)
