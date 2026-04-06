"""Array util functions."""

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import torch


def unique_nodata_value(nodata_values: tuple[int | float, ...]) -> int | float:
    """Return the single unique nodata value from a per-band tuple.

    NaN values are treated as equal to each other, unlike the default Python
    ``set`` behaviour where each NaN object is distinct.

    Args:
        nodata_values: per-band nodata values (must be non-empty).

    Returns:
        The unique nodata value.

    Raises:
        ValueError: if *nodata_values* contains more than one distinct value.
    """
    unique: list[int | float] = []
    for v in nodata_values:
        if any((math.isnan(u) if math.isnan(v) else u == v) for u in unique):
            continue
        unique.append(v)
    if len(unique) != 1:
        raise ValueError(
            f"Expected a single unique nodata value but got "
            f"different per-band values: {nodata_values}"
        )
    return unique[0]


def nodata_eq(
    array: npt.NDArray[np.generic],
    nodata_vals: npt.NDArray[np.generic],
) -> npt.NDArray[np.bool_]:
    """NaN-aware element-wise equality between *array* and a nodata sentinel.

    Equivalent to ``array == nodata_vals`` but also matches NaN positions when
    the corresponding nodata sentinel is NaN.

    Args:
        array: the data array (any shape).
        nodata_vals: nodata sentinel(s), broadcastable against *array*.

    Returns:
        Boolean mask with the same shape as the broadcast result; True where the
        value equals (or is NaN-matching) the nodata sentinel.
    """
    result = array == nodata_vals
    if np.issubdtype(nodata_vals.dtype, np.floating):
        nan_mask = np.isnan(nodata_vals)
        if nan_mask.any():
            result = result | (np.isnan(array) & nan_mask)
    return result


def copy_spatial_array(
    src: "torch.Tensor | npt.NDArray[Any]",
    dst: "torch.Tensor | npt.NDArray[Any]",
    src_offset: tuple[int, int],
    dst_offset: tuple[int, int],
) -> None:
    """Copy image content from a source array onto a destination array.

    The source and destination might be in the same coordinate system. Only the portion
    of the source array that overlaps in the coordinate system with the destination
    array will be copied, and other parts of the destination array will not be
    overwritten.

    Args:
        src: the source array (HW or CHW).
        dst: the destination array (HW or CHW).
        src_offset: the (col, row) position of the top-left pixel of src in the coordinate
            system.
        dst_offset: the (col, row) position of the top-left pixel of dst in the coordinate
            system.
    """
    src_height, src_width = src.shape[-2:]
    dst_height, dst_width = dst.shape[-2:]
    # The top-left position within src that intersects with dst.
    src_col_offset = max(dst_offset[0] - src_offset[0], 0)
    src_row_offset = max(dst_offset[1] - src_offset[1], 0)
    # The top-left position within dst that intersects with src.
    # This is the position in dst of the same pixel as the one above in src.
    dst_col_offset = max(src_offset[0] - dst_offset[0], 0)
    dst_row_offset = max(src_offset[1] - dst_offset[1], 0)
    # Now compute how much of src we can copy.
    col_overlap = min(src_width - src_col_offset, dst_width - dst_col_offset)
    row_overlap = min(src_height - src_row_offset, dst_height - dst_row_offset)

    if len(src.shape) == 2:
        dst[
            dst_row_offset : dst_row_offset + row_overlap,
            dst_col_offset : dst_col_offset + col_overlap,
        ] = src[
            src_row_offset : src_row_offset + row_overlap,
            src_col_offset : src_col_offset + col_overlap,
        ]
    elif len(src.shape) == 3:
        dst[
            :,
            dst_row_offset : dst_row_offset + row_overlap,
            dst_col_offset : dst_col_offset + col_overlap,
        ] = src[
            :,
            src_row_offset : src_row_offset + row_overlap,
            src_col_offset : src_col_offset + col_overlap,
        ]
    else:
        raise ValueError(f"Unsupported src shape: {src.shape}")
