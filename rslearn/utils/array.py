"""Array util functions."""

from typing import Any

import numpy.typing as npt
import torch


def copy_spatial_array(
    src: torch.Tensor | npt.NDArray[Any],
    dst: torch.Tensor | npt.NDArray[Any],
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
