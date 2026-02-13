"""Container class for a CTHW raster that has timestamps and other metadata."""

from datetime import datetime

import numpy as np
import numpy.typing as npt


class RasterArray:
    """RasterArray specifies a CTHW raster along with associated metadata."""

    array: npt.NDArray[np.generic]  # (C, T, H, W)
    timestamps: list[tuple[datetime, datetime]] | None

    def __init__(
        self,
        *,
        array: npt.NDArray[np.generic] | None = None,
        timestamps: list[tuple[datetime, datetime]] | None = None,
        chw_array: npt.NDArray[np.generic] | None = None,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> None:
        """Initialize a RasterArray.

        Exactly one of ``array`` or ``chw_array`` must be provided.

        Args:
            array: a 4D numpy array with shape (C, T, H, W). Use with
                ``timestamps`` for multi-timestep data.
            timestamps: optional list of (start, end) datetime tuples, one per
                timestep. Length must equal T. Only valid with ``array``.
            chw_array: a 3D numpy array with shape (C, H, W). Convenience path
                for single-timestep data; internally expanded to (C, 1, H, W).
            time_range: optional single (start, end) datetime tuple for the one
                timestep. Only valid with ``chw_array``.

        Raises:
            ValueError: if both or neither of ``array``/``chw_array`` are given,
                if ``timestamps`` is used with ``chw_array``, if ``time_range``
                is used with ``array``, or if shapes/lengths are inconsistent.
        """
        has_array = array is not None
        has_chw = chw_array is not None

        if has_array == has_chw:
            raise ValueError(
                "Exactly one of 'array' (CTHW) or 'chw_array' (CHW) must be provided."
            )

        if has_chw:
            if timestamps is not None:
                raise ValueError(
                    "'timestamps' cannot be used with 'chw_array'; use 'time_range' instead."
                )
            assert chw_array is not None
            if chw_array.ndim != 3:
                raise ValueError(
                    f"chw_array expects a 3D CHW array, got {chw_array.ndim}D "
                    f"with shape {chw_array.shape}"
                )
            self.array = chw_array[:, np.newaxis, :, :]
            self.timestamps = [time_range] if time_range is not None else None
        else:
            if time_range is not None:
                raise ValueError(
                    "'time_range' cannot be used with 'array'; use 'timestamps' instead."
                )
            assert array is not None
            if array.ndim != 4:
                raise ValueError(
                    f"RasterArray expects a 4D CTHW array, got {array.ndim}D "
                    f"with shape {array.shape}"
                )
            self.array = array
            self.timestamps = timestamps

        if self.timestamps is not None and len(self.timestamps) != self.array.shape[1]:
            raise ValueError(
                f"timestamps length ({len(self.timestamps)}) does not match "
                f"T dimension ({self.array.shape[1]})"
            )

    def get_chw_array(self) -> npt.NDArray[np.generic]:
        """Return the array as (C, H, W), requiring T=1.

        Raises:
            ValueError: if the T dimension is not 1.
        """
        if self.array.shape[1] != 1:
            raise ValueError(f"get_chw_array requires T=1, got T={self.array.shape[1]}")
        return self.array[:, 0, :, :]

    def __repr__(self) -> str:
        """Return a string representation of this RasterArray."""
        return (
            f"RasterArray(array=<shape={self.array.shape}, dtype={self.array.dtype}>, "
            f"timestamps={self.timestamps})"
        )

    # -- Proposed future fields --
    # band_names: list[str] | None = None
    # nodata_value: int | float | None = None
