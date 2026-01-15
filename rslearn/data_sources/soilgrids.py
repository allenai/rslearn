"""Data source for SoilGrids via the `soilgrids` Python package.

This source is intended to be used with `ingest: false` (direct materialization),
since data is fetched on-demand per window.
"""

from __future__ import annotations

import tempfile
from typing import Any

import numpy as np
import numpy.typing as npt
import rasterio
import rasterio.warp
import shapely
from rasterio.crs import CRS
from rasterio.enums import Resampling
from upath import UPath

from rslearn.config import LayerConfig, QueryConfig
from rslearn.dataset import Window
from rslearn.dataset.materialize import RasterMaterializer
from rslearn.tile_stores import TileStore, TileStoreWithLayer
from rslearn.utils import Feature, PixelBounds, Projection, STGeometry
from rslearn.utils.geometry import get_global_geometry
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from .data_source import DataSource, DataSourceContext, Item
from .utils import match_candidate_items_to_window


def _crs_to_rasterio(crs: str) -> CRS:
    """Best-effort conversion of CRS strings used by `soilgrids` to rasterio CRS."""
    try:
        return CRS.from_string(crs)
    except Exception:
        # Handle URNs like "urn:ogc:def:crs:EPSG::4326".
        parts = [p for p in crs.replace(":", " ").split() if p.isdigit()]
        if parts:
            return CRS.from_epsg(int(parts[-1]))
        raise


def _crs_to_soilgrids_urn(crs: str) -> str:
    """Convert common CRS spellings to the URN form expected by `soilgrids`.

    The `soilgrids` package compares CRS strings against the supported CRS URNs from
    OWSLib (e.g. "urn:ogc:def:crs:EPSG::3857"). This helper allows users to specify
    simpler forms like "EPSG:3857" while still working.
    """
    s = crs.strip()

    # If already an EPSG URN, canonicalize to the form soilgrids expects.
    if s.lower().startswith("urn:ogc:def:crs:") and "epsg" in s.lower():
        parts = [p for p in s.replace(":", " ").split() if p.isdigit()]
        if parts:
            return f"urn:ogc:def:crs:EPSG::{parts[-1]}"
        return s

    # Accept "EPSG:3857", "epsg:3857", or other strings containing an EPSG code.
    if "epsg" in s.lower():
        parts = [p for p in s.replace(":", " ").split() if p.isdigit()]
        if parts:
            return f"urn:ogc:def:crs:EPSG::{parts[-1]}"

    return s


class SoilGrids(DataSource, TileStore):
    """Access SoilGrids coverages as an rslearn raster data source."""

    def __init__(
        self,
        service_id: str,
        coverage_id: str,
        crs: str = "EPSG:3857",
        width: int | None = None,
        height: int | None = None,
        resx: float | None = None,
        resy: float | None = None,
        response_crs: str | None = None,
        band_names: list[str] = ["B1"],
        context: DataSourceContext = DataSourceContext(),
    ):
        """Create a new SoilGrids data source.

        Args:
            service_id: SoilGrids map service id (e.g., "clay", "phh2o").
            coverage_id: coverage id within the service (e.g., "clay_0-5cm_mean").
            crs: request CRS string passed through to `soilgrids.SoilGrids`, typically
                a URN like "urn:ogc:def:crs:EPSG::4326" or "urn:ogc:def:crs:EPSG::152160".
            width: optional WCS WIDTH parameter. Required by SoilGrids WCS when CRS is
                EPSG:4326.
            height: optional WCS HEIGHT parameter.
            resx: optional WCS RESX parameter (projection units / pixel).
            resy: optional WCS RESY parameter (projection units / pixel).
            response_crs: optional response CRS (defaults to `crs`).
            band_names: band names exposed to rslearn. For a single coverage, this
                should have length 1.
            context: rslearn data source context.
        """
        if len(band_names) != 1:
            raise ValueError("SoilGrids currently supports only single-band coverages")
        if (width is None) != (height is None):
            raise ValueError("width and height must be specified together")
        if (resx is None) != (resy is None):
            raise ValueError("resx and resy must be specified together")
        if width is not None and resx is not None:
            raise ValueError("specify either width/height or resx/resy, not both")

        self.service_id = service_id
        self.coverage_id = coverage_id
        self.crs = crs
        self.width = width
        self.height = height
        self.resx = resx
        self.resy = resy
        self.response_crs = response_crs
        self.band_names = band_names

        # Represent the coverage as a single item that matches all windows.
        item_name = f"{self.service_id}:{self.coverage_id}"
        self._items = [Item(item_name, get_global_geometry(time_range=None))]

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[Item]]]:
        """Get item groups matching each requested geometry."""
        groups = []
        for geometry in geometries:
            cur_groups = match_candidate_items_to_window(
                geometry, self._items, query_config
            )
            groups.append(cur_groups)
        return groups

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserialize an item from JSON-decoded data."""
        return Item.deserialize(serialized_item)

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest is not supported (direct materialization only)."""
        raise NotImplementedError(
            "SoilGrids is intended for direct materialization; set data_source.ingest=false."
        )

    def is_raster_ready(self, layer_name: str, item_name: str, bands: list[str]) -> bool:
        """Return whether the requested raster is ready (always true for direct reads)."""
        return True

    def get_raster_bands(self, layer_name: str, item_name: str) -> list[list[str]]:
        """Return the band sets available for this coverage."""
        return [self.band_names]

    def get_raster_bounds(
        self, layer_name: str, item_name: str, bands: list[str], projection: Projection
    ) -> PixelBounds:
        """Return (approximate) bounds for this raster in the requested projection."""
        # We don't know bounds without an extra metadata request; treat as "very large"
        # so materialization always attempts reads for windows.
        return (-10**9, -10**9, 10**9, 10**9)

    def _download_geotiff(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
        output: str,
        width: int | None,
        height: int | None,
        resx: float | None,
        resy: float | None,
    ) -> None:
        try:
            from soilgrids import SoilGrids as SoilGridsClient
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Missing dependency 'soilgrids'. Install rslearn with its dependencies."
            ) from e

        client = SoilGridsClient()
        kwargs: dict[str, Any] = dict(
            service_id=self.service_id,
            coverage_id=self.coverage_id,
            crs=_crs_to_soilgrids_urn(self.crs),
            west=west,
            south=south,
            east=east,
            north=north,
            output=output,
        )
        if width is not None and height is not None:
            kwargs["width"] = width
            kwargs["height"] = height
        elif resx is not None and resy is not None:
            kwargs["resx"] = resx
            kwargs["resy"] = resy

        if self.response_crs is not None:
            kwargs["response_crs"] = _crs_to_soilgrids_urn(self.response_crs)

        client.get_coverage_data(**kwargs)

    def read_raster(
        self,
        layer_name: str,
        item_name: str,
        bands: list[str],
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> npt.NDArray[Any]:
        """Read and reproject a SoilGrids coverage subset into the requested grid."""
        if bands != self.band_names:
            raise ValueError(f"expected request for bands {self.band_names} but got {bands}")

        # Compute bounding box in CRS coordinates for the request.
        request_crs = _crs_to_rasterio(self.crs)
        request_projection = Projection(request_crs, 1.0, 1.0)
        request_geom = STGeometry(projection, shapely.box(*bounds), None).to_projection(
            request_projection
        )
        west, south, east, north = request_geom.shp.bounds

        # Determine output grid for the WCS request.
        #
        # If the user explicitly configured an output grid (width/height or resx/resy),
        # we respect it.
        #
        # Otherwise, default to requesting at ~250 m resolution in the request CRS
        # (when it is projected), and then reprojecting to the window grid.
        #
        # For EPSG:4326 requests, SoilGrids WCS requires WIDTH/HEIGHT, so we default
        # to matching the window pixel size.
        window_width = bounds[2] - bounds[0]
        window_height = bounds[3] - bounds[1]

        out_width = self.width
        out_height = self.height
        out_resx = self.resx
        out_resy = self.resy

        if request_crs.to_epsg() == 4326 and out_width is None:
            # Required by the SoilGrids WCS for EPSG:4326; resx/resy is not accepted.
            out_width = window_width
            out_height = window_height
            out_resx = None
            out_resy = None
        elif out_width is None and out_resx is None:
            # Default to native-ish SoilGrids resolution (~250 m) in projected CRSs.
            out_resx = 250.0
            out_resy = 250.0

        with tempfile.TemporaryDirectory(prefix="rslearn_soilgrids_") as tmpdir:
            output_path = str(UPath(tmpdir) / "coverage.tif")
            self._download_geotiff(
                west=west,
                south=south,
                east=east,
                north=north,
                output=output_path,
                width=out_width,
                height=out_height,
                resx=out_resx,
                resy=out_resy,
            )

            with rasterio.open(output_path) as src:
                src_array = src.read(1).astype(np.float32)
                src_nodata = src.nodata
                scale = float(src.scales[0]) if src.scales else 1.0
                offset = float(src.offsets[0]) if src.offsets else 0.0

                if src_nodata is not None:
                    valid_mask = src_array != float(src_nodata)
                    src_array[valid_mask] = src_array[valid_mask] * scale + offset
                    dst_nodata = float(src_nodata)
                    src_nodata_val = dst_nodata
                else:
                    src_array = src_array * scale + offset
                    dst_nodata = -32768.0
                    src_nodata_val = None

                src_chw = src_array[None, :, :]
                dst = np.full(
                    (1, bounds[3] - bounds[1], bounds[2] - bounds[0]),
                    dst_nodata,
                    dtype=np.float32,
                )
                dst_transform = get_transform_from_projection_and_bounds(projection, bounds)

                rasterio.warp.reproject(
                    source=src_chw,
                    src_crs=src.crs,
                    src_transform=src.transform,
                    src_nodata=src_nodata_val,
                    destination=dst,
                    dst_crs=projection.crs,
                    dst_transform=dst_transform,
                    dst_nodata=dst_nodata,
                    resampling=resampling,
                )
                return dst

    def write_raster(
        self,
        layer_name: str,
        item_name: str,
        bands: list[str],
        projection: Projection,
        bounds: PixelBounds,
        array: npt.NDArray[Any],
    ) -> None:
        """Writing is not supported (read-only source)."""
        raise NotImplementedError("SoilGrids is read-only.")

    def write_raster_file(
        self, layer_name: str, item_name: str, bands: list[str], fname: UPath
    ) -> None:
        """Writing is not supported (read-only source)."""
        raise NotImplementedError("SoilGrids is read-only.")

    def is_vector_ready(self, layer_name: str, item_name: str) -> bool:
        """Vector access is not supported."""
        raise NotImplementedError("SoilGrids does not support vector data.")

    def read_vector(
        self, layer_name: str, item_name: str, projection: Projection, bounds: PixelBounds
    ) -> list[Feature]:
        """Vector access is not supported."""
        raise NotImplementedError("SoilGrids does not support vector data.")

    def write_vector(self, layer_name: str, item_name: str, features: list[Feature]) -> None:
        """Vector access is not supported."""
        raise NotImplementedError("SoilGrids does not support vector data.")

    def materialize(
        self,
        window: Window,
        item_groups: list[list[Item]],
        layer_name: str,
        layer_cfg: LayerConfig,
    ) -> None:
        """Materialize a window by reading from SoilGrids on-demand."""
        RasterMaterializer().materialize(
            TileStoreWithLayer(self, layer_name),
            window,
            layer_name,
            layer_cfg,
            item_groups,
        )
