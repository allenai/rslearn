"""Read a live European soil texture window and write it as a GeoTIFF.

Example:
    python scripts/burn_european_soil_textures.py \
        --output tmp/european_soil_textures.tif
"""

import argparse
import math
from pathlib import Path

import affine
import numpy as np
import rasterio
import shapely
from rasterio import CRS

from rslearn.config import QueryConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.european_soil_properties import (
    TEXTURE_BANDS,
    EuropeanSoilProperties,
)
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry


def get_utm_epsg(longitude: float, latitude: float) -> int:
    """Return the UTM EPSG code containing a longitude/latitude point."""
    zone = min(60, max(1, int((longitude + 180) // 6) + 1))
    return (32600 if latitude >= 0 else 32700) + zone


def get_window(
    longitude: float,
    latitude: float,
    resolution: float,
    size: int,
) -> tuple[Projection, PixelBounds, STGeometry]:
    """Build a square projected rslearn window centered on a WGS84 point."""
    projection = Projection(
        CRS.from_epsg(get_utm_epsg(longitude, latitude)),
        resolution,
        -resolution,
    )
    point = STGeometry(
        WGS84_PROJECTION, shapely.Point(longitude, latitude), None
    ).to_projection(projection)
    half_size = size / 2
    min_x = math.floor(point.shp.x - half_size)
    min_y = math.floor(point.shp.y - half_size)
    bounds = (min_x, min_y, min_x + size, min_y + size)
    geometry = STGeometry(projection, shapely.box(*bounds), None)
    return projection, bounds, geometry


def read_live_texture(
    projection: Projection,
    bounds: PixelBounds,
    geometry: STGeometry,
) -> np.ndarray:
    """Read and mosaic live clay, sand, and silt tiles for a window."""
    data_source = EuropeanSoilProperties()
    groups = data_source.get_items([geometry], QueryConfig())[0]
    if not groups:
        raise ValueError("the requested window is outside the dataset coverage")

    output = np.full(
        (len(TEXTURE_BANDS), bounds[3] - bounds[1], bounds[2] - bounds[0]),
        np.nan,
        dtype=np.float32,
    )
    for item in groups[0].items:
        raster = data_source.read_raster(
            layer_name="soil_texture",
            item=item,
            bands=TEXTURE_BANDS,
            projection=projection,
            bounds=bounds,
        ).get_chw_array()
        missing = ~np.isfinite(output)
        output[missing] = raster[missing]

    return output


def write_geotiff(
    output_path: Path,
    array: np.ndarray,
    projection: Projection,
    bounds: PixelBounds,
) -> None:
    """Write a CHW float32 array to a compressed, tiled GeoTIFF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transform = affine.Affine(
        projection.x_resolution,
        0,
        bounds[0] * projection.x_resolution,
        0,
        projection.y_resolution,
        bounds[1] * projection.y_resolution,
    )
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        width=array.shape[2],
        height=array.shape[1],
        count=array.shape[0],
        dtype=np.float32,
        crs=projection.crs,
        transform=transform,
        nodata=np.nan,
        tiled=True,
        compress="deflate",
        predictor=3,
    ) as dst:
        dst.write(array)
        for band_index, band_name in enumerate(TEXTURE_BANDS, start=1):
            dst.set_band_description(band_index, band_name)
            dst.update_tags(band_index, units="percent")
        dst.update_tags(
            source="BOKU European 10 m soil property maps",
            dataset_doi="10.5281/zenodo.21281853",
            paper_doi="10.1038/s41597-026-08273-1",
        )


def main() -> None:
    """Parse arguments, read the live source, and write a GeoTIFF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--longitude", type=float, default=16.7)
    parser.add_argument("--latitude", type=float, default=48.2)
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--size", type=int, default=256, help="width/height in pixels")
    args = parser.parse_args()

    if args.resolution <= 0:
        parser.error("--resolution must be positive")
    if args.size <= 0:
        parser.error("--size must be positive")

    projection, bounds, geometry = get_window(
        args.longitude,
        args.latitude,
        args.resolution,
        args.size,
    )
    array = read_live_texture(projection, bounds, geometry)
    write_geotiff(args.output, array, projection, bounds)

    valid = np.all(np.isfinite(array), axis=0)
    print(f"wrote {args.output} ({array.shape[2]}x{array.shape[1]} pixels)")
    print(f"CRS: {projection.crs.to_string()}, valid pixels: {valid.sum()}")
    if valid.any():
        for band_name, band in zip(TEXTURE_BANDS, array):
            print(
                f"{band_name}: min={band[valid].min():.2f}, "
                f"max={band[valid].max():.2f}, mean={band[valid].mean():.2f}"
            )


if __name__ == "__main__":
    main()
