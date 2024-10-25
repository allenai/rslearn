import json
import pathlib

import fiona
import pytest
import shapely
from rasterio.crs import CRS
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.vector_format import GeojsonCoordinateMode, GeojsonVectorFormat


@pytest.mark.parametrize(
    "coordinate_mode",
    [
        GeojsonCoordinateMode.PIXEL,
        GeojsonCoordinateMode.CRS,
        GeojsonCoordinateMode.WGS84,
    ],
)
def test_geojson(
    tmp_path: pathlib.Path, coordinate_mode: GeojsonCoordinateMode
) -> None:
    projection = Projection(CRS.from_epsg(3857), 10, -10)
    col = 1234
    row = 5678
    geometry = STGeometry(projection, shapely.Point(col, row), None)
    features = [Feature(geometry)]
    out_dir = UPath(tmp_path)
    out_fname = out_dir / "data.geojson"

    GeojsonVectorFormat(coordinate_mode).encode_vector(out_dir, projection, features)
    with out_fname.open() as f:
        fc = json.load(f)
        feat_x, feat_y = fc["features"][0]["geometry"]["coordinates"]

    if coordinate_mode == GeojsonCoordinateMode.PIXEL:
        # In pixel mode, the written coordinate should match.
        assert feat_x == pytest.approx(col)
        assert feat_y == pytest.approx(row)

    elif coordinate_mode == GeojsonCoordinateMode.CRS:
        # In CRS mode, it should be in CRS coordinates.
        GeojsonVectorFormat(GeojsonCoordinateMode.CRS).encode_vector(
            out_dir, projection, features
        )
        assert feat_x == pytest.approx(12340)
        assert feat_y == pytest.approx(-56780)

    elif coordinate_mode == GeojsonCoordinateMode.WGS84:
        # In WGS84 mode, it should be longitude/latitude.
        wgs84_geom = geometry.to_projection(WGS84_PROJECTION)
        assert feat_x == pytest.approx(wgs84_geom.shp.x)
        assert feat_y == pytest.approx(wgs84_geom.shp.y)

    # Make sure that when we read the features back, we get the same geometry as
    # before. The bounds is ignored since it is GeoJSON (no index).
    result = GeojsonVectorFormat().decode_vector(out_dir, (0, 0, 0, 0))[0].geometry
    result = result.to_projection(projection)
    assert result.shp.x == pytest.approx(geometry.shp.x)
    assert result.shp.y == pytest.approx(geometry.shp.y)

    # Make sure that fiona can interpret it too (if not PIXEL).
    # This is mainly to check that when using CRS mode, we are writing the "crs"
    # attribute of the FeatureCollection correctly.
    if coordinate_mode in [GeojsonCoordinateMode.CRS, GeojsonCoordinateMode.WGS84]:
        with out_fname.open("rb") as f:
            with fiona.open(f) as src:
                # Convert fiona CRS to rasterio CRS to get Projection object.
                src_projection = Projection(CRS.from_wkt(src.crs.to_wkt()), 1, 1)

                for feat in src:
                    result = STGeometry(
                        src_projection,
                        shapely.geometry.shape(feat.geometry),
                        None,
                    )
                    result = result.to_projection(projection)
                    assert result.shp.x == pytest.approx(geometry.shp.x)
                    assert result.shp.y == pytest.approx(geometry.shp.y)
