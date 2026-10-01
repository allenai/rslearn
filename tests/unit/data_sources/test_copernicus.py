import os
import pathlib
import zipfile
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import shapely
import xarray as xr
from upath import UPath

from rslearn.config import BandSetConfig, DType, LayerConfig, LayerType
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSourceContext
from rslearn.data_sources.copernicus import (
    CopernicusItem,
    Sentinel3OlciEFR,
    Sentinel3SlstrRBT,
    _interpolate_tie_points,
    _interpolate_tie_points_xy,
    _radiance_to_reflectance,
    get_sentinel2_tiles,
)
from rslearn.utils.geometry import STGeometry


class TestGetSentinel2Tiles:
    """Tests for get_sentinel2_tiles."""

    def test_antimeridian_handling(self, tmp_path: pathlib.Path) -> None:
        """Make sure that get_sentinel2_tiles handles the antimeridian correctly.

        Previously we returned tiles that spanned the antimeridian for any geometry
        that had a matching latitude.
        """
        # We use a 1x1 degree geometry that should match with these tiles:
        # - 10UFU
        # - 10TFT
        # - 10UEU
        # - 10TET
        geom = STGeometry(
            WGS84_PROJECTION,
            shapely.box(-122, 47, -121, 48),
            (
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 2, 1, tzinfo=UTC),
            ),
        )
        tiles = get_sentinel2_tiles(geom, UPath(tmp_path))
        assert set(tiles) == {
            "10UFU",
            "10TFT",
            "10UEU",
            "10TET",
        }, f"Got incorrect tile list {tiles}"


def _write_netcdf(
    path: pathlib.Path, variables: dict[str, tuple[tuple[str, ...], Any]]
) -> None:
    xr.Dataset(variables).to_netcdf(path)


def _zip_files(zip_path: pathlib.Path, files: list[pathlib.Path]) -> None:
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in files:
            archive.write(path, arcname=f"product.SEN3/{path.name}")


def _test_item() -> CopernicusItem:
    geometry = STGeometry(
        WGS84_PROJECTION,
        shapely.box(0, 0, 1, 1),
        (
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
        ),
    )
    return CopernicusItem("test", geometry, "test-uuid")


class TestSentinel3:
    """Tests for native Sentinel-3 Copernicus data sources."""

    def test_tie_point_interpolation(self) -> None:
        values = np.array([[0, 2], [2, 4]], dtype=np.float32)
        result = _interpolate_tie_points(values, (3, 3))
        np.testing.assert_allclose(
            result,
            np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.float32),
        )

    def test_tie_point_interpolation_uses_xy_coordinates(self) -> None:
        # Tie grid spans x in [-2000, 2000] (decreasing) but the image only covers
        # [0, 1000], so the image edges must not be stretched to the tie-grid edges.
        tie_x = np.array([[2000, 0, -2000], [2000, 0, -2000]], dtype=np.float64)
        tie_y = np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float64)
        values = np.array([[20, 0, -20], [30, 10, -10]], dtype=np.float32)
        # Points within one tie spacing of the grid are extrapolated (y=15); points
        # further out (x=5000) or without coordinates are NaN.
        x = np.array([[0, 1000, np.nan], [500, 5000, 0]])
        y = np.array([[0, 5, 0], [10, 0, 15]])
        result = _interpolate_tie_points_xy(values, tie_x, tie_y, x, y)
        np.testing.assert_allclose(
            result, [[0, 15, np.nan], [15, np.nan, 15]], rtol=1e-6
        )

    def test_radiance_to_reflectance_preserves_unclipped_values(self) -> None:
        result = _radiance_to_reflectance(
            np.array([[1.0, 4.0]], dtype=np.float32),
            np.array([[2.0, 2.0]], dtype=np.float32),
            np.ones((1, 2), dtype=np.float32),
        )
        np.testing.assert_allclose(result, [[np.pi / 2, 2 * np.pi]])

    def test_band_selection_and_product_filters(self) -> None:
        layer_cfg = LayerConfig(
            type=LayerType.RASTER,
            band_sets=[
                BandSetConfig(
                    dtype=DType.FLOAT32,
                    bands=["S7_BT", "S1_reflectance"],
                )
            ],
        )
        source = Sentinel3SlstrRBT(
            context=DataSourceContext(layer_config=layer_cfg),
            access_token="test-token",
        )
        assert source.band_names == ["S1_reflectance", "S7_BT"]
        assert source.query_filter is not None
        assert "Collection/Name eq 'SENTINEL-3'" in source.query_filter
        assert "SL_1_RBT___" in source.query_filter
        assert "SLSTR" in source.query_filter

    def test_catalogue_source_does_not_require_download_credentials(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            source = Sentinel3OlciEFR(band_names=["Oa01_reflectance"])

        with pytest.raises(ValueError, match="downloads require authentication"):
            source._get_access_token()

    def test_olci_processes_synthetic_safe_product(
        self, tmp_path: pathlib.Path
    ) -> None:
        shape = (2, 2)
        files = []

        instrument_path = tmp_path / "instrument_data.nc"
        _write_netcdf(
            instrument_path,
            {
                "solar_flux": (
                    ("band", "detector"),
                    np.full((21, 2), 2.0, dtype=np.float32),
                ),
                "detector_index": (
                    ("rows", "cols"),
                    np.array([[0, 1], [0, 1]], dtype=np.int16),
                ),
            },
        )
        files.append(instrument_path)

        geometries_path = tmp_path / "tie_geometries.nc"
        _write_netcdf(
            geometries_path,
            {"SZA": (("tie_rows", "tie_cols"), np.zeros(shape, np.float32))},
        )
        files.append(geometries_path)

        coordinates_path = tmp_path / "geo_coordinates.nc"
        _write_netcdf(
            coordinates_path,
            {
                "latitude": (
                    ("rows", "cols"),
                    np.array([[0, 0], [1, 1]], dtype=np.float32),
                ),
                "longitude": (
                    ("rows", "cols"),
                    np.array([[0, 1], [0, 1]], dtype=np.float32),
                ),
            },
        )
        files.append(coordinates_path)

        radiance_path = tmp_path / "Oa01_radiance.nc"
        _write_netcdf(
            radiance_path,
            {
                "Oa01_radiance": (
                    ("rows", "cols"),
                    np.ones(shape, dtype=np.float32),
                )
            },
        )
        files.append(radiance_path)

        zip_path = tmp_path / "olci.zip"
        _zip_files(zip_path, files)
        source = Sentinel3OlciEFR(
            band_names=["Oa01_reflectance"], access_token="test-token"
        )
        tile_store = MagicMock()
        tile_store.is_raster_ready.return_value = False

        with patch("rslearn.data_sources.copernicus._write_swath") as write_swath:
            source._process_product_zip(tile_store, _test_item(), str(zip_path))

        assert write_swath.call_count == 1
        args = write_swath.call_args.args
        assert args[2] == ["Oa01_reflectance"]
        np.testing.assert_allclose(args[3], np.full((1, 2, 2), np.pi / 2))

    def test_slstr_processes_reflectance_and_bt_grids(
        self, tmp_path: pathlib.Path
    ) -> None:
        shape = (2, 2)
        files = []
        datasets: dict[str, dict[str, tuple[tuple[str, ...], Any]]] = {
            "indices_an.nc": {
                "detector_an": (
                    ("rows", "cols"),
                    np.array([[0, 1], [0, 1]], dtype=np.float32),
                )
            },
            # The tie grid is wider than the nadir image and x decreases, as in real
            # products: image columns at x=0 and x=1000 fall within tie columns 1-2.
            "geometry_tn.nc": {
                "solar_zenith_tn": (
                    ("tie_rows", "tie_cols"),
                    np.array([[120, 0, 120], [120, 0, 120]], dtype=np.float32),
                )
            },
            "cartesian_tx.nc": {
                "x_tx": (
                    ("tie_rows", "tie_cols"),
                    np.array([[2000, 0, -2000], [2000, 0, -2000]], dtype=np.float64),
                ),
                "y_tx": (
                    ("tie_rows", "tie_cols"),
                    np.array([[0, 0, 0], [1000, 1000, 1000]], dtype=np.float64),
                ),
            },
            "cartesian_an.nc": {
                "x_an": (
                    ("rows", "cols"),
                    np.array([[0, 1000], [0, 1000]], dtype=np.float64),
                ),
                "y_an": (
                    ("rows", "cols"),
                    np.array([[0, 0], [1000, 1000]], dtype=np.float64),
                ),
            },
            "geodetic_an.nc": {
                "latitude_an": (
                    ("rows", "cols"),
                    np.array([[0, 0], [1, 1]], dtype=np.float32),
                ),
                "longitude_an": (
                    ("rows", "cols"),
                    np.array([[0, 1], [0, 1]], dtype=np.float32),
                ),
            },
            "S1_radiance_an.nc": {
                "S1_radiance_an": (
                    ("rows", "cols"),
                    np.ones(shape, dtype=np.float32),
                )
            },
            "S1_quality_an.nc": {
                "S1_solar_irradiance_an": (
                    ("detector",),
                    np.array([2, 4], dtype=np.float32),
                )
            },
            "geodetic_in.nc": {
                "latitude_in": (
                    ("rows", "cols"),
                    np.array([[0, 0], [1, 1]], dtype=np.float32),
                ),
                "longitude_in": (
                    ("rows", "cols"),
                    np.array([[0, 1], [0, 1]], dtype=np.float32),
                ),
            },
            "S7_BT_in.nc": {
                "S7_BT_in": (
                    ("rows", "cols"),
                    np.full(shape, 280, dtype=np.float32),
                )
            },
        }
        for filename, variables in datasets.items():
            path = tmp_path / filename
            _write_netcdf(path, variables)
            files.append(path)

        zip_path = tmp_path / "slstr.zip"
        _zip_files(zip_path, files)
        source = Sentinel3SlstrRBT(
            band_names=["S1_reflectance", "S7_BT"], access_token="test-token"
        )
        tile_store = MagicMock()
        tile_store.is_raster_ready.return_value = False

        with patch("rslearn.data_sources.copernicus._write_swath") as write_swath:
            source._process_product_zip(tile_store, _test_item(), str(zip_path))

        assert write_swath.call_count == 2
        reflectance_args = write_swath.call_args_list[0].args
        bt_args = write_swath.call_args_list[1].args
        assert reflectance_args[2] == ["S1_reflectance"]
        # Column 0: SZA 0, irradiance 2. Column 1: SZA 60 (cos 0.5), irradiance 4.
        np.testing.assert_allclose(
            reflectance_args[3], np.full((1, 2, 2), np.pi / 2), rtol=1e-5
        )
        assert bt_args[2] == ["S7_BT"]
        np.testing.assert_allclose(bt_args[3], np.full((1, 2, 2), 280))
