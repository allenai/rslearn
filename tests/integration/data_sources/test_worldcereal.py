import copy
import hashlib
import os
import pathlib
import zipfile
from typing import Any

import numpy as np
import pytest
import shapely
from pytest_httpserver import HTTPServer
from upath import UPath
from werkzeug.wrappers import Request, Response

from rslearn.config import (
    QueryConfig,
    SpaceMode,
)
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import DataSourceContext
from rslearn.data_sources.worldcereal import WorldCereal
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat

# Degrees per pixel to use in the GeoTIFF.
# This roughly corresponds to 10 m/pixel.
DEGREES_PER_PIXEL = 0.0001

# Size of the GeoTIFF.
SIZE = 16

# See tests/integration/fixtures/geometries/seattle2020/seattle2020.py.
SEATTLE_POINT = shapely.Point(-122.33, 47.61)


def _make_test_zips(tmp_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """Make a sample zip file similar to the ESA WorldCereal 2021 ones.

    This is a little bit circular since it uses the class to define where the
    tif files go (and how they are named).

    Our zip file will just contain a single 16x16 GeoTIFF. We make sure it corresponds
    the seattle2020 test geometry so that it can be used to test the data source.

    Args:
        tmp_path: temporary directory that will be used to store the GeoTIFF and zip
            files.

    Returns:
        the filename of the zip files
    """
    seattle_aez = 1
    # Make the GeoTIFF 16x16 centered at the same point as seattle2020.
    src_geom = STGeometry(WGS84_PROJECTION, SEATTLE_POINT, None)
    projection = Projection(WGS84_PROJECTION.crs, DEGREES_PER_PIXEL, -DEGREES_PER_PIXEL)
    dst_geom = src_geom.to_projection(projection)
    bounds = (
        int(dst_geom.shp.x) - SIZE // 2,
        int(dst_geom.shp.y) - SIZE // 2,
        int(dst_geom.shp.x) + SIZE // 2,
        int(dst_geom.shp.y) + SIZE // 2,
    )
    array = np.ones((1, SIZE, SIZE), dtype=np.uint8)
    raster_path = UPath(tmp_path)

    return_dict = {}
    for zip_file in WorldCereal.ZIP_FILENAMES:
        filepath = WorldCereal.zip_filepath_from_filename(zip_file)
        raster_path = UPath(tmp_path / "zips" / filepath)
        raster_path.mkdir(parents=True)
        raster_format = GeotiffRasterFormat()
        raster_format.encode_raster(
            raster_path,
            projection,
            bounds,
            RasterArray(chw_array=array),
            fname=f"{seattle_aez}_{raster_path.stem}.tif",
        )

        # Create a zip file containing it.
        zip_fname = tmp_path / "zips" / zip_file
        zipf = zipfile.ZipFile(zip_fname, "w")
        zipf.write(
            raster_path / f"{seattle_aez}_{raster_path.stem}.tif",
            arcname=UPath(filepath) / f"{seattle_aez}_{raster_path.stem}.tif",
        )
        zipf.close()

        return_dict[zip_file] = zip_fname
    return return_dict


def _make_range_handler(zip_data: bytes) -> Any:
    """Build a handler that serves zip_data honoring Range: bytes=start-end requests.

    WorldCereal._download_with_resume always fetches in bounded Range requests
    (even for the first chunk of a brand new download), so the test server needs
    to respond with real 206 Partial Content responses rather than pytest
    httpserver's default of always returning the full body with 200.
    """

    def handler(request: Request) -> Response:
        range_header = request.headers.get("Range")
        if not range_header:
            return Response(zip_data, status=200, content_type="application/zip")
        range_spec = range_header.split("=", 1)[1]
        start_str, _, end_str = range_spec.partition("-")
        start = int(start_str)
        end = int(end_str) if end_str else len(zip_data) - 1
        end = min(end, len(zip_data) - 1)
        response = Response(
            zip_data[start : end + 1], status=206, content_type="application/zip"
        )
        response.headers["Content-Range"] = f"bytes {start}-{end}/{len(zip_data)}"
        return response

    return handler


def _setup_worldcereal_httpserver(
    worldcereal_dir: UPath, httpserver: HTTPServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Create test zips and configure httpserver to serve them.

    Args:
        worldcereal_dir: directory to create test zips in.
        httpserver: the pytest httpserver to configure.
        monkeypatch: used to patch WorldCereal.ZENODO_FILES_DATA for the duration
            of the test rather than mutating the class-level list in place, since
            that list is shared across the whole test session.
    """
    # ZENODO_FILES_DATA hardcodes the real production Zenodo URL, size, and
    # checksum for each file. Point it at our local httpserver instead, and patch
    # the size/checksum to match our small generated fixture zips, so the test
    # exercises the download path against a controlled server rather than
    # (accidentally) hitting production Zenodo. Patch a deep copy via monkeypatch
    # (rather than mutating WorldCereal.ZENODO_FILES_DATA's dicts in place) so the
    # original production metadata is restored after this test, and other tests
    # in the same session don't inherit stale localhost URLs.
    zenodo_files_data = copy.deepcopy(WorldCereal.ZENODO_FILES_DATA)
    monkeypatch.setattr(WorldCereal, "ZENODO_FILES_DATA", zenodo_files_data)

    zip_name_paths = _make_test_zips(worldcereal_dir)
    for zip_file, zip_fname in zip_name_paths.items():
        with zip_fname.open("rb") as f:
            zip_data = f.read()
        httpserver.expect_request(f"/{zip_file}", method="GET").respond_with_handler(
            _make_range_handler(zip_data)
        )

        for file_data in zenodo_files_data:
            if file_data["filename"] != zip_file:
                continue
            file_data["filesize"] = float(len(zip_data))
            file_data["checksum"] = hashlib.md5(zip_data).hexdigest()
            file_data["links"]["download"] = httpserver.url_for(f"/{zip_file}")


def test_download_with_resume_multi_chunk(
    tmp_path: pathlib.Path, httpserver: HTTPServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WorldCereal._download_with_resume fetches files as many bounded chunks.

    The other integration tests use tiny fixture zips that fit in a single
    RANGE_SIZE_BYTES chunk, so they wouldn't catch a bug in reassembling multiple
    chunks. This forces a small RANGE_SIZE_BYTES so a small file is fetched across
    many separate Range requests, and checks that the request count matches and
    the reassembled bytes/checksum are correct.
    """
    data = os.urandom(10_000)
    checksum = hashlib.md5(data).hexdigest()
    request_count = 0

    def handler(request: Request) -> Response:
        nonlocal request_count
        request_count += 1
        start_str, _, end_str = request.headers["Range"].split("=", 1)[1].partition("-")
        start, end = int(start_str), min(int(end_str), len(data) - 1)
        response = Response(
            data[start : end + 1], status=206, content_type="application/octet-stream"
        )
        response.headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
        return response

    httpserver.expect_request("/file", method="GET").respond_with_handler(handler)
    monkeypatch.setattr(WorldCereal, "RANGE_SIZE_BYTES", 1000)

    dest_path = UPath(tmp_path) / "file.bin"
    WorldCereal._download_with_resume(
        file_url=httpserver.url_for("/file"),
        dest_path=dest_path,
        expected_size=len(data),
        expected_checksum=checksum,
    )

    with dest_path.open("rb") as f:
        assert f.read() == data
    assert request_count == 10


def test_download_with_resume_continues_from_partial_file(
    tmp_path: pathlib.Path, httpserver: HTTPServer
) -> None:
    """A pre-existing .partial file is resumed from its exact byte offset.

    Simulates the interrupted-download scenario the fix targets: a previous
    attempt left some bytes on disk, and the next attempt must continue from
    there via a Range request rather than restarting from byte 0.
    """
    data = os.urandom(5_000)
    checksum = hashlib.md5(data).hexdigest()
    seen_ranges: list[str] = []

    def handler(request: Request) -> Response:
        range_header = request.headers["Range"]
        seen_ranges.append(range_header)
        start_str, _, end_str = range_header.split("=", 1)[1].partition("-")
        start, end = int(start_str), min(int(end_str), len(data) - 1)
        response = Response(
            data[start : end + 1], status=206, content_type="application/octet-stream"
        )
        response.headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
        return response

    httpserver.expect_request("/file", method="GET").respond_with_handler(handler)

    dest_path = UPath(tmp_path) / "file.bin"
    prefix_len = 2000
    with open(dest_path.path + ".partial", "wb") as f:
        f.write(data[:prefix_len])

    WorldCereal._download_with_resume(
        file_url=httpserver.url_for("/file"),
        dest_path=dest_path,
        expected_size=len(data),
        expected_checksum=checksum,
    )

    with dest_path.open("rb") as f:
        assert f.read() == data
    assert seen_ranges == [f"bytes={prefix_len}-{len(data) - 1}"]


def test_download_with_resume_checksum_mismatch_raises_and_cleans_up(
    tmp_path: pathlib.Path, httpserver: HTTPServer
) -> None:
    """A completed download that fails checksum verification is deleted, not kept."""
    data = os.urandom(5_000)
    wrong_checksum = hashlib.md5(b"not the right data").hexdigest()

    def handler(request: Request) -> Response:
        start_str, _, end_str = request.headers["Range"].split("=", 1)[1].partition("-")
        start, end = int(start_str), min(int(end_str), len(data) - 1)
        response = Response(
            data[start : end + 1], status=206, content_type="application/octet-stream"
        )
        response.headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
        return response

    httpserver.expect_request("/file", method="GET").respond_with_handler(handler)

    dest_path = UPath(tmp_path) / "file.bin"
    with pytest.raises(ValueError, match="checksum mismatch"):
        WorldCereal._download_with_resume(
            file_url=httpserver.url_for("/file"),
            dest_path=dest_path,
            expected_size=len(data),
            expected_checksum=wrong_checksum,
        )

    assert not dest_path.exists()
    assert not pathlib.Path(dest_path.path + ".partial").exists()


def test_download_with_resume_remote_destination_verifies_before_publishing(
    httpserver: HTTPServer,
) -> None:
    """Remote destinations get the same retry/checksum validation as local ones.

    Uses fsspec's in-memory filesystem to stand in for a remote (non-local)
    destination cheaply. Downloads are staged to a local temp file and verified
    before being copied to the destination, so a checksum mismatch must not
    leave a corrupt object behind at the final remote path -- unlike the
    previous version, which wrote directly to the remote destination with no
    retry or verification at all.
    """
    data = os.urandom(5_000)
    checksum = hashlib.md5(data).hexdigest()

    def handler(request: Request) -> Response:
        start_str, _, end_str = request.headers["Range"].split("=", 1)[1].partition("-")
        start, end = int(start_str), min(int(end_str), len(data) - 1)
        response = Response(
            data[start : end + 1], status=206, content_type="application/octet-stream"
        )
        response.headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
        return response

    httpserver.expect_request("/file", method="GET").respond_with_handler(handler)

    good_dest = UPath("memory://worldcereal_test/good/file.bin")
    WorldCereal._download_with_resume(
        file_url=httpserver.url_for("/file"),
        dest_path=good_dest,
        expected_size=len(data),
        expected_checksum=checksum,
    )
    with good_dest.open("rb") as f:
        assert f.read() == data

    bad_dest = UPath("memory://worldcereal_test/bad/file.bin")
    with pytest.raises(ValueError, match="checksum mismatch"):
        WorldCereal._download_with_resume(
            file_url=httpserver.url_for("/file"),
            dest_path=bad_dest,
            expected_size=len(data),
            expected_checksum=hashlib.md5(b"wrong").hexdigest(),
        )
    assert not bad_dest.exists()


def test_with_worldcereal_dir(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    httpserver: HTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests ingesting the example data corresponding to seattle2020."""
    worldcereal_dir = UPath(tmp_path) / "worldcereal"
    _setup_worldcereal_httpserver(worldcereal_dir, httpserver, monkeypatch)

    bands = [WorldCereal.band_from_zipfilename(f) for f in WorldCereal.ZIP_FILENAMES]
    for band in bands:
        print(f"Testing {band}")
        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
        data_source = WorldCereal(
            band=band,
            worldcereal_dir=worldcereal_dir,
        )

        print("get items")
        item_groups = data_source.get_items([seattle2020], query_config)
        item = item_groups[0][0].items[0]
        tile_store_dir = UPath(worldcereal_dir) / "tile_store"
        tile_store = DefaultTileStore(str(tile_store_dir))
        tile_store.set_dataset_path(tile_store_dir)

        print("ingest")
        layer_name = "layer"
        data_source.ingest(
            TileStoreWithLayer(tile_store, layer_name),
            item_groups[0][0].items,
            [[seattle2020]],
        )
        print(list(tile_store_dir.glob("layer/1/*")))
        assert tile_store.is_raster_ready(layer_name, item, [band])
        # Double check that the data intersected our example GeoTIFF and isn't just all 0.
        bounds = (
            int(seattle2020.shp.bounds[0]),
            int(seattle2020.shp.bounds[1]),
            int(seattle2020.shp.bounds[2]),
            int(seattle2020.shp.bounds[3]),
        )
        raster_data = tile_store.read_raster(
            layer_name, item, [band], seattle2020.projection, bounds
        )
        assert raster_data.get_chw_array().max() == 1
        print(f"Succeeded for {band}")


def test_with_context_ds_path(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    httpserver: HTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests WorldCereal when context.ds_path is set.

    Previously there was a bug where WorldCereal would pass a UPath to
    LocalFiles.__init__, which would have error when calling join_upath since it
    expects src_dir to be a string. This test prevents regression by making sure the
    code path works when including a DataSourceContext when initializing worldCereal.
    """
    ds_path = UPath(tmp_path) / "dataset"
    ds_path.mkdir(parents=True, exist_ok=True)

    # Use a relative worldcereal_dir path (relative to ds_path)
    worldcereal_dir = "worldcereal_data"
    worldcereal_abs_path = ds_path / worldcereal_dir

    _setup_worldcereal_httpserver(worldcereal_abs_path, httpserver, monkeypatch)

    # Test with just the first band to keep the test fast
    band = WorldCereal.band_from_zipfilename(WorldCereal.ZIP_FILENAMES[0])

    # Create context with ds_path set - this triggers join_upath code path
    context = DataSourceContext(ds_path=ds_path)
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    data_source = WorldCereal(
        band=band,
        worldcereal_dir=worldcereal_dir,  # relative path
        context=context,
    )

    # Verify we can get items (this exercises list_items -> join_upath)
    item_groups = data_source.get_items([seattle2020], query_config)
    assert len(item_groups) == 1
    assert len(item_groups[0]) > 0
    item = item_groups[0][0].items[0]

    # Verify ingest works
    tile_store_dir = ds_path / "tile_store"
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)

    layer_name = "layer"
    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name),
        item_groups[0][0].items,
        [[seattle2020]],
    )
    assert tile_store.is_raster_ready(layer_name, item, [band])
