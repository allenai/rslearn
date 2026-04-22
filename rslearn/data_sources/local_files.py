"""Data source for raster or vector data in local files."""

import functools
import json
from typing import Any, Generic, TypeVar

import fiona
import shapely
import shapely.geometry
from rasterio.crs import CRS
from rasterio.transform import from_gcps
from upath import UPath

import rslearn.data_sources.utils
from rslearn.config import LayerType
from rslearn.const import SHAPEFILE_AUX_EXTENSIONS
from rslearn.data_sources.utils import MatchedItemGroup
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.feature import Feature
from rslearn.utils.fsspec import (
    get_relative_suffix,
    get_upath_local,
    join_upath,
    open_rasterio_upath_reader,
)
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.grid_index import GridIndex

from .data_source import DataSource, DataSourceContext, Item, QueryConfig

logger = get_logger("__name__")


ItemType = TypeVar("ItemType", bound=Item)
ImporterType = TypeVar("ImporterType", bound="Importer")

SOURCE_NAME = "rslearn.data_sources.local_files.LocalFiles"


class Importer(Generic[ItemType]):
    """An abstract base class for importing data from local files."""

    def list_items(self, src_dir: UPath) -> list[ItemType]:
        """Extract a list of Items from the source directory.

        Args:
            src_dir: the source directory.
        """
        raise NotImplementedError

    def ingest_item(
        self,
        tile_store: TileStoreWithLayer,
        item: ItemType,
        cur_geometries: list[STGeometry],
    ) -> None:
        """Ingest the specified local file item.

        Args:
            tile_store: the tile store to ingest the data into.
            item: the Item to ingest
            cur_geometries: the geometries where the item is needed.
        """
        raise NotImplementedError


class RasterItemSpec:
    """Representation of configuration that directly specifies the available items."""

    def __init__(
        self,
        fnames: list[str],
        bands: list[list[str]] | None = None,
        name: str | None = None,
    ):
        """Create a new RasterItemSpec.

        Args:
            fnames: the list of image files in this item.
            bands: the bands provided by each of the image files.
            name: what the item should be named
        """
        self.fnames = fnames
        self.bands = bands
        self.name = name

    def serialize(self) -> dict[str, Any]:
        """Serializes the RasterItemSpec to a JSON-encodable dictionary."""
        return {
            "fnames": [str(path) for path in self.fnames],
            "bands": self.bands,
            "name": self.name,
        }

    @staticmethod
    def deserialize(d: dict[str, Any]) -> "RasterItemSpec":
        """Deserializes a RasterItemSpec from a JSON-decoded dictionary."""
        return RasterItemSpec(
            fnames=[s for s in d["fnames"]],
            bands=d["bands"],
            name=d["name"],
        )


class RasterItem(Item):
    """An item corresponding to a local file."""

    def __init__(
        self, name: str, geometry: STGeometry, src_dir: str, spec: RasterItemSpec
    ):
        """Creates a new RasterItem.

        Args:
            name: unique name of the item
            geometry: the spatial and temporal extent of the item
            src_dir: the source directory.
            spec: the RasterItemSpec that specifies the filename(s) and bands.
        """
        super().__init__(name, geometry)
        self.src_dir = src_dir
        self.spec = spec

    def serialize(self) -> dict:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["src_dir"] = str(self.src_dir)
        d["spec"] = self.spec.serialize()
        return d

    @staticmethod
    def deserialize(d: dict) -> "RasterItem":
        """Deserializes an item from a JSON-decoded dictionary."""
        item = super(RasterItem, RasterItem).deserialize(d)
        src_dir = d["src_dir"]
        spec = RasterItemSpec.deserialize(d["spec"])
        return RasterItem(
            name=item.name, geometry=item.geometry, src_dir=src_dir, spec=spec
        )


class VectorItem(Item):
    """An item corresponding to a local file."""

    def __init__(self, name: str, geometry: STGeometry, path_uri: str):
        """Creates a new LocalFileItem.

        Args:
            name: unique name of the item
            geometry: the spatial and temporal extent of the item
            path_uri: URI representation of the path of this file
        """
        super().__init__(name, geometry)
        self.path_uri = path_uri

    def serialize(self) -> dict:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["path_uri"] = self.path_uri
        return d

    @staticmethod
    def deserialize(d: dict) -> "VectorItem":
        """Deserializes an item from a JSON-decoded dictionary."""
        item = super(VectorItem, VectorItem).deserialize(d)
        return VectorItem(
            name=item.name, geometry=item.geometry, path_uri=d["path_uri"]
        )


class RasterImporter(Importer):
    """An Importer for raster data."""

    def __init__(self, item_specs: list[RasterItemSpec] | None = None):
        """Create a new RasterImporter.

        Args:
            item_specs: the specs to specify the raster items directly. If None, the
                raster items are automatically detected from the files in the source
                directory.
        """
        self.item_specs = item_specs

    def list_items(self, src_dir: UPath) -> list[Item]:
        """Extract a list of Items from the source directory.

        Args:
            src_dir: the source directory.
        """
        item_specs: list[RasterItemSpec]

        # See if user has provided the item specs directly.
        if self.item_specs is not None:
            item_specs = self.item_specs
        else:
            # Otherwise we need to list files and assume each one is separate.
            # And we'll need to autodetect the bands later.
            item_specs = []
            file_paths = src_dir.glob("**/*.*")
            for path in file_paths:
                # Ignore JSON files.
                if path.name.endswith(".json"):
                    continue

                # Ignore temporary files that may be created by open_atomic.
                # The suffix should be like "X.tif.tmp.1234".
                parts = path.name.split(".")
                if len(parts) >= 4 and parts[-2] == "tmp" and parts[-1].isdigit():
                    continue

                spec = RasterItemSpec(
                    fnames=[get_relative_suffix(src_dir, path)], bands=None
                )
                item_specs.append(spec)

        items: list[Item] = []
        for spec in item_specs:
            # Get geometry from the first raster file.
            # We assume files are readable with rasterio.
            fname = join_upath(src_dir, spec.fnames[0])
            with open_rasterio_upath_reader(fname) as src:
                gcps, gcp_crs = src.gcps
                if src.crs is None and len(gcps) > 0:
                    # Fit an affine transform from the GCPs and project the
                    # image corners to get image's geographic extent.
                    transform = from_gcps(gcps)
                    corners = [
                        transform * (0, 0),
                        transform * (src.width, 0),
                        transform * (src.width, src.height),
                        transform * (0, src.height),
                    ]
                    xs = [c[0] for c in corners]
                    ys = [c[1] for c in corners]
                    shp = shapely.box(min(xs), min(ys), max(xs), max(ys))
                    # We use 1 unit/pixel projection so it is compatible with
                    # the shape above. The actual resolution may differ, but
                    # it is okay since this geometry is only for item-window
                    # matching. Re-projection will be handled by the tile
                    # store, e.g. DefaultTileStore applies WarpedVRT.
                    projection = Projection(gcp_crs, 1, 1)
                    geometry = STGeometry(projection, shp, None)
                else:
                    crs = src.crs
                    left = src.transform.c
                    top = src.transform.f
                    x_resolution = src.transform.a
                    y_resolution = src.transform.e
                    start = (int(left / x_resolution), int(top / y_resolution))
                    shp = shapely.box(
                        start[0],
                        start[1],
                        start[0] + src.width,
                        start[1] + src.height,
                    )
                    projection = Projection(crs, x_resolution, y_resolution)
                    geometry = STGeometry(projection, shp, None)

            if spec.name:
                item_name = spec.name
            else:
                item_name = fname.name.split(".")[0]

            logger.debug(
                "RasterImporter.list_items: got bounds of %s: %s", item_name, geometry
            )
            items.append(RasterItem(item_name, geometry, str(src_dir), spec))

        logger.debug("RasterImporter.list_items: discovered %d items", len(items))
        return items

    def ingest_item(
        self,
        tile_store: TileStoreWithLayer,
        item: Item,
        cur_geometries: list[STGeometry],
    ) -> None:
        """Ingest the specified local file item.

        Args:
            tile_store: the tile store to ingest the data into.
            item: the RasterItem to ingest
            cur_geometries: the geometries where the item is needed.
        """
        assert isinstance(item, RasterItem)
        for file_idx, fname in enumerate(item.spec.fnames):
            fname_upath = join_upath(UPath(item.src_dir), fname)
            with open_rasterio_upath_reader(fname_upath) as src:
                if item.spec.bands:
                    bands = item.spec.bands[file_idx]
                else:
                    bands = [f"B{band_idx + 1}" for band_idx in range(src.count)]

            if tile_store.is_raster_ready(item, bands):
                continue
            tile_store.write_raster_file(
                item, bands, fname_upath, time_range=item.geometry.time_range
            )


class VectorImporter(Importer):
    """An Importer for vector data."""

    # We need some buffer around GeoJSON bounds in case it just contains one point.
    item_buffer_epsilon = 1e-4

    def list_items(self, src_dir: UPath) -> list[Item]:
        """Extract a list of Items from the source directory.

        Args:
            src_dir: the source directory.
        """
        file_paths = src_dir.glob("**/*.*")
        items: list[Item] = []

        for path in file_paths:
            # Ignore JSON files.
            if path.name.endswith(".json"):
                continue

            # Get the bounds of the features in the vector file, which we assume fiona can
            # read.
            # For shapefile, to open it we need to copy all the aux files.
            aux_files: list[UPath] = []
            if path.name.endswith(".shp"):
                prefix = ".".join(path.name.split(".")[:-1])
                for ext in SHAPEFILE_AUX_EXTENSIONS:
                    aux_files.append(path.parent / (prefix + ext))

            with get_upath_local(path, extra_paths=aux_files) as local_fname:
                with fiona.open(local_fname) as src:
                    crs = CRS.from_wkt(src.crs.to_wkt())
                    bounds = None
                    for feat in src:
                        shp = shapely.geometry.shape(feat.geometry)
                        cur_bounds = shp.bounds
                        if bounds is None:
                            bounds = list(cur_bounds)
                        else:
                            bounds[0] = min(bounds[0], cur_bounds[0])
                            bounds[1] = min(bounds[1], cur_bounds[1])
                            bounds[2] = max(bounds[2], cur_bounds[2])
                            bounds[3] = max(bounds[3], cur_bounds[3])

                    # Normal GeoJSON should have coordinates in CRS coordinates, i.e. it
                    # should be 1 projection unit/pixel.
                    projection = Projection(crs, 1, 1)
                    geometry = STGeometry(
                        projection,
                        shapely.box(*bounds).buffer(self.item_buffer_epsilon),
                        None,
                    )

            logger.debug(
                "VectorImporter.list_items: got bounds of %s: %s", path, geometry
            )
            items.append(
                VectorItem(path.name.split(".")[0], geometry, path.absolute().as_uri())
            )

        logger.debug("VectorImporter.list_items: discovered %d items", len(items))
        return items

    def ingest_item(
        self,
        tile_store: TileStoreWithLayer,
        item: Item,
        cur_geometries: list[STGeometry],
    ) -> None:
        """Ingest the specified local file item.

        Args:
            tile_store: the TileStore to ingest the data into.
            item: the Item to ingest
            cur_geometries: the geometries where the item is needed.
        """
        if tile_store.is_vector_ready(item):
            return

        assert isinstance(item, VectorItem)
        path = UPath(item.path_uri)

        aux_files: list[UPath] = []
        if path.name.split(".")[-1] == "shp":
            prefix = ".".join(path.name.split(".")[:-1])
            for ext in SHAPEFILE_AUX_EXTENSIONS:
                aux_files.append(path.parent / (prefix + ext))

        # TODO: move converting fiona file to list[Feature] to utility function.
        with get_upath_local(path, extra_paths=aux_files) as local_fname:
            with fiona.open(local_fname) as src:
                crs = CRS.from_wkt(src.crs.to_wkt())
                # Normal GeoJSON should have coordinates in CRS coordinates, i.e. it
                # should be 1 projection unit/pixel.
                projection = Projection(crs, 1, 1)

                features = []
                for feat in src:
                    features.append(
                        Feature.from_geojson(
                            projection,
                            {
                                "type": "Feature",
                                "geometry": dict(feat.geometry),
                                "properties": dict(feat.properties),
                            },
                        )
                    )

                tile_store.write_vector(item, features)


class LocalFiles(DataSource):
    """A data source for ingesting data from local files."""

    def __init__(
        self,
        src_dir: str,
        raster_item_specs: list[RasterItemSpec] | None = None,
        layer_type: LayerType | None = None,
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Initialize a new LocalFiles instance.

        Args:
            src_dir: source directory to ingest
            raster_item_specs: the specs to specify the raster items directly. If None,
                the raster items are automatically detected from the files in the
                source directory.
            layer_type: the layer type. It only needs to be set if the layer_config is
                missing from the context.
            context: the data source context. The layer config must be in the context.
        """
        if context.ds_path is not None:
            self.src_dir = join_upath(context.ds_path, src_dir)
        else:
            self.src_dir = UPath(src_dir)

        # Determine layer type.
        if context.layer_config is not None:
            self.layer_type = context.layer_config.type
        elif layer_type is not None:
            self.layer_type = layer_type
        else:
            raise ValueError(
                "layer type must be specified if the layer config is not in the context"
            )

        self.importer: Importer
        if self.layer_type == LayerType.RASTER:
            self.importer = RasterImporter(item_specs=raster_item_specs)
        elif self.layer_type == LayerType.VECTOR:
            self.importer = VectorImporter()
        else:
            raise ValueError(f"unknown layer type {self.layer_type}")

    @functools.cache
    def list_items(self) -> list[Item]:
        """Lists items from the source directory while maintaining a cache file."""
        cache_fname = self.src_dir / "summary.json"
        if not cache_fname.exists():
            logger.debug("cache at %s does not exist, listing items", cache_fname)
            items = self.importer.list_items(self.src_dir)
            serialized_items = [item.serialize() for item in items]
            with cache_fname.open("w") as f:
                json.dump(serialized_items, f)
            return items

        logger.debug("loading item list from cache at %s", cache_fname)
        with cache_fname.open() as f:
            serialized_items = json.load(f)
        return [
            self.deserialize_item(serialized_item)
            for serialized_item in serialized_items
        ]

    @functools.cache
    def _get_spatial_index(self) -> GridIndex:
        """Build an in-memory GridIndex over items in WGS84 for fast spatial lookup."""
        grid_index = GridIndex(1.0)
        for item in self.list_items():
            wgs84_geom = item.geometry.to_wgs84()
            grid_index.insert(wgs84_geom.shp.bounds, item)
        return grid_index

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[MatchedItemGroup[Item]]]:
        """Get a list of items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        grid_index = self._get_spatial_index()

        groups = []
        for geometry in geometries:
            wgs84_geometry = geometry.to_wgs84()
            cur_items = []
            for item in grid_index.query(wgs84_geometry.shp.bounds):
                if not item.geometry.intersects(geometry):
                    continue
                cur_items.append(item)

            cur_groups = rslearn.data_sources.utils.match_candidate_items_to_window(
                geometry, cur_items, query_config
            )
            groups.append(cur_groups)
        return groups

    def deserialize_item(self, serialized_item: dict) -> RasterItem | VectorItem:
        """Deserializes an item from JSON-decoded data."""
        if self.layer_type == LayerType.RASTER:
            return RasterItem.deserialize(serialized_item)
        elif self.layer_type == LayerType.VECTOR:
            return VectorItem.deserialize(serialized_item)
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        for item, cur_geometries in zip(items, geometries):
            self.importer.ingest_item(tile_store, item, cur_geometries)
