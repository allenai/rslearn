"""Data source for raster or vector data in local files."""

from typing import Any

import fiona
import rasterio
import shapely
import shapely.geometry
from class_registry import ClassRegistry
from rasterio.crs import CRS
from upath import UPath

import rslearn.data_sources.utils
from rslearn.config import LayerConfig, VectorLayerConfig
from rslearn.const import SHAPEFILE_AUX_EXTENSIONS, WGS84_PROJECTION
from rslearn.tile_stores import LayerMetadata, PrefixedTileStore, TileStore
from rslearn.utils import Feature, Projection, STGeometry
from rslearn.utils.fsspec import get_upath_local, join_upath

from .data_source import DataSource, Item, QueryConfig
from .raster_source import get_needed_projections, ingest_raster


class LocalFileItem(Item):
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
    def deserialize(d: dict) -> Item:
        """Deserializes an item from a JSON-decoded dictionary."""
        item = super(LocalFileItem, LocalFileItem).deserialize(d)
        return LocalFileItem(
            name=item.name, geometry=item.geometry, path_uri=d["path_uri"]
        )


Importers = ClassRegistry()


class Importer:
    """An abstract class for importing data from local files."""

    def get_item(config: LayerConfig, path: UPath) -> LocalFileItem:
        """Get a LocalFileItem for this file that can then be ingested.

        Args:
            config: the configuration of the layer.
            path: the file path
        """
        raise NotImplementedError

    def ingest_item(
        config: LayerConfig,
        tile_store: TileStore,
        item: LocalFileItem,
        cur_geometries: list[STGeometry],
    ):
        """Ingest the specified local file.

        Args:
            config: the configuration of the layer.
            tile_store: the TileStore to ingest the data into.
            item: the LocalFileItem
            cur_geometries: the geometries where the item is needed.
        """
        raise NotImplementedError


@Importers.register("raster")
class RasterImporter(Importer):
    """An Importer for raster data."""

    def get_item(config: LayerConfig, path: UPath) -> Item:
        """Get a LocalFileItem for this file that can then be ingested.

        Args:
            config: the configuration of the layer.
            path: the file path
        """
        # Get geometry from the raster file.
        # We assume files are readable with rasterio.
        with path.open("rb") as f:
            with rasterio.open(f) as src:
                crs = src.crs
                left = src.transform.c
                top = src.transform.f
                # Resolutions in projection units per pixel.
                x_resolution = src.transform.a
                y_resolution = src.transform.e
                start = (int(left / x_resolution), int(top / y_resolution))
                shp = shapely.box(
                    start[0], start[1], start[0] + src.width, start[1] + src.height
                )
                projection = Projection(crs, x_resolution, y_resolution)
                geometry = STGeometry(projection, shp, None)

        return LocalFileItem(
            path.name.split(".")[0], geometry, path.absolute().as_uri()
        )

    def ingest_item(
        self,
        config: LayerConfig,
        tile_store: TileStore,
        item: LocalFileItem,
        cur_geometries: list[STGeometry],
    ):
        """Ingest the specified local file.

        Args:
            config: the configuration of the layer.
            tile_store: the TileStore to ingest the data into.
            item: the LocalFileItem
            cur_geometries: the geometries where the item is needed.
        """
        path = UPath(item.path_uri)
        with path.open("rb") as f:
            with rasterio.open(f) as src:
                bands = [f"B{idx+1}" for idx in range(src.count)]
                tile_store = PrefixedTileStore(tile_store, ("_".join(bands),))
                needed_projections = get_needed_projections(
                    tile_store, bands, config.band_sets, cur_geometries
                )
                if not needed_projections:
                    return

                for projection in needed_projections:
                    ingest_raster(
                        tile_store=tile_store,
                        raster=src,
                        projection=projection,
                        time_range=item.geometry.time_range,
                        layer_config=config,
                    )


@Importers.register("vector")
class VectorImporter(Importer):
    """An Importer for vector data."""

    def get_item(config: LayerConfig, path: UPath) -> Item:
        """Get a LocalFileItem for this file that can then be ingested.

        Args:
            config: the configuration of the layer.
            path: the file path
        """
        # Get the bounds of the features in the vector file, which we assume fiona can
        # read.
        # For shapefile, to open it we need to copy all the aux files.
        aux_files: list[UPath] = []
        if path.name.split(".")[-1] == "shp":
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

                projection = Projection(crs, 1, 1)
                geometry = STGeometry(projection, shapely.box(*bounds), None)

        return LocalFileItem(
            path.name.split(".")[0], geometry, path.absolute().as_uri()
        )

    def ingest_item(
        self,
        config: LayerConfig,
        tile_store: TileStore,
        item: LocalFileItem,
        cur_geometries: list[STGeometry],
    ):
        """Ingest the specified local file.

        Args:
            config: the configuration of the layer.
            tile_store: the TileStore to ingest the data into.
            item: the LocalFileItem
            cur_geometries: the geometries where the item is needed.
        """
        assert isinstance(config, VectorLayerConfig)
        path = UPath(item.path_uri)

        aux_files: list[UPath] = []
        if path.name.split(".")[-1] == "shp":
            prefix = ".".join(path.name.split(".")[:-1])
            for ext in SHAPEFILE_AUX_EXTENSIONS:
                aux_files.append(path.parent / (prefix + ext))

        # TODO: move converting fiona file to list[Feature] to utility function.
        # TODO: don't assume WGS-84 projection here.
        with get_upath_local(path, extra_paths=aux_files) as local_fname:
            with fiona.open(local_fname) as src:
                features = []
                for feat in src:
                    features.append(
                        Feature.from_geojson(
                            WGS84_PROJECTION,
                            {
                                "type": "Feature",
                                "geometry": dict(feat.geometry),
                                "properties": dict(feat.properties),
                            },
                        )
                    )

                projections = set()
                for geometry in cur_geometries:
                    projection, _ = config.get_final_projection_and_bounds(
                        geometry.projection, None
                    )
                    projections.add(projection)

                for projection in projections:
                    cur_features = [feat.to_projection(projection) for feat in features]
                    layer = tile_store.create_layer(
                        (str(projection),),
                        LayerMetadata(projection, None, {}),
                    )
                    layer.write_vector(cur_features)


class LocalFiles(DataSource):
    """A data source for ingesting data from local files."""

    def __init__(self, config: LayerConfig, src_dir: UPath) -> None:
        """Initialize a new LocalFiles instance.

        Args:
            config: configuration for this layer.
            src_dir: source directory to ingest
        """
        self.config = config
        self.src_dir = src_dir

        self.importer = Importers[config.layer_type.value]

        file_paths = self.src_dir.glob("**/*.*")
        self.items = []
        for path in file_paths:
            item = self.importer.get_item(path)
            self.items.append(item)

    @staticmethod
    def from_config(config: LayerConfig, ds_path: UPath) -> "LocalFiles":
        """Creates a new LocalFiles instance from a configuration dictionary."""
        d = config.data_source.config_dict
        return LocalFiles(config=config, src_dir=join_upath(ds_path, d["src_dir"]))

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[Item]]]:
        """Get a list of items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        groups = []
        for geometry in geometries:
            cur_items = []
            for item in self.items:
                if not item.geometry.intersects(geometry):
                    continue
                cur_items.append(item)

            cur_groups = rslearn.data_sources.utils.match_candidate_items_to_window(
                geometry, cur_items, query_config
            )
            groups.append(cur_groups)
        return groups

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return LocalFileItem.deserialize(serialized_item)

    def ingest(
        self,
        tile_store: TileStore,
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
            cur_tile_store = PrefixedTileStore(tile_store, (item.name,))
            self.importer.ingest_item(self.config, cur_tile_store, item, cur_geometries)
