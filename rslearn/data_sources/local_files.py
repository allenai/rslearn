"""Data source for raster or vector data in local files."""

from typing import Any, Generic, TypeVar

import fiona
import rasterio
import shapely
import shapely.geometry
from class_registry import ClassRegistry
from rasterio.crs import CRS
from upath import UPath

import rslearn.data_sources.utils
from rslearn.config import LayerConfig, LayerType, RasterLayerConfig, VectorLayerConfig
from rslearn.const import SHAPEFILE_AUX_EXTENSIONS
from rslearn.tile_stores import LayerMetadata, PrefixedTileStore, TileStore
from rslearn.utils import Feature, Projection, STGeometry
from rslearn.utils.fsspec import get_upath_local, join_upath

from .data_source import DataSource, Item, QueryConfig
from .raster_source import get_needed_projections, ingest_raster

Importers = ClassRegistry()

ItemType = TypeVar("ItemType", bound=Item)
LayerConfigType = TypeVar("LayerConfigType", bound=LayerConfig)
ImporterType = TypeVar("ImporterType", bound="Importer")


class Importer(Generic[ItemType, LayerConfigType]):
    """An abstract base class for importing data from local files."""

    def list_items(self, config: LayerConfigType, src_dir: UPath) -> list[ItemType]:
        """Extract a list of Items from the source directory.

        Args:
            config: the configuration of the layer.
            src_dir: the source directory.
        """
        raise NotImplementedError

    def ingest_item(
        self,
        config: LayerConfigType,
        tile_store: TileStore,
        item: ItemType,
        cur_geometries: list[STGeometry],
    ) -> None:
        """Ingest the specified local file item.

        Args:
            config: the configuration of the layer.
            tile_store: the TileStore to ingest the data into.
            item: the Item to ingest
            cur_geometries: the geometries where the item is needed.
        """
        raise NotImplementedError


class RasterItemSpec:
    """Representation of configuration that directly specifies the available items."""

    def __init__(
        self,
        fnames: list[UPath],
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

    @staticmethod
    def from_config(src_dir: UPath, d: dict[str, Any]) -> "RasterItemSpec":
        """Decode a dict into a RasterItemSpec.

        Args:
            src_dir: the source directory.
            d: the configuration dict.

        Returns:
            the RasterItemSpec.
        """
        kwargs = dict(
            fnames=[join_upath(src_dir, suffix) for suffix in d["fnames"]],
            bands=d["bands"],
        )
        if "name" in d:
            kwargs["name"] = d["name"]
        return RasterItemSpec(**kwargs)

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
            fnames=[UPath(s) for s in d["fnames"]],
            bands=d["bands"],
            name=d["name"],
        )


class RasterItem(Item):
    """An item corresponding to a local file."""

    def __init__(self, name: str, geometry: STGeometry, spec: RasterItemSpec):
        """Creates a new LocalFileItem.

        Args:
            name: unique name of the item
            geometry: the spatial and temporal extent of the item
            spec: the RasterItemSpec that specifies the filename(s) and bands.
        """
        super().__init__(name, geometry)
        self.spec = spec

    def serialize(self) -> dict:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["spec"] = self.spec.serialize()
        return d

    @staticmethod
    def deserialize(d: dict) -> "RasterItem":
        """Deserializes an item from a JSON-decoded dictionary."""
        item = super(RasterItem, RasterItem).deserialize(d)
        spec = RasterItemSpec.deserialize(d["spec"])
        return RasterItem(name=item.name, geometry=item.geometry, spec=spec)


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


@Importers.register("raster")
class RasterImporter(Importer):
    """An Importer for raster data."""

    def list_items(self, config: LayerConfig, src_dir: UPath) -> list[RasterItem]:
        """Extract a list of Items from the source directory.

        Args:
            config: the configuration of the layer.
            src_dir: the source directory.
        """
        item_specs: list[RasterItemSpec] = []
        if config.data_source is None:
            raise ValueError("RasterImporter requires a data source config")
        # See if user has provided the item specs directly.
        if "item_specs" in config.data_source.config_dict:
            for spec_dict in config.data_source.config_dict["item_specs"]:
                spec = RasterItemSpec.from_config(src_dir, spec_dict)
                item_specs.append(spec)
        else:
            # Otherwise we need to list files and assume each one is separate.
            # And we'll need to autodetect the bands later.
            file_paths = src_dir.glob("**/*.*")
            for path in file_paths:
                spec = RasterItemSpec(fnames=[path], bands=None)
                item_specs.append(spec)

        items = []
        for spec in item_specs:
            # Get geometry from the first raster file.
            # We assume files are readable with rasterio.
            with spec.fnames[0].open("rb") as f:
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

            if spec.name:
                item_name = spec.name
            else:
                item_name = spec.fnames[0].name.split(".")[0]

            items.append(RasterItem(item_name, geometry, spec))

        return items

    def ingest_item(
        self,
        config: RasterLayerConfig,
        tile_store: TileStore,
        item: RasterItem,
        cur_geometries: list[STGeometry],
    ) -> None:
        """Ingest the specified local file item.

        Args:
            config: the configuration of the layer.
            tile_store: the TileStore to ingest the data into.
            item: the RasterItem to ingest
            cur_geometries: the geometries where the item is needed.
        """
        for file_idx, fname in enumerate(item.spec.fnames):
            with fname.open("rb") as f:
                with rasterio.open(f) as src:
                    if item.spec.bands:
                        bands = item.spec.bands[file_idx]
                    else:
                        bands = [f"B{band_idx+1}" for band_idx in range(src.count)]
                    cur_tile_store = PrefixedTileStore(tile_store, ("_".join(bands),))
                    needed_projections = get_needed_projections(
                        cur_tile_store, bands, config.band_sets, cur_geometries
                    )
                    if not needed_projections:
                        return

                    for projection in needed_projections:
                        ingest_raster(
                            tile_store=cur_tile_store,
                            raster=src,
                            projection=projection,
                            time_range=item.geometry.time_range,
                            layer_config=config,
                        )


@Importers.register("vector")
class VectorImporter(Importer):
    """An Importer for vector data."""

    # We need some buffer around GeoJSON bounds in case it just contains one point.
    item_buffer_epsilon = 1e-4

    def list_items(self, config: LayerConfig, src_dir: UPath) -> list[VectorItem]:
        """Extract a list of Items from the source directory.

        Args:
            config: the configuration of the layer.
            src_dir: the source directory.
        """
        file_paths = src_dir.glob("**/*.*")
        items: list[VectorItem] = []

        for path in file_paths:
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

                    # Normal GeoJSON should have coordinates in CRS coordinates, i.e. it
                    # should be 1 projection unit/pixel.
                    projection = Projection(crs, 1, 1)
                    geometry = STGeometry(
                        projection,
                        shapely.box(*bounds).buffer(self.item_buffer_epsilon),
                        None,
                    )

            items.append(
                VectorItem(path.name.split(".")[0], geometry, path.absolute().as_uri())
            )

        return items

    def ingest_item(
        self,
        config: VectorLayerConfig,
        tile_store: TileStore,
        item: VectorItem,
        cur_geometries: list[STGeometry],
    ) -> None:
        """Ingest the specified local file item.

        Args:
            config: the configuration of the layer.
            tile_store: the TileStore to ingest the data into.
            item: the Item to ingest
            cur_geometries: the geometries where the item is needed.
        """
        if not isinstance(config, VectorLayerConfig):
            raise ValueError("VectorImporter requires a VectorLayerConfig")

        needed_projections = set()
        for geometry in cur_geometries:
            projection, _ = config.get_final_projection_and_bounds(
                geometry.projection, None
            )
            ts_layer = tile_store.get_layer((str(projection),))
            if ts_layer and ts_layer.get_metadata().properties.get("completed"):
                continue
            needed_projections.add(projection)

        if not needed_projections:
            return

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

                for projection in needed_projections:
                    cur_features = [feat.to_projection(projection) for feat in features]
                    layer = tile_store.create_layer(
                        (str(projection),),
                        LayerMetadata(projection, None, {}),
                    )
                    layer.write_vector(cur_features)
                    layer.set_property("completed", True)


class LocalFiles(DataSource):
    """A data source for ingesting data from local files."""

    def __init__(
        self,
        config: LayerConfig,
        src_dir: UPath,
    ) -> None:
        """Initialize a new LocalFiles instance.

        Args:
            config: configuration for this layer.
            src_dir: source directory to ingest
        """
        self.config = config

        self.importer = Importers[config.layer_type.value]
        self.src_dir = src_dir

        self.items = self.importer.list_items(self.config, src_dir)

    @staticmethod
    def from_config(config: LayerConfig, ds_path: UPath) -> "LocalFiles":
        """Creates a new LocalFiles instance from a configuration dictionary."""
        if config.data_source is None:
            raise ValueError("LocalFiles data source requires a data source config")
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

    def deserialize_item(self, serialized_item: Any) -> RasterItem | VectorItem:
        """Deserializes an item from JSON-decoded data."""
        if self.config.layer_type == LayerType.RASTER:
            return RasterItem.deserialize(serialized_item)
        elif self.config.layer_type == LayerType.VECTOR:
            return VectorItem.deserialize(serialized_item)
        else:
            raise ValueError(f"Unknown layer type: {self.config.layer_type}")

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
