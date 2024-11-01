"""Data source for raster data on public Cloud Storage buckets."""

import json
import shutil
import urllib.request
from enum import Enum
from typing import Any

import osmium
import osmium.osm.types
import shapely
from upath import UPath

from rslearn.config import QueryConfig, VectorLayerConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource, Item
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.tile_stores import LayerMetadata, TileStore
from rslearn.utils import Feature, GridIndex, STGeometry
from rslearn.utils.fsspec import get_upath_local, join_upath


class FeatureType(Enum):
    """OpenStreetMap feature type."""

    NODE = "node"
    WAY = "way"
    RELATION = "relation"


class Filter:
    """Specifies filters corresponding to one category to extract from OSM data."""

    def __init__(
        self,
        feature_types: list[FeatureType] | None = None,
        tag_conditions: dict[str, list[str]] | None = None,
        tag_properties: dict[str, str] | None = None,
        to_geometry: str | None = None,
    ) -> None:
        """Create a new Filter instance.

        Args:
            feature_types: limit which types of features to match
            tag_conditions: for each entry (tag_name, values), only match features with
                that tag, and if values is not empty, where tag value matches some
                element of values.
            tag_properties: for each entry (tag_name, prop_name), save a property
                prop_name of output feature with value of the feature's tag_name tag
            to_geometry: output geometry as the specified type (Point, LineString, or
                Polygon). Otherwise defaults to Point or LineString.
        """
        self.feature_types = feature_types
        self.tag_conditions = tag_conditions
        self.tag_properties = tag_properties
        self.to_geometry = to_geometry

    @staticmethod
    def from_config(d: dict[str, Any]) -> "Filter":
        """Creates a Filter from a config dict.

        Args:
            d: the config dict

        Returns:
            the Filter object
        """
        kwargs: dict[str, Any] = {}
        if "feature_types" in d:
            kwargs["feature_types"] = [FeatureType(el) for el in d["feature_types"]]
        if "tag_conditions" in d:
            kwargs["tag_conditions"] = d["tag_conditions"]
        if "tag_properties" in d:
            kwargs["tag_properties"] = d["tag_properties"]
        if "to_geometry" in d:
            kwargs["to_geometry"] = d["to_geometry"]
        return Filter(**kwargs)

    def match_tags(self, tags: dict[str, str]) -> bool:
        """Returns whether this filter matches based on the tags."""
        if not self.tag_conditions:
            return True
        for tag_name, values in self.tag_conditions.items():
            if tag_name not in tags:
                return False
            if not values:
                continue
            if tags[tag_name] not in values:
                return False
        return True

    def get_properties(
        self, tags: dict[str, str], category_name: str
    ) -> dict[str, Any]:
        """Returns properties for the output feature."""
        properties = {"category": category_name}
        if self.tag_properties:
            for tag_name, prop_name in self.tag_properties.items():
                if tag_name not in tags:
                    continue
                properties[prop_name] = tags[tag_name]
        return properties


class BoundsHandler(osmium.SimpleHandler):
    """An osmium handler for computing the bounds of an input file."""

    def __init__(self) -> None:
        """Initialize a new BoundsHandler."""
        osmium.SimpleHandler.__init__(self)
        self.bounds: tuple[float, float, float, float] = (180, 90, -180, -90)

    def node(self, n: osmium.osm.types.Node) -> None:
        """Handle nodes and update the computed bounds."""
        lon = n.location.lon
        lat = n.location.lat
        self.bounds = (
            min(self.bounds[0], lon),
            min(self.bounds[1], lat),
            max(self.bounds[2], lon),
            max(self.bounds[3], lat),
        )


class OsmHandler(osmium.SimpleHandler):
    """An osmium handler for recording the vector data in an input file."""

    def __init__(
        self,
        categories: dict[str, Filter],
        geometries: list[STGeometry],
        grid_size: float = 0.03,
        padding: float = 0.03,
    ) -> None:
        """Initialize a new OsmHandler.

        Args:
            categories: a map from category name to a corresponding Filter. If an OSM
                feature matches the filter, then it is converted to a vector Feature
                under that category.
            geometries: only consider features falling in these geometries.
            grid_size: grid size of grid index created over the geometries
            padding: padding added to the geometries so that enough points along OSM
                features intersecting a geometry are retained.
        """
        osmium.SimpleHandler.__init__(self)

        self.categories = categories

        geometries = [
            geometry.to_projection(WGS84_PROJECTION) for geometry in geometries
        ]
        self.grid_index = GridIndex(grid_size)
        for geometry in geometries:
            bounds = geometry.shp.bounds
            # Add some padding so that vertices that are relevant to an object
            # intersecting a geometry are included even if the vertex is outside the
            # geometry.
            bounds = (
                bounds[0] - padding,
                bounds[1] - padding,
                bounds[2] + padding,
                bounds[3] + padding,
            )
            self.grid_index.insert(bounds, 1)

        self.cached_nodes: dict = {}
        self.cached_ways: dict = {}

        self.features: list[Feature] = []

    def node(self, n: osmium.osm.types.Node) -> None:
        """Handle nodes."""
        # Check if node is relevant to our geometries.
        lon = n.location.lon
        lat = n.location.lat
        matches = self.grid_index.query((lon, lat, lon, lat))
        if not matches:
            return

        self.cached_nodes[n.id] = (lon, lat)

        # Add feature if there's a match.
        for category_name, f in self.categories.items():
            if f.feature_types and FeatureType.NODE not in f.feature_types:
                continue
            if not f.match_tags(n.tags):
                continue
            assert not f.to_geometry or f.to_geometry == "Point"
            shp = shapely.Point(lon, lat)
            feat = Feature(
                STGeometry(WGS84_PROJECTION, shp, None),
                f.get_properties(n.tags, category_name),
            )
            self.features.append(feat)

    def _get_way_coords(self, node_ids: list[int]) -> list:
        coords = []
        for id in node_ids:
            if id not in self.cached_nodes:
                continue
            coords.append(self.cached_nodes[id])
        return coords

    def way(self, w: osmium.osm.types.Way) -> None:
        """Handle ways."""
        # Collect nodes, skip if too few.
        node_ids = [member.ref for member in w.nodes]
        coords = self._get_way_coords(node_ids)
        if len(coords) == 0:
            return

        self.cached_ways[w.id] = node_ids

        if len(coords) < 2:
            return

        for category_name, f in self.categories.items():
            if f.feature_types and FeatureType.WAY not in f.feature_types:
                continue
            if not f.match_tags(w.tags):
                continue
            if not f.to_geometry or f.to_geometry == "LineString":
                shp = shapely.LineString(coords)
            elif f.to_geometry == "Point":
                shp = shapely.LineString(coords).centroid
            elif f.to_geometry == "Polygon":
                if len(coords) < 3:
                    continue
                shp = shapely.Polygon(coords)
            if not shp.is_valid:
                continue
            feat = Feature(
                STGeometry(WGS84_PROJECTION, shp, None),
                f.get_properties(w.tags, category_name),
            )
            self.features.append(feat)

    def match_relation(self, r: osmium.osm.types.Relation) -> None:
        """Handle relations."""
        # Collect ways and distinguish exterior vs holes, skip if none found.
        exterior_ways = []
        interior_ways = []
        for member in r.members:
            if member.ref not in self.cached_ways:
                continue
            way = self.cached_ways[member.ref]
            if member.role == "outer":
                exterior_ways.append(way)
            else:
                interior_ways.append(way)

        if len(exterior_ways) == 0:
            return

        # Now skip if it doesn't match anything.
        needed = False
        for f in self.categories.values():
            if f.feature_types and FeatureType.RELATION not in f.feature_types:
                continue
            if not f.match_tags(r.tags):
                continue
            needed = True
            break
        if not needed:
            return

        # Merge the ways in case some exterior/interior polygons are split into
        # multiple ways.
        # And convert them from node IDs to coordinates.
        def get_polygons(ways: list) -> list:
            polygons: list[list[int]] = []
            for way in ways:
                # Attempt to match the way to an existing polygon.
                # We assume the ways are ordered so that we can attach the way to the
                # beginning or end of a previous way.
                matched = False
                for i, polygon in enumerate(polygons):
                    if polygon[0] == polygon[-1]:
                        # Polygon is already closed, don't match any more.
                        continue
                    if polygon[-1] == way[0]:
                        polygon.extend(way)
                        matched = True
                        break
                    if polygon[-1] == way[-1]:
                        polygon.extend(reversed(way))
                        matched = True
                        break
                    if polygon[0] == way[0]:
                        polygons[i] = list(reversed(way)) + polygon
                        matched = True
                        break
                    if polygon[0] == way[-1]:
                        polygons[i] = way + polygon
                        matched = True
                        break
                if matched:
                    continue

                # Need to create new polygon.
                polygons.append(way)

            # Now convert to the actual coordinates.
            # We do this after because some node IDs may not exist and we don't want
            # that to mess up the polygon connection.
            coords = []
            for polygon in polygons:
                cur = self._get_way_coords(polygon)
                if len(cur) < 3:
                    continue
                coords.append(cur)
            return coords

        exteriors = get_polygons(exterior_ways)
        interiors = get_polygons(interior_ways)

        if not exteriors:
            return

        for category_name, f in self.categories.items():
            if f.feature_types and FeatureType.RELATION not in f.feature_types:
                continue
            if not f.match_tags(r.tags):
                continue
            assert not f.to_geometry or f.to_geometry == "Polygon"
            for exterior in exteriors:
                exterior_polygon = shapely.Polygon(exterior)
                interior_polygons = [
                    shapely.Polygon(interior) for interior in interiors
                ]
                cur_interiors = []
                for interior_polygon, interior in zip(interior_polygons, interiors):
                    if not interior_polygon.is_valid:
                        continue
                    if not exterior_polygon.contains(interior_polygon):
                        continue
                    cur_interiors.append(interior)
                shp = shapely.Polygon(exterior, cur_interiors)
                if not shp.is_valid:
                    continue
                feat = Feature(
                    STGeometry(WGS84_PROJECTION, shp, None),
                    f.get_properties(r.tags, category_name),
                )
                self.features.append(feat)


class OsmItem(Item):
    """An item in the OpenStreetMap data source."""

    def __init__(self, name: str, geometry: STGeometry, path_uri: str):
        """Creates a new OsmItem.

        Args:
            name: unique name of the item
            geometry: the spatial and temporal extent of the item
            path_uri: the path URI of the pbf
        """
        super().__init__(name, geometry)
        self.path_uri = path_uri

    def serialize(self) -> dict:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["path_uri"] = self.path_uri
        return d

    @staticmethod
    def deserialize(d: dict) -> "OsmItem":
        """Deserializes an item from a JSON-decoded dictionary."""
        item = super(OsmItem, OsmItem).deserialize(d)
        return OsmItem(name=item.name, geometry=item.geometry, path_uri=d["path_uri"])


class OpenStreetMap(DataSource[OsmItem]):
    """A data source for OpenStreetMap data from PBF file.

    An existing local PBF file can be used, or if the provided path doesn't exist, then
    the global OSM PBF will be downloaded.

    This data source uses a single item. If more windows are added, data in the
    TileStore will need to be completely re-computed.
    """

    planet_pbf_url = "https://planet.openstreetmap.org/pbf/planet-latest.osm.pbf"

    def __init__(
        self,
        config: VectorLayerConfig,
        pbf_fnames: list[UPath],
        bounds_fname: UPath,
        categories: dict[str, Filter],
    ):
        """Initialize a new Sentinel2 instance.

        Args:
            config: the configuration of this layer.
            pbf_fnames: the PBF filenames to read from. If a single filename is
                provided and it doesn't exist, the latest planet PBF will be downloaded
                there.
            bounds_fname: filename where the bounds of the PBF are cached.
            categories: dictionary of (category name, filter). Features that match the
                filter will be emitted under the corresponding category.
        """
        self.config = config
        self.pbf_fnames = pbf_fnames
        self.bounds_fname = bounds_fname
        self.categories = categories

        if len(self.pbf_fnames) == 1 and not self.pbf_fnames[0].exists():
            print(
                "Downloading planet.osm.pbf from "
                + f"{self.planet_pbf_url} to {self.pbf_fnames[0]}"
            )
            with urllib.request.urlopen(self.planet_pbf_url) as response:
                with self.pbf_fnames[0].open("wb") as f:
                    shutil.copyfileobj(response, f)

        # Detect bounds of each pbf file if needed.
        self.pbf_bounds = self._get_pbf_bounds()

    @staticmethod
    def from_config(config: VectorLayerConfig, ds_path: UPath) -> "OpenStreetMap":
        """Creates a new OpenStreetMap instance from a configuration dictionary."""
        if config.data_source is None:
            raise ValueError("data_source is required")
        d = config.data_source.config_dict
        categories = {
            category_name: Filter.from_config(filter_config_dict)
            for category_name, filter_config_dict in d["categories"].items()
        }
        pbf_fnames = [join_upath(ds_path, pbf_fname) for pbf_fname in d["pbf_fnames"]]
        bounds_fname = join_upath(ds_path, d["bounds_fname"])
        return OpenStreetMap(
            config=config,
            pbf_fnames=pbf_fnames,
            bounds_fname=bounds_fname,
            categories=categories,
        )

    def _get_pbf_bounds(self) -> list[tuple[float, float, float, float]]:
        # Determine WGS84 bounds of each PBF file by processing them through
        # BoundsHandler.
        if not self.bounds_fname.exists():
            pbf_bounds = []
            for pbf_fname in self.pbf_fnames:
                print(f"detecting bounds of {pbf_fname}")
                handler = BoundsHandler()
                with get_upath_local(pbf_fname) as local_fname:
                    handler.apply_file(local_fname)
                pbf_bounds.append(handler.bounds)

            with self.bounds_fname.open("w") as f:
                json.dump(pbf_bounds, f)

        else:
            with self.bounds_fname.open() as f:
                pbf_bounds = json.load(f)

        return pbf_bounds

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[OsmItem]]]:
        """Get a list of items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        items = []
        for pbf_fname, bounds in zip(self.pbf_fnames, self.pbf_bounds):
            items.append(
                OsmItem(
                    pbf_fname.name,
                    STGeometry(WGS84_PROJECTION, shapely.box(*bounds), None),
                    pbf_fname.absolute().as_uri(),
                )
            )

        wgs84_geometries = [
            geometry.to_projection(WGS84_PROJECTION) for geometry in geometries
        ]
        groups = []
        for geometry in wgs84_geometries:
            cur_groups = match_candidate_items_to_window(geometry, items, query_config)
            groups.append(cur_groups)
        return groups

    def deserialize_item(self, serialized_item: Any) -> OsmItem:
        """Deserializes an item from JSON-decoded data."""
        return OsmItem.deserialize(serialized_item)

    def ingest(
        self,
        tile_store: TileStore,
        items: list[OsmItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        item_names = [item.name for item in items]
        item_names.sort()
        for cur_item, cur_geometries in zip(items, geometries):
            print(
                f"ingesting osm item {cur_item.name} "
                + f"with {len(cur_geometries)} geometries"
            )
            handler = OsmHandler(self.categories, cur_geometries)
            with get_upath_local(UPath(cur_item.path_uri)) as local_fname:
                handler.apply_file(local_fname)

            projections = set()
            for geometry in cur_geometries:
                projection, _ = self.config.get_final_projection_and_bounds(
                    geometry.projection, None
                )
                projections.add(projection)

            for projection in projections:
                features = [feat.to_projection(projection) for feat in handler.features]
                layer = tile_store.create_layer(
                    (cur_item.name, str(projection)),
                    LayerMetadata(projection, None, {}),
                )
                layer.write_vector(features)
