"""OpenStreetMap road-network loading for local intersection studies.

OSM describes geography and road attributes, not traffic volumes.  This
module keeps those attributes intact and leaves demand to the simulation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
from shapely.geometry import Point


@dataclass(frozen=True)
class OSMNetwork:
    graph: nx.MultiDiGraph
    nodes: Any
    edges: Any
    source: str


@dataclass(frozen=True)
class IntersectionGeometry:
    center: Point
    zone: object
    nearby_edges: object
    projected_crs: object


def _import_osmnx():
    try:
        import osmnx as ox
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise RuntimeError("OSM loading requires the 'osmnx' package") from exc
    return ox


def edge_attribute(edge: dict, name: str, fallback):
    """Read an optional OSM edge attribute without assuming uniform tagging."""
    value = edge.get(name, fallback)
    if value is None or value == "":
        return fallback
    if isinstance(value, (list, tuple)):
        value = value[0]
    return value


def load_osm_network(
    *,
    place: str | None = "Bangkok, Thailand",
    latitude: float | None = None,
    longitude: float | None = None,
    radius_m: float = 1000,
) -> OSMNetwork:
    """Load a drivable local OSM graph by place or point and return GeoDataFrames.

    A point query is preferred for repeatable local studies.  OSMnx is imported
    lazily so synthetic experiments do not require network access.
    """
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    if (latitude is None) != (longitude is None):
        raise ValueError("latitude and longitude must be supplied together")
    if latitude is None and not place:
        raise ValueError("provide a place or latitude/longitude")

    ox = _import_osmnx()

    try:
        if latitude is not None:
            source = f"point:{latitude},{longitude};radius_m:{radius_m}"
            graph = ox.graph_from_point(
                (latitude, longitude), dist=radius_m, network_type="drive"
            )
        else:
            source = f"place:{place}"
            graph = ox.graph_from_place(place, network_type="drive")
        nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    except Exception as exc:
        raise RuntimeError(f"Unable to download OSM network ({source})") from exc

    return OSMNetwork(graph=graph, nodes=nodes, edges=edges, source=source)


def save_osm_network(network: OSMNetwork, filepath: str | Path) -> Path:
    """Persist an OSM graph as GraphML so future runs work offline."""
    ox = _import_osmnx()
    target = Path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)
    graph = network.graph.copy()
    graph.graph["traffic_graph_source"] = network.source
    try:
        ox.save_graphml(graph, filepath=target)
    except Exception as exc:
        raise RuntimeError(f"Unable to save OSM graph to {target}") from exc
    return target


def load_osm_network_file(filepath: str | Path) -> OSMNetwork:
    """Load a previously saved GraphML road network without downloading it."""
    ox = _import_osmnx()
    source_path = Path(filepath)
    if not source_path.is_file():
        raise ValueError(f"saved OSM graph does not exist: {source_path}")
    try:
        graph = ox.load_graphml(filepath=source_path)
        nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    except Exception as exc:
        raise RuntimeError(f"Unable to load OSM graph from {source_path}") from exc
    source = graph.graph.get("traffic_graph_source", f"file:{source_path}")
    return OSMNetwork(graph=graph, nodes=nodes, edges=edges, source=str(source))


def extract_intersection_geometry(
    nodes, edges, node_id, radius_m: float = 40
) -> IntersectionGeometry:
    """Buffer an OSM node in metres and select roads entering that zone."""
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    if node_id not in nodes.index:
        raise KeyError(f"OSM node {node_id!r} was not found")
    projected_crs = nodes.estimate_utm_crs()
    projected_nodes = nodes.to_crs(projected_crs)
    projected_edges = edges.to_crs(projected_crs)
    center = projected_nodes.loc[node_id].geometry
    zone = center.buffer(radius_m)
    nearby_edges = projected_edges[projected_edges.geometry.intersects(zone)].copy()
    return IntersectionGeometry(center, zone, nearby_edges, projected_crs)


def intersection_candidates(graph: nx.Graph, minimum_degree: int = 3) -> list[dict]:
    """Rank likely intersections without treating OSM traffic tags as counts."""
    if minimum_degree < 2:
        raise ValueError("minimum_degree must be at least 2")
    candidates = []
    for node, data in graph.nodes(data=True):
        degree = graph.degree(node)
        if degree >= minimum_degree:
            candidates.append(
                {
                    "node_id": node,
                    "degree": degree,
                    "in_degree": graph.in_degree(node)
                    if graph.is_directed()
                    else degree,
                    "out_degree": graph.out_degree(node)
                    if graph.is_directed()
                    else degree,
                    "latitude": data.get("y"),
                    "longitude": data.get("x"),
                }
            )
    return sorted(candidates, key=lambda item: item["degree"], reverse=True)
