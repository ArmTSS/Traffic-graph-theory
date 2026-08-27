"""OpenStreetMap road-network loading for local intersection studies.

OSM describes geography and road attributes, not traffic volumes.  This
module keeps those attributes intact and leaves demand to the simulation.
"""

from dataclasses import dataclass
from typing import Any

import networkx as nx


@dataclass(frozen=True)
class OSMNetwork:
    graph: nx.MultiDiGraph
    nodes: Any
    edges: Any
    source: str


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

    try:
        import osmnx as ox
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise RuntimeError("OSM loading requires the 'osmnx' package") from exc

    try:
        if latitude is not None:
            graph = ox.graph_from_point(
                (latitude, longitude), dist=radius_m, network_type="drive"
            )
            source = f"point:{latitude},{longitude};radius_m:{radius_m}"
        else:
            graph = ox.graph_from_place(place, network_type="drive")
            source = f"place:{place}"
        nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    except Exception as exc:
        raise RuntimeError(f"Unable to download OSM network ({source})") from exc

    return OSMNetwork(graph=graph, nodes=nodes, edges=edges, source=source)


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
