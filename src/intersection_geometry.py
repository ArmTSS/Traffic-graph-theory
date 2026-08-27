"""Projected Shapely geometry helpers for a selected OSM intersection."""

from dataclasses import dataclass

from shapely.geometry import Point


@dataclass(frozen=True)
class IntersectionGeometry:
    center: Point
    zone: object
    nearby_edges: object
    projected_crs: object


def extract_intersection_geometry(
    nodes, edges, node_id, radius_m: float = 40
) -> IntersectionGeometry:
    """Buffer a node in metres and select road geometries entering that zone.

    GeoPandas performs the projection so the buffer is never calculated in
    latitude/longitude degrees.
    """
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    if node_id not in nodes.index:
        raise KeyError(f"OSM node {node_id!r} was not found")

    node = nodes.loc[node_id]
    source_crs = nodes.crs or "EPSG:4326"
    projected_crs = nodes.estimate_utm_crs()
    projected_nodes = nodes.to_crs(projected_crs)
    projected_edges = edges.to_crs(projected_crs)
    center = projected_nodes.loc[node_id].geometry
    zone = center.buffer(radius_m)
    nearby_edges = projected_edges[projected_edges.geometry.intersects(zone)].copy()
    return IntersectionGeometry(center, zone, nearby_edges, projected_crs)
