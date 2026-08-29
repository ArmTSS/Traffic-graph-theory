"""Extract auditable four-way intersection candidates from an OSM road graph."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def _values(value: Any) -> set[str]:
    if value is None or value == "":
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if item not in (None, "")}
    return {str(value)}


def _bearing(latitude_1, longitude_1, latitude_2, longitude_2) -> float:
    """Return initial compass bearing from point 1 to point 2."""
    lat_1, lat_2 = math.radians(latitude_1), math.radians(latitude_2)
    delta_lon = math.radians(longitude_2 - longitude_1)
    y = math.sin(delta_lon) * math.cos(lat_2)
    x = math.cos(lat_1) * math.sin(lat_2) - (
        math.sin(lat_1) * math.cos(lat_2) * math.cos(delta_lon)
    )
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def _edge_records(graph, node, neighbor) -> list[dict]:
    records = []
    for start, end in ((node, neighbor), (neighbor, node)):
        edge_data = graph.get_edge_data(start, end, default={})
        records.extend(dict(data) for data in edge_data.values())
    return records


def _local_branch_point(graph, node, neighbor, records) -> tuple[float, float]:
    node_x = float(graph.nodes[node]["x"])
    node_y = float(graph.nodes[node]["y"])
    with_geometry = [record for record in records if record.get("geometry") is not None]
    if with_geometry:
        record = min(with_geometry, key=lambda item: float(item.get("length", math.inf)))
        geometry = record["geometry"]
        if hasattr(geometry, "coords"):
            coordinates = list(geometry.coords)
            if len(coordinates) >= 2:
                first_distance = (coordinates[0][0] - node_x) ** 2 + (
                    coordinates[0][1] - node_y
                ) ** 2
                last_distance = (coordinates[-1][0] - node_x) ** 2 + (
                    coordinates[-1][1] - node_y
                ) ** 2
                point = coordinates[1] if first_distance <= last_distance else coordinates[-2]
                return float(point[0]), float(point[1])
    neighbor_data = graph.nodes[neighbor]
    return float(neighbor_data["x"]), float(neighbor_data["y"])


def _branch_details(graph, node, neighbor) -> dict:
    records = _edge_records(graph, node, neighbor)
    point_x, point_y = _local_branch_point(graph, node, neighbor, records)
    node_data = graph.nodes[node]
    names: set[str] = set()
    highways: set[str] = set()
    for record in records:
        names.update(_values(record.get("name")))
        highways.update(_values(record.get("highway")))
    return {
        "neighbor": str(neighbor),
        "bearing": _bearing(
            float(node_data["y"]), float(node_data["x"]), point_y, point_x
        ),
        "names": "; ".join(sorted(names)),
        "highways": "; ".join(sorted(highways)),
    }


def extract_four_way_intersections(graph) -> pd.DataFrame:
    """Return every node with four distinct road branches and geometry checks.

    ``likely_geometric_cross`` rejects extremely acute branch arrangements but
    remains a model flag, not a claim that aerial imagery has been inspected.
    """
    rows = []
    for node, node_data in graph.nodes(data=True):
        if node_data.get("x") is None or node_data.get("y") is None:
            continue
        neighbors = set(graph.predecessors(node)) | set(graph.successors(node))
        if len(neighbors) != 4:
            continue
        branches = sorted(
            (_branch_details(graph, node, neighbor) for neighbor in neighbors),
            key=lambda branch: branch["bearing"],
        )
        bearings = [branch["bearing"] for branch in branches]
        gaps = [
            (bearings[(index + 1) % 4] - bearings[index]) % 360
            for index in range(4)
        ]
        cross_score = max(0.0, 1.0 - sum(abs(gap - 90) for gap in gaps) / 360)
        names = sorted(
            {
                name
                for branch in branches
                for name in branch["names"].split("; ")
                if name
            }
        )
        highways = sorted(
            {
                highway
                for branch in branches
                for highway in branch["highways"].split("; ")
                if highway
            }
        )
        highway_tag = _values(node_data.get("highway"))
        row = {
            "osm_node_id": str(node),
            "latitude": float(node_data["y"]),
            "longitude": float(node_data["x"]),
            "street_names": "; ".join(names),
            "highway_types": "; ".join(highways),
            "in_degree": graph.in_degree(node),
            "out_degree": graph.out_degree(node),
            "fully_bidirectional": graph.in_degree(node) == 4
            and graph.out_degree(node) == 4,
            "traffic_signals": "traffic_signals" in highway_tag,
            "junction": "; ".join(sorted(_values(node_data.get("junction")))),
            "minimum_branch_angle": round(min(gaps), 2),
            "maximum_branch_angle": round(max(gaps), 2),
            "cross_score": round(cross_score, 4),
            "likely_geometric_cross": min(gaps) >= 40 and max(gaps) <= 140,
            "osm_url": f"https://www.openstreetmap.org/node/{node}",
        }
        for index, branch in enumerate(branches, start=1):
            row[f"branch_{index}_neighbor"] = branch["neighbor"]
            row[f"branch_{index}_bearing"] = round(branch["bearing"], 2)
            row[f"branch_{index}_street"] = branch["names"]
            row[f"branch_{index}_highway"] = branch["highways"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["likely_geometric_cross", "cross_score", "osm_node_id"],
        ascending=[False, False, True],
    )


def export_four_way_intersections(graph, output_prefix: str | Path) -> dict:
    """Export the candidate catalog to analysis-friendly CSV and GeoJSON."""
    prefix = Path(output_prefix)
    if prefix.suffix.lower() in {".csv", ".geojson"}:
        prefix = prefix.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    geojson_path = prefix.with_suffix(".geojson")

    table = extract_four_way_intersections(graph)
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    geodata = gpd.GeoDataFrame(
        table.copy(),
        geometry=[Point(xy) for xy in zip(table["longitude"], table["latitude"])],
        crs="EPSG:4326",
    )
    geodata.to_file(geojson_path, driver="GeoJSON")
    return {"table": table, "csv": csv_path, "geojson": geojson_path}
