"""Convert an OSMnx street graph into the project's traffic simulation model."""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from typing import Any

import networkx as nx

import config as cfg
from controller import FreeFlowController
from graph_model import IntersectionNetwork
from metrics import aggregate_runs, compute_metrics
from simulation import TrafficSimulation


DEFAULT_SPEED_KPH = {
    "motorway": 90.0,
    "motorway_link": 50.0,
    "trunk": 80.0,
    "trunk_link": 50.0,
    "primary": 60.0,
    "primary_link": 40.0,
    "secondary": 50.0,
    "secondary_link": 40.0,
    "tertiary": 40.0,
    "tertiary_link": 30.0,
    "residential": 30.0,
    "living_street": 20.0,
    "service": 20.0,
    "unclassified": 30.0,
}


@dataclass(frozen=True)
class RoadEstimate:
    length_m: float
    highway: str
    lanes: int
    speed_kph: float
    travel_time_s: int
    capacity: int
    oneway: bool


def _first(value: Any, fallback: Any = None) -> Any:
    if value is None or value == "":
        return fallback
    if isinstance(value, (list, tuple)):
        return value[0] if value else fallback
    return value


def _number(value: Any, fallback: float) -> float:
    value = _first(value, fallback)
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else fallback


def parse_maxspeed_kph(value: Any, highway: str) -> float:
    """Parse common OSM maxspeed values, falling back by road class."""
    fallback = DEFAULT_SPEED_KPH.get(highway, 30.0)
    raw = _first(value)
    if raw is None:
        return fallback
    speed = _number(raw, fallback)
    if "mph" in str(raw).lower():
        speed *= 1.609344
    return max(speed, 5.0)


def estimate_road(edge: dict[str, Any]) -> RoadEstimate:
    """Estimate simulation storage capacity and free-flow time from OSM tags."""
    highway = str(_first(edge.get("highway"), "unclassified"))
    length_m = max(_number(edge.get("length"), 1.0), 1.0)
    tagged_lanes = max(1, round(_number(edge.get("lanes"), cfg.OSM_DEFAULT_LANES)))
    raw_oneway = _first(edge.get("oneway"), False)
    oneway = raw_oneway is True or str(raw_oneway).lower() in {"yes", "true", "1", "-1"}

    # A lanes tag on a two-way street commonly describes both directions.
    lanes = tagged_lanes if oneway else max(1, math.ceil(tagged_lanes / 2))
    speed_kph = parse_maxspeed_kph(edge.get("maxspeed"), highway)
    travel_time_s = max(1, math.ceil(length_m / (speed_kph / 3.6)))

    # Road.capacity is simultaneous occupancy, not hourly flow capacity.
    capacity = max(1, math.floor(length_m * lanes / cfg.OSM_VEHICLE_SPACING_M))
    return RoadEstimate(
        length_m=length_m,
        highway=highway,
        lanes=lanes,
        speed_kph=speed_kph,
        travel_time_s=travel_time_s,
        capacity=capacity,
        oneway=oneway,
    )


def _largest_strong_component(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    if not graph.is_directed():
        graph = graph.to_directed()
    components = list(nx.strongly_connected_components(graph))
    if not components:
        raise ValueError("OSM graph contains no connected road nodes")
    return graph.subgraph(max(components, key=len)).copy()


def _node_label(node: Any) -> str:
    return f"osm:{node}"


def _select_portals(graph: nx.MultiDiGraph, count: int) -> list[Any]:
    """Choose geographically separated nodes around the network boundary."""
    if count < 2:
        raise ValueError("portal_count must be at least 2")
    candidates = [
        (node, float(data["x"]), float(data["y"]))
        for node, data in graph.nodes(data=True)
        if data.get("x") is not None and data.get("y") is not None
    ]
    if len(candidates) < count:
        raise ValueError("not enough geocoded nodes for the requested portals")

    mean_x = sum(item[1] for item in candidates) / len(candidates)
    mean_y = sum(item[2] for item in candidates) / len(candidates)
    x_scale = math.cos(math.radians(mean_y))
    selected: list[Any] = []
    for index in range(count):
        angle = 2 * math.pi * index / count
        dx, dy = math.cos(angle), math.sin(angle)
        ranked = sorted(
            candidates,
            key=lambda item: ((item[1] - mean_x) * x_scale * dx + (item[2] - mean_y) * dy),
            reverse=True,
        )
        selected.append(next(node for node, _, _ in ranked if node not in selected))
    return selected


def osm_graph_to_intersection_network(
    graph: nx.MultiDiGraph,
    *,
    name: str = "OSM road network",
    portal_count: int = cfg.OSM_PORTAL_COUNT,
) -> IntersectionNetwork:
    """Convert the largest routable OSM component to Road objects and a DiGraph."""
    component = _largest_strong_component(graph)
    network = IntersectionNetwork(name)

    # IntersectionNetwork is a DiGraph, so retain the fastest parallel OSM edge.
    best_edges: dict[tuple[Any, Any], tuple[RoadEstimate, dict[str, Any]]] = {}
    for start, end, data in component.edges(data=True):
        estimate = estimate_road(data)
        key = (start, end)
        previous = best_edges.get(key)
        score = (estimate.travel_time_s, -estimate.capacity)
        if previous is None or score < (
            previous[0].travel_time_s,
            -previous[0].capacity,
        ):
            best_edges[key] = (estimate, dict(data))

    for (start, end), (estimate, raw) in best_edges.items():
        start_label, end_label = _node_label(start), _node_label(end)
        network.add_road(
            start_label,
            end_label,
            capacity=estimate.capacity,
            travel_time=estimate.travel_time_s,
            length=estimate.length_m,
        )
        network.nx_graph.nodes[start_label].update(component.nodes[start])
        network.nx_graph.nodes[end_label].update(component.nodes[end])
        network.nx_graph.edges[start_label, end_label].update(
            osm_start=start,
            osm_end=end,
            highway=estimate.highway,
            lanes=estimate.lanes,
            maxspeed_kph=round(estimate.speed_kph, 2),
            length_m=round(estimate.length_m, 2),
            oneway=estimate.oneway,
            osm_id=raw.get("osmid"),
        )

    portals = [_node_label(node) for node in _select_portals(component, portal_count)]
    network.set_entry_nodes(portals)
    network.set_exit_nodes(portals)
    return network


def generate_synthetic_od_demand(
    network: IntersectionNetwork,
    total_rate: float = cfg.OSM_TOTAL_DEMAND_PER_SECOND,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Create uniform portal-to-portal demand with no same-portal trips."""
    if total_rate <= 0:
        raise ValueError("total_rate must be positive")
    portals = network.entry_nodes
    if len(portals) < 2:
        raise ValueError("at least two entry portals are required")
    rate_per_portal = total_rate / len(portals)
    demand = {portal: rate_per_portal for portal in portals}
    destination_probs = {
        origin: {
            destination: 1.0 / (len(portals) - 1)
            for destination in portals
            if destination != origin
        }
        for origin in portals
    }
    return demand, destination_probs


def _graph_metrics(network: IntersectionNetwork, demand, destination_probs) -> dict:
    od_demand = {
        (origin, destination): demand[origin] * probability
        for origin, probabilities in destination_probs.items()
        for destination, probability in probabilities.items()
    }
    node_count = network.num_nodes()
    sample_count = min(100, node_count)
    centrality = nx.betweenness_centrality(
        network.nx_graph,
        k=sample_count if sample_count < node_count else None,
        weight="weight",
        seed=cfg.RANDOM_SEED,
    )
    top_nodes = sorted(centrality.items(), key=lambda item: item[1], reverse=True)[:5]
    return {
        "nodes": node_count,
        "edges": network.num_edges(),
        "portals": len(network.entry_nodes),
        "avg_portal_path_time_s": network.average_path_length(),
        "demand_weighted_graph_efficiency": network.weighted_efficiency(od_demand),
        "top_betweenness_nodes": top_nodes,
    }


def run_osm_simulation_study(
    graph: nx.MultiDiGraph,
    *,
    name: str,
    sim_time: int,
    n_runs: int,
    total_demand_rate: float = cfg.OSM_TOTAL_DEMAND_PER_SECOND,
    portal_count: int = cfg.OSM_PORTAL_COUNT,
    engine: str = "step",
    seed: int = cfg.RANDOM_SEED,
) -> dict:
    """Build, simulate, and summarize one real-map road network."""
    base_network = osm_graph_to_intersection_network(
        graph, name=name, portal_count=portal_count
    )
    demand, destination_probs = generate_synthetic_od_demand(
        base_network, total_rate=total_demand_rate
    )
    simulator_class = TrafficSimulation
    if engine == "simpy":
        from simpy_simulation import SimPyTrafficSimulation

        simulator_class = SimPyTrafficSimulation
    elif engine != "step":
        raise ValueError("engine must be 'step' or 'simpy'")

    raw_results = []
    derived_results = []
    for index in range(n_runs):
        network = copy.deepcopy(base_network)
        simulator = simulator_class(
            network,
            FreeFlowController(),
            destination_probs,
            demand,
            "osm",
            name,
        )
        result = simulator.run(sim_time=sim_time, seed=seed + index)
        raw_results.append(result)
        derived_results.append(compute_metrics(result, demand))

    mean_saturation = {
        edge: sum(result.edge_saturation[edge] for result in raw_results) / n_runs
        for edge in base_network.roads
    }
    bottlenecks = sorted(
        mean_saturation.items(), key=lambda item: item[1], reverse=True
    )[:5]
    return {
        "network": base_network,
        "demand": demand,
        "destination_probs": destination_probs,
        "traffic_metrics": aggregate_runs(derived_results),
        "graph_metrics": _graph_metrics(base_network, demand, destination_probs),
        "bottlenecks": bottlenecks,
        "raw_results": raw_results,
    }
