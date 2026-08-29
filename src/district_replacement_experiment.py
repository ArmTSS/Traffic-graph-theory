"""Coupled replacement experiments inside one shared OSM district graph."""

from __future__ import annotations

import copy
import statistics
from pathlib import Path

import networkx as nx
import pandas as pd

import config as cfg
from controller import CompositeController, FixedTimeSignalController, YieldController
from osm_simulation import generate_synthetic_od_demand, osm_graph_to_intersection_network
from replacement_experiment import (
    DESIGN_NAMES,
    _site_arms,
    export_replacement_traffic_map,
    plot_replacement_traffic_map,
    select_important_four_way_intersections,
)
from simulation import TrafficSimulation


def _osm_node(node) -> str:
    return f"osm:{node}"


def _remove_road(network, edge):
    network.roads.pop(edge)
    network.controlled_edges.discard(edge)
    network.nx_graph.remove_edge(*edge)


def _copy_road(network, road, start, end, *, controlled=False):
    return network.add_road(
        start,
        end,
        capacity=road.capacity,
        travel_time=road.travel_time,
        controlled=controlled,
        length=road.length,
    )


def _signal_for_edges(incoming_edges):
    controller = FixedTimeSignalController(
        [
            [incoming_edges["A"], incoming_edges["C"]],
            [incoming_edges["B"], incoming_edges["D"]],
        ],
        [cfg.GREEN_LIGHT_TIME, cfg.GREEN_LIGHT_TIME],
        phase_names=["AC_GREEN", "BD_GREEN"],
    )
    return controller


def build_coupled_district_network(graph, sites, design: str, *, portal_count: int = 8):
    """Embed one design at every selected site in a shared district network."""
    if design not in DESIGN_NAMES:
        raise ValueError(f"unknown replacement design: {design}")
    network = osm_graph_to_intersection_network(
        graph,
        name=f"Khlong Sam Wa coupled {design}",
        portal_count=portal_count,
    )
    edge_controllers = {}
    observed_edges = {}
    district_portals = list(network.entry_nodes)
    site_zone_portals = []

    for _, site in sites.iterrows():
        raw_node = int(site["osm_node_id"])
        center = _osm_node(raw_node)
        if center in network.entry_nodes:
            raise ValueError(f"selected replacement node {raw_node} is a demand portal")
        arms = _site_arms(graph, raw_node)
        incoming = {
            arm["label"]: (_osm_node(arm["neighbor"]), center) for arm in arms
        }
        outgoing = {
            arm["label"]: (center, _osm_node(arm["neighbor"])) for arm in arms
        }
        site_zone_portals.extend(edge[0] for edge in incoming.values())
        if not all(edge in network.roads for edge in [*incoming.values(), *outgoing.values()]):
            raise ValueError(f"selected site {raw_node} is outside the routable component")

        if design != "roundabout":
            controller = _signal_for_edges(incoming)
            for edge in incoming.values():
                network.roads[edge].controlled = True
                network.controlled_edges.add(edge)
                network.nx_graph.edges[edge]["controlled"] = True
                edge_controllers[edge] = controller
            site_observed = list(incoming.values())

            if design in {"flyover", "underpass"}:
                axis_ac = (
                    network.roads[incoming["A"]].capacity
                    + network.roads[incoming["C"]].capacity
                )
                axis_bd = (
                    network.roads[incoming["B"]].capacity
                    + network.roads[incoming["D"]].capacity
                )
                major = ("A", "C") if axis_ac >= axis_bd else ("B", "D")
                opposite = {"A": "C", "C": "A", "B": "D", "D": "B"}
                for label in major:
                    target = opposite[label]
                    source_node = incoming[label][0]
                    target_node = outgoing[target][1]
                    grade_node = f"site:{raw_node}:{design}:{label}"
                    source_road = network.roads[incoming[label]]
                    target_road = network.roads[outgoing[target]]
                    capacity = min(source_road.capacity, target_road.capacity)
                    travel_time = max(
                        1, source_road.travel_time + target_road.travel_time - 2
                    )
                    bypass_edge = (source_node, grade_node)
                    network.add_road(
                        *bypass_edge,
                        capacity=capacity,
                        travel_time=travel_time,
                        length=source_road.length + target_road.length,
                    )
                    network.add_road(
                        grade_node,
                        target_node,
                        capacity=capacity,
                        travel_time=1,
                        length=1.0,
                    )
                    site_observed.append(bypass_edge)
            observed_edges[str(raw_node)] = site_observed
            continue

        original_incoming = {
            label: network.roads[edge] for label, edge in incoming.items()
        }
        original_outgoing = {
            label: network.roads[edge] for label, edge in outgoing.items()
        }
        for edge in [*incoming.values(), *outgoing.values()]:
            _remove_road(network, edge)
        network.nx_graph.remove_node(center)

        labels = [arm["label"] for arm in arms]
        queues = {label: f"site:{raw_node}:Q:{label}" for label in labels}
        rings = {label: f"site:{raw_node}:R:{label}" for label in labels}
        merge_edges = []
        conflicts, downstream = {}, {}
        for label in labels:
            _copy_road(
                network,
                original_incoming[label],
                incoming[label][0],
                queues[label],
            )
            merge_edge = (queues[label], rings[label])
            network.add_road(
                *merge_edge,
                capacity=cfg.DEFAULT_LINK_CAPACITY,
                travel_time=1,
                controlled=True,
                length=5.0,
            )
            _copy_road(
                network,
                original_outgoing[label],
                rings[label],
                outgoing[label][1],
            )
            merge_edges.append(merge_edge)
        for index, label in enumerate(labels):
            previous = labels[(index - 1) % len(labels)]
            following = labels[(index + 1) % len(labels)]
            network.add_road(
                rings[label],
                rings[following],
                capacity=cfg.DEFAULT_LINK_CAPACITY,
                travel_time=cfg.DEFAULT_LINK_TRAVEL_TIME,
                length=15.0,
            )
            merge_edge = (queues[label], rings[label])
            conflicts[merge_edge] = (rings[previous], rings[label])
            downstream[merge_edge] = (rings[label], rings[following])
        controller = YieldController(
            merge_edges,
            cfg.ROUNDABOUT_MAX_MERGE_PER_STEP,
            roads=network.roads,
            conflict_edges=conflicts,
            downstream_edges=downstream,
        )
        for edge in merge_edges:
            edge_controllers[edge] = controller
        observed_edges[str(raw_node)] = merge_edges

    demand_portals = list(dict.fromkeys(district_portals + site_zone_portals))
    network.set_entry_nodes(demand_portals)
    network.set_exit_nodes(demand_portals)
    return network, CompositeController(edge_controllers), observed_edges


def _mean(values):
    return statistics.mean(values) if values else 0.0


def run_coupled_district_sweep(
    graph,
    *,
    site_count: int,
    demand_levels: dict[str, float],
    sim_time: int = 600,
    n_runs: int = 1,
    portal_count: int = 8,
    seed: int = cfg.RANDOM_SEED,
):
    """Run designs as district-wide scenarios so selected sites interact."""
    directed_graph = graph if graph.is_directed() else graph.to_directed()
    component_nodes = max(nx.strongly_connected_components(directed_graph), key=len)
    routable_graph = directed_graph.subgraph(component_nodes).copy()
    sites = select_important_four_way_intersections(
        routable_graph, count=site_count, exclude_adjacent=True
    )
    rows = []
    for design, design_name in DESIGN_NAMES.items():
        base = build_coupled_district_network(
            routable_graph, sites, design, portal_count=portal_count
        )
        for demand_level, demand_rate in demand_levels.items():
            run_values = {str(int(site["osm_node_id"])): [] for _, site in sites.iterrows()}
            for run_index in range(n_runs):
                network, controller, observed = copy.deepcopy(base)
                demand, probabilities = generate_synthetic_od_demand(
                    network, total_rate=demand_rate
                )
                result = TrafficSimulation(
                    network,
                    controller,
                    probabilities,
                    demand,
                    f"coupled_{design}",
                    f"Coupled {design_name}",
                ).run(sim_time=sim_time, seed=seed + run_index)
                district_completion = (
                    result.vehicles_completed / result.vehicles_generated
                    if result.vehicles_generated
                    else 0.0
                )
                for node_id, edges in observed.items():
                    waiting = sum(
                        result.edge_waiting_vehicle_seconds.get(edge, 0)
                        for edge in edges
                    )
                    served = sum(result.edge_total_entries.get(edge, 0) for edge in edges)
                    avg_queue = waiting / sim_time
                    avg_wait = waiting / served if served else 0.0
                    wait_score = 1 / (1 + avg_wait / cfg.EFFICIENCY_REFERENCE_WAIT)
                    queue_score = 1 / (
                        1 + avg_queue / cfg.EFFICIENCY_REFERENCE_QUEUE
                    )
                    run_values[node_id].append(
                        {
                            "completion": district_completion,
                            "waiting": avg_wait,
                            "queue": avg_queue,
                            "throughput": served / sim_time,
                            "efficiency": 0.6 * wait_score + 0.4 * queue_score,
                        }
                    )
            for _, site in sites.iterrows():
                node_id = str(int(site["osm_node_id"]))
                values = run_values[node_id]
                rows.append(
                    {
                        "importance_rank": int(site["importance_rank"]),
                        "osm_node_id": node_id,
                        "latitude": float(site["latitude"]),
                        "longitude": float(site["longitude"]),
                        "street_names": site["street_names"],
                        "demand_level": demand_level,
                        "total_demand_rate": demand_rate,
                        "design": design,
                        "design_name": design_name,
                        "completion_rate": _mean([value["completion"] for value in values]),
                        "avg_waiting_time_s": _mean([value["waiting"] for value in values]),
                        "avg_queue_length": _mean([value["queue"] for value in values]),
                        "throughput_veh_s": _mean([value["throughput"] for value in values]),
                        "traffic_efficiency": _mean([value["efficiency"] for value in values]),
                    }
                )
    return sites, pd.DataFrame(rows)


def export_coupled_district_sweep(
    graph,
    output_dir: str | Path,
    **kwargs,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sites, results = run_coupled_district_sweep(graph, **kwargs)
    results_path = output_dir / "coupled_district_results.csv"
    picture_path = output_dir / "coupled_traffic_map.png"
    interactive_path = output_dir / "coupled_traffic_map.html"
    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    plot_replacement_traffic_map(graph, results, picture_path)
    export_replacement_traffic_map(results, interactive_path)
    return {
        "sites": sites,
        "results": results,
        "results_csv": results_path,
        "picture": picture_path,
        "interactive": interactive_path,
    }
