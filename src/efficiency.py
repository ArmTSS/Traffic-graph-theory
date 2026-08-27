"""Weighted graph-efficiency metrics used by the research comparisons."""

import math
import networkx as nx


def weighted_efficiency(
    graph: nx.DiGraph,
    demand: dict[tuple[object, object], float] | None = None,
    weight: str = "travel_time",
) -> float:
    """Return mean reciprocal shortest-path travel time.

    With demand, the result is demand-weighted.  Units are reciprocal units of
    ``weight`` (normally 1/second); unreachable pairs contribute zero.
    """
    if graph.number_of_nodes() < 2:
        return 0.0
    pairs = demand or {
        (source, target): 1.0
        for source in graph
        for target in graph
        if source != target
    }
    total_weight = 0.0
    total_efficiency = 0.0
    for (source, target), pair_demand in pairs.items():
        if pair_demand < 0:
            raise ValueError("demand values must be non-negative")
        total_weight += pair_demand
        try:
            travel_time = nx.shortest_path_length(graph, source, target, weight=weight)
        except nx.NetworkXNoPath, nx.NodeNotFound:
            continue
        if travel_time > 0 and math.isfinite(travel_time):
            total_efficiency += pair_demand / travel_time
    return total_efficiency / total_weight if total_weight else 0.0
