"""Controlled local replacement experiments for important real intersections."""

from __future__ import annotations

import itertools
import json
import math
from html import escape
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import Point

import config as cfg
from controller import FixedTimeSignalController, YieldController
from graph_model import IntersectionNetwork
from intersection_catalog import extract_four_way_intersections
from metrics import aggregate_runs, compute_metrics
from osm_simulation import estimate_road
from simulation import TrafficSimulation


DESIGN_NAMES = {
    "four_way": "Existing Signalized Four-Way",
    "roundabout": "Roundabout",
    "flyover": "Flyover",
    "underpass": "Underpass",
}

ARTERIAL_CLASSES = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
}

COMPASS_BEARINGS = {
    "north": 0.0,
    "east": 90.0,
    "south": 180.0,
    "west": 270.0,
}

PLOT_STYLES = {
    "four_way": {"color": "#374151", "marker": "o", "linestyle": "-"},
    "roundabout": {"color": "#15803d", "marker": "s", "linestyle": "-"},
    "flyover": {"color": "#ea580c", "marker": "^", "linestyle": "-"},
    "underpass": {"color": "#2563eb", "marker": "v", "linestyle": "--"},
}


def _haversine_m(lat_1, lon_1, lat_2, lon_2) -> float:
    radius = 6_371_000.0
    phi_1, phi_2 = math.radians(lat_1), math.radians(lat_2)
    delta_phi = math.radians(lat_2 - lat_1)
    delta_lambda = math.radians(lon_2 - lon_1)
    value = math.sin(delta_phi / 2) ** 2 + (
        math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(value))


def _angular_difference(first: float, second: float) -> float:
    return abs((first - second + 180) % 360 - 180)


def _compass_branches(site) -> dict[str, dict]:
    branches = [
        {
            "neighbor": str(site[f"branch_{index}_neighbor"]),
            "bearing": float(site[f"branch_{index}_bearing"]),
            "street": site[f"branch_{index}_street"],
            "highway": site[f"branch_{index}_highway"],
        }
        for index in range(1, 5)
    ]
    directions = list(COMPASS_BEARINGS)
    assignment = min(
        itertools.permutations(branches),
        key=lambda permutation: sum(
            _angular_difference(branch["bearing"], COMPASS_BEARINGS[direction])
            for direction, branch in zip(directions, permutation)
        ),
    )
    return dict(zip(directions, assignment))


def _optional_text(value):
    return None if pd.isna(value) or value == "" else str(value)


def build_selected_intersections_json(graph, sites: pd.DataFrame) -> dict:
    """Build a readable compass-oriented representation of selected sites."""
    intersections = []
    for _, site in sites.iterrows():
        node_text = str(site["osm_node_id"])
        node = int(node_text) if int(node_text) in graph else node_text
        roads = {}
        for direction, branch in _compass_branches(site).items():
            neighbor_text = branch["neighbor"]
            neighbor = int(neighbor_text) if int(neighbor_text) in graph else neighbor_text
            roads[direction] = {
                "connected_osm_node": neighbor_text,
                "street_name": _optional_text(branch["street"]),
                "highway_type": _optional_text(branch["highway"]),
                "bearing_degrees": round(branch["bearing"], 2),
                "bearing_offset_from_compass_degrees": round(
                    _angular_difference(
                        branch["bearing"], COMPASS_BEARINGS[direction]
                    ),
                    2,
                ),
                "travel_toward_intersection": graph.has_edge(neighbor, node),
                "travel_away_from_intersection": graph.has_edge(node, neighbor),
            }
        intersections.append(
            {
                "importance_rank": int(site["importance_rank"]),
                "osm_node_id": node_text,
                "osm_url": str(site["osm_url"]),
                "coordinates": {
                    "latitude": float(site["latitude"]),
                    "longitude": float(site["longitude"]),
                },
                "street_names": _optional_text(site["street_names"]),
                "highway_types": _optional_text(site["highway_types"]),
                "betweenness_centrality": float(site["betweenness_centrality"]),
                "cross_score": float(site["cross_score"]),
                "osm_traffic_signal_tag": bool(site["traffic_signals"]),
                "baseline_control_assumption": str(site["baseline_control"]),
                "opposite_road_pairs": [
                    ["north", "south"],
                    ["east", "west"],
                ],
                "roads": roads,
            }
        )
    return {
        "district": "Khlong Sam Wa, Bangkok, Thailand",
        "coordinate_reference_system": "EPSG:4326",
        "selection": {
            "count": len(intersections),
            "method": (
                "Likely geometric, fully bidirectional arterial four-way nodes; "
                "ranked by approximate length-weighted betweenness with spatial separation."
            ),
            "compass_assignment": (
                "One-to-one assignment minimizing angular difference from north, "
                "east, south, and west. Directions are approximate."
            ),
        },
        "intersections": intersections,
    }


def select_important_four_way_intersections(
    graph,
    count: int = 10,
    *,
    centrality_samples: int = 200,
    minimum_spacing_m: float = 500.0,
    exclude_adjacent: bool = False,
) -> pd.DataFrame:
    """Select spatially separated arterial four-ways by graph betweenness."""
    if count <= 0:
        raise ValueError("count must be positive")
    catalog = extract_four_way_intersections(graph)
    eligible = catalog[
        catalog["likely_geometric_cross"]
        & catalog["fully_bidirectional"]
        & catalog["highway_types"].fillna("").apply(
            lambda value: bool(set(value.split("; ")) & ARTERIAL_CLASSES)
        )
    ].copy()
    if len(eligible) < count:
        raise ValueError(
            f"only {len(eligible)} eligible arterial four-way intersections found"
        )

    sample_count = min(centrality_samples, graph.number_of_nodes())
    centrality = nx.betweenness_centrality(
        graph,
        k=sample_count if sample_count < graph.number_of_nodes() else None,
        normalized=True,
        weight="length",
        seed=cfg.RANDOM_SEED,
    )
    eligible["betweenness_centrality"] = eligible["osm_node_id"].map(
        lambda node: centrality.get(int(node), centrality.get(node, 0.0))
    )
    eligible = eligible.sort_values(
        ["betweenness_centrality", "traffic_signals", "cross_score"],
        ascending=[False, False, False],
    )

    selected_rows = []
    for _, row in eligible.iterrows():
        row_node = int(row["osm_node_id"])
        separated = all(
            _haversine_m(
                row["latitude"],
                row["longitude"],
                chosen["latitude"],
                chosen["longitude"],
            )
            >= minimum_spacing_m
            for chosen in selected_rows
        )
        nonadjacent = not exclude_adjacent or all(
            not graph.has_edge(row_node, int(chosen["osm_node_id"]))
            and not graph.has_edge(int(chosen["osm_node_id"]), row_node)
            for chosen in selected_rows
        )
        if separated and nonadjacent:
            selected_rows.append(row)
        if len(selected_rows) == count:
            break
    if len(selected_rows) < count:
        selected_ids = {row["osm_node_id"] for row in selected_rows}
        for _, row in eligible.iterrows():
            row_node = int(row["osm_node_id"])
            nonadjacent = not exclude_adjacent or all(
                not graph.has_edge(row_node, int(chosen["osm_node_id"]))
                and not graph.has_edge(int(chosen["osm_node_id"]), row_node)
                for chosen in selected_rows
            )
            if row["osm_node_id"] not in selected_ids and nonadjacent:
                selected_rows.append(row)
                selected_ids.add(row["osm_node_id"])
            if len(selected_rows) == count:
                break
    if len(selected_rows) < count:
        raise ValueError(
            f"only {len(selected_rows)} nonadjacent eligible intersections available"
        )

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected.insert(0, "importance_rank", range(1, len(selected) + 1))
    selected["baseline_control"] = "assumed fixed-time signal"
    return selected


def _best_directional_edge(graph, start, end) -> dict:
    records = list(graph.get_edge_data(start, end, default={}).values())
    if not records:
        raise ValueError(f"missing directed OSM branch {start!r}->{end!r}")
    return min(
        records,
        key=lambda data: (
            estimate_road(data).travel_time_s,
            -estimate_road(data).capacity,
        ),
    )


def _bearing(lat_1, lon_1, lat_2, lon_2) -> float:
    lat_1, lat_2 = math.radians(lat_1), math.radians(lat_2)
    delta_lon = math.radians(lon_2 - lon_1)
    y = math.sin(delta_lon) * math.cos(lat_2)
    x = math.cos(lat_1) * math.sin(lat_2) - (
        math.sin(lat_1) * math.cos(lat_2) * math.cos(delta_lon)
    )
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def _site_arms(graph, node) -> list[dict]:
    node_data = graph.nodes[node]
    neighbors = set(graph.predecessors(node)) | set(graph.successors(node))
    if len(neighbors) != 4:
        raise ValueError(f"OSM node {node} does not have four distinct branches")
    arms = []
    for neighbor in neighbors:
        inbound = estimate_road(_best_directional_edge(graph, neighbor, node))
        outbound = estimate_road(_best_directional_edge(graph, node, neighbor))
        neighbor_data = graph.nodes[neighbor]
        arms.append(
            {
                "neighbor": neighbor,
                "bearing": _bearing(
                    float(node_data["y"]),
                    float(node_data["x"]),
                    float(neighbor_data["y"]),
                    float(neighbor_data["x"]),
                ),
                "inbound": inbound,
                "outbound": outbound,
            }
        )
    arms.sort(key=lambda arm: arm["bearing"])
    for label, arm in zip(("A", "B", "C", "D"), arms):
        arm["label"] = label
    return arms


def _add_real_approaches(network, arms, center_nodes, *, controlled=True):
    for arm in arms:
        label = arm["label"]
        inbound, outbound = arm["inbound"], arm["outbound"]
        center = center_nodes[label]
        network.add_road(
            f"{label}_in",
            center,
            inbound.capacity,
            inbound.travel_time_s,
            controlled=controlled,
            length=inbound.length_m,
        )
        network.add_road(
            center,
            f"{label}_out",
            outbound.capacity,
            outbound.travel_time_s,
            length=outbound.length_m,
        )


def _signal_controller(center_nodes) -> FixedTimeSignalController:
    phase_groups = [
        [("A_in", center_nodes["A"]), ("C_in", center_nodes["C"])],
        [("B_in", center_nodes["B"]), ("D_in", center_nodes["D"])],
    ]
    return FixedTimeSignalController(
        phase_groups,
        [cfg.GREEN_LIGHT_TIME, cfg.GREEN_LIGHT_TIME],
        phase_names=["AC_GREEN", "BD_GREEN"],
    )


def _common_demand(network, total_demand_rate):
    rate = total_demand_rate / len(network.entry_nodes)
    demand = {entry: rate for entry in network.entry_nodes}
    probabilities = {
        entry: {
            destination: 1 / (len(network.exit_nodes) - 1)
            for destination in network.exit_nodes
            if destination.split("_")[0] != entry.split("_")[0]
        }
        for entry in network.entry_nodes
    }
    return demand, probabilities


def build_local_replacement_variants(
    graph, node, *, total_demand_rate: float = 1.0
) -> dict:
    """Build four alternatives while preserving one site's real OSM approaches."""
    arms = _site_arms(graph, node)
    entries = [f"{arm['label']}_in" for arm in arms]
    exits = [f"{arm['label']}_out" for arm in arms]
    variants = {}

    baseline = IntersectionNetwork(f"OSM {node}: signalized four-way")
    baseline.set_entry_nodes(entries)
    baseline.set_exit_nodes(exits)
    baseline_centers = {arm["label"]: "I" for arm in arms}
    _add_real_approaches(baseline, arms, baseline_centers)
    demand, probabilities = _common_demand(baseline, total_demand_rate)
    variants["four_way"] = (
        baseline,
        _signal_controller(baseline_centers),
        probabilities,
        demand,
    )

    roundabout = IntersectionNetwork(f"OSM {node}: roundabout")
    roundabout.set_entry_nodes(entries)
    roundabout.set_exit_nodes(exits)
    ring_nodes = {arm["label"]: f"R_{arm['label']}" for arm in arms}
    queue_nodes = {arm["label"]: f"Q_{arm['label']}" for arm in arms}
    for arm in arms:
        label = arm["label"]
        inbound, outbound = arm["inbound"], arm["outbound"]
        roundabout.add_road(
            f"{label}_in",
            queue_nodes[label],
            inbound.capacity,
            inbound.travel_time_s,
            length=inbound.length_m,
        )
        roundabout.add_road(
            queue_nodes[label],
            ring_nodes[label],
            cfg.DEFAULT_LINK_CAPACITY,
            1,
            controlled=True,
            length=5.0,
        )
        roundabout.add_road(
            ring_nodes[label],
            f"{label}_out",
            outbound.capacity,
            outbound.travel_time_s,
            length=outbound.length_m,
        )
    for current, following in zip(arms, arms[1:] + arms[:1]):
        roundabout.add_road(
            ring_nodes[current["label"]],
            ring_nodes[following["label"]],
            cfg.ROUNDABOUT_RING_CAPACITY,
            cfg.ROUNDABOUT_RING_TRAVEL_TIME,
            length=15.0,
        )
    entry_edges = [
        (queue_nodes[entry.split("_")[0]], ring_nodes[entry.split("_")[0]])
        for entry in entries
    ]
    conflict_edges = {}
    downstream_edges = {}
    labels = [arm["label"] for arm in arms]
    for index, label in enumerate(labels):
        previous = labels[(index - 1) % len(labels)]
        following = labels[(index + 1) % len(labels)]
        merge_edge = (queue_nodes[label], ring_nodes[label])
        conflict_edges[merge_edge] = (ring_nodes[previous], ring_nodes[label])
        downstream_edges[merge_edge] = (ring_nodes[label], ring_nodes[following])
    variants["roundabout"] = (
        roundabout,
        YieldController(
            entry_edges,
            cfg.ROUNDABOUT_MAX_MERGE_PER_STEP,
            roads=roundabout.roads,
            conflict_edges=conflict_edges,
            downstream_edges=downstream_edges,
            critical_occupancy=cfg.ROUNDABOUT_CRITICAL_GAP_OCCUPANCY,
        ),
        probabilities,
        demand,
    )

    # A/C is the first opposite axis. Choose the axis with more storage as the
    # grade-separated through movement so the decision is deterministic.
    axis_ac = arms[0]["inbound"].capacity + arms[2]["inbound"].capacity
    axis_bd = arms[1]["inbound"].capacity + arms[3]["inbound"].capacity
    major_labels = ("A", "C") if axis_ac >= axis_bd else ("B", "D")
    opposite = {"A": "C", "C": "A", "B": "D", "D": "B"}
    for design in ("flyover", "underpass"):
        network = IntersectionNetwork(f"OSM {node}: {design}")
        network.set_entry_nodes(entries)
        network.set_exit_nodes(exits)
        centers = {arm["label"]: "I_surface" for arm in arms}
        _add_real_approaches(network, arms, centers)
        for label in major_labels:
            source_arm = next(arm for arm in arms if arm["label"] == label)
            target_arm = next(
                arm for arm in arms if arm["label"] == opposite[label]
            )
            free_flow = (
                source_arm["inbound"].travel_time_s
                + target_arm["outbound"].travel_time_s
            )
            network.add_road(
                f"{label}_in",
                f"{opposite[label]}_out",
                min(
                    source_arm["inbound"].capacity,
                    target_arm["outbound"].capacity,
                ),
                max(1, free_flow - 1),
                length=(
                    source_arm["inbound"].length_m
                    + target_arm["outbound"].length_m
                ),
            )
        variants[design] = (
            network,
            _signal_controller(centers),
            probabilities,
            demand,
        )
    return variants


def run_replacement_experiment(
    graph,
    selected_sites: pd.DataFrame,
    *,
    sim_time: int,
    n_runs: int,
    total_demand_rate: float = 1.0,
    seed: int = cfg.RANDOM_SEED,
) -> pd.DataFrame:
    rows = []
    for _, site in selected_sites.iterrows():
        node = int(site["osm_node_id"])
        for design_key in DESIGN_NAMES:
            derived_runs = []
            network_for_graph_metrics = None
            for run_index in range(n_runs):
                variants = build_local_replacement_variants(
                    graph, node, total_demand_rate=total_demand_rate
                )
                network, controller, probabilities, demand = variants[design_key]
                network_for_graph_metrics = network
                result = TrafficSimulation(
                    network,
                    controller,
                    probabilities,
                    demand,
                    design_key,
                    DESIGN_NAMES[design_key],
                ).run(sim_time=sim_time, seed=seed + run_index)
                derived_runs.append(compute_metrics(result, demand))
            aggregate = aggregate_runs(derived_runs)
            stats = aggregate.stats
            od_demand = {
                (origin, destination): demand[origin] * probability
                for origin, destinations in probabilities.items()
                for destination, probability in destinations.items()
            }
            rows.append(
                {
                    "importance_rank": site["importance_rank"],
                    "osm_node_id": str(node),
                    "latitude": site["latitude"],
                    "longitude": site["longitude"],
                    "street_names": site["street_names"],
                    "betweenness_centrality": site["betweenness_centrality"],
                    "design": design_key,
                    "design_name": DESIGN_NAMES[design_key],
                    "completion_rate": stats["completion_rate"]["mean"],
                    "completion_rate_std": stats["completion_rate"]["std"],
                    "avg_waiting_time_s": stats["avg_waiting_time"]["mean"],
                    "avg_waiting_time_s_std": stats["avg_waiting_time"]["std"],
                    "avg_queue_length": stats["avg_queue_length"]["mean"],
                    "avg_queue_length_std": stats["avg_queue_length"]["std"],
                    "throughput_veh_s": stats["throughput_per_sec"]["mean"],
                    "throughput_veh_s_std": stats["throughput_per_sec"]["std"],
                    "traffic_efficiency": stats["efficiency"]["mean"],
                    "traffic_efficiency_std": stats["efficiency"]["std"],
                    "avg_path_time_s": network_for_graph_metrics.average_path_length(),
                    "graph_efficiency": network_for_graph_metrics.weighted_efficiency(
                        od_demand
                    ),
                }
            )
    results = pd.DataFrame(rows)
    baseline = results[results["design"] == "four_way"].set_index("osm_node_id")
    for metric in (
        "completion_rate",
        "avg_waiting_time_s",
        "avg_queue_length",
        "throughput_veh_s",
        "traffic_efficiency",
        "avg_path_time_s",
        "graph_efficiency",
    ):
        baseline_values = results["osm_node_id"].map(baseline[metric])
        denominator = baseline_values.where(baseline_values.abs() > 1e-12)
        results[f"{metric}_change_pct"] = (
            (results[metric] - baseline_values) / denominator * 100
        ).fillna(0.0)
    return results


def plot_replacement_comparison(
    results: pd.DataFrame, output_path: str | Path
) -> Path:
    """Plot traffic outcomes at each selected intersection for all designs."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("traffic_efficiency", "traffic_efficiency_std", "Traffic efficiency"),
        ("avg_waiting_time_s", "avg_waiting_time_s_std", "Average waiting (s)"),
        ("throughput_veh_s", "throughput_veh_s_std", "Throughput (vehicles/s)"),
    ]
    figure, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for axis, (metric, std_metric, label) in zip(axes, panels):
        for design_key, design_name in DESIGN_NAMES.items():
            design_rows = results[results["design"] == design_key].sort_values(
                "importance_rank"
            )
            x = design_rows["importance_rank"].to_numpy(dtype=float)
            y = design_rows[metric].to_numpy(dtype=float)
            deviation = design_rows[std_metric].to_numpy(dtype=float)
            style = PLOT_STYLES[design_key]
            line_width = 3.5 if design_key == "flyover" else 2.0
            axis.plot(
                x,
                y,
                label=design_name,
                linewidth=line_width,
                markersize=6,
                **style,
            )
            axis.fill_between(
                x,
                y - deviation,
                y + deviation,
                color=style["color"],
                alpha=0.08,
                linewidth=0,
            )
        axis.set_ylabel(label)
        axis.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0, 1.02)
    ranks = sorted(results["importance_rank"].unique())
    axes[-1].set_xticks(ranks, [f"#{int(rank)}" for rank in ranks])
    axes[-1].set_xlabel("Selected intersection (importance rank)")
    axes[0].legend(ncol=2, frameon=False, loc="lower right")
    figure.suptitle("Khlong Sam Wa: Four Intersection Designs by Site", fontsize=15)
    figure.text(
        0.5,
        0.01,
        "Shaded bands show +/- one standard deviation across repeated runs.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.97))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_demand_sweep_comparison(
    results: pd.DataFrame, output_path: str | Path
) -> Path:
    """Plot every design by site across low, medium, and high demand."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    levels = results[["demand_level", "total_demand_rate"]].drop_duplicates()
    panels = [
        ("traffic_efficiency", "traffic_efficiency_std", "Traffic efficiency"),
        ("avg_waiting_time_s", "avg_waiting_time_s_std", "Average waiting (s)"),
        ("throughput_veh_s", "throughput_veh_s_std", "Throughput (vehicles/s)"),
    ]
    figure, axes = plt.subplots(
        len(levels), 3, figsize=(18, 4.2 * len(levels)), sharex=True, squeeze=False
    )
    for row_index, level in levels.reset_index(drop=True).iterrows():
        level_rows = results[results["demand_level"] == level["demand_level"]]
        for column_index, (metric, std_metric, label) in enumerate(panels):
            axis = axes[row_index, column_index]
            for design_key, design_name in DESIGN_NAMES.items():
                design_rows = level_rows[
                    level_rows["design"] == design_key
                ].sort_values("importance_rank")
                x = design_rows["importance_rank"].to_numpy(dtype=float)
                y = design_rows[metric].to_numpy(dtype=float)
                deviation = design_rows[std_metric].to_numpy(dtype=float)
                style = PLOT_STYLES[design_key]
                axis.plot(
                    x,
                    y,
                    label=design_name,
                    linewidth=3.5 if design_key == "flyover" else 2.0,
                    markersize=5,
                    **style,
                )
                axis.fill_between(
                    x,
                    y - deviation,
                    y + deviation,
                    color=style["color"],
                    alpha=0.07,
                    linewidth=0,
                )
            axis.set_title(
                f"{str(level['demand_level']).title()} demand "
                f"({level['total_demand_rate']:g} veh/s)"
            )
            axis.set_ylabel(label)
            axis.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.8)
            axis.spines[["top", "right"]].set_visible(False)
            if metric == "traffic_efficiency":
                axis.set_ylim(0, 1.02)
            if row_index == len(levels) - 1:
                ranks = sorted(level_rows["importance_rank"].unique())
                axis.set_xticks(ranks, [f"#{int(rank)}" for rank in ranks])
                axis.set_xlabel("Intersection importance rank")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
    )
    figure.suptitle(
        "Khlong Sam Wa: Intersection Designs Across Traffic Demand",
        fontsize=16,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.925))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def export_replacement_traffic_map(
    results: pd.DataFrame, output_path: str | Path
) -> Path:
    """Write an interactive OSM map of design results by demand and site."""
    required = {
        "importance_rank",
        "osm_node_id",
        "latitude",
        "longitude",
        "street_names",
        "demand_level",
        "total_demand_rate",
        "design",
        "design_name",
        "completion_rate",
        "avg_waiting_time_s",
        "avg_queue_length",
        "throughput_veh_s",
        "traffic_efficiency",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"map results are missing columns: {sorted(missing)}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(results[list(required)].to_json(orient="records"))
    data_json = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    design_json = json.dumps(
        {
            "four_way": {"label": DESIGN_NAMES["four_way"], "symbol": "+"},
            "roundabout": {"label": DESIGN_NAMES["roundabout"], "symbol": "↻"},
            "flyover": {"label": DESIGN_NAMES["flyover"], "symbol": "↑"},
            "underpass": {"label": DESIGN_NAMES["underpass"], "symbol": "↓"},
        },
        ensure_ascii=False,
    )
    demand_options = "".join(
        f'<option value="{escape(str(level))}">{escape(str(level).title())} '
        f'({rate:g} veh/s)</option>'
        for level, rate in results[
            ["demand_level", "total_demand_rate"]
        ].drop_duplicates().itertuples(index=False, name=None)
    )
    design_options = "".join(
        f'<option value="{key}">{escape(label)}</option>'
        for key, label in DESIGN_NAMES.items()
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Khlong Sam Wa Traffic Replacement Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    :root {{ --ink:#17212b; --muted:#59636e; --panel:#ffffff; --line:#d7dde3; }}
    * {{ box-sizing:border-box; letter-spacing:0; }}
    html, body, #map {{ width:100%; height:100%; margin:0; }}
    body {{ font-family:Inter,Segoe UI,Arial,sans-serif; color:var(--ink); }}
    #map {{ background:#dfe7e5; }}
    .toolbar {{ position:absolute; z-index:1000; top:16px; left:16px; width:310px;
      background:var(--panel); border:1px solid var(--line); border-radius:6px;
      box-shadow:0 6px 22px rgba(23,33,43,.16); padding:14px; }}
    h1 {{ margin:0 0 12px; font-size:18px; line-height:1.25; }}
    .controls {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
    label {{ display:block; color:var(--muted); font-size:11px; font-weight:700;
      text-transform:uppercase; margin-bottom:5px; }}
    select {{ width:100%; min-width:0; height:36px; border:1px solid #aeb7c0;
      border-radius:4px; background:#fff; color:var(--ink); padding:0 8px; font-size:13px; }}
    .counts {{ display:grid; grid-template-columns:repeat(3,1fr); gap:6px; margin-top:12px; }}
    .count {{ border-top:3px solid var(--color); background:#f5f7f8; padding:7px 5px;
      text-align:center; font-size:11px; }}
    .count strong {{ display:block; font-size:17px; }}
    .method {{ margin:10px 0 0; color:var(--muted); font-size:11px; line-height:1.4; }}
    .traffic-marker {{ width:34px; height:34px; border:3px solid white; border-radius:50%;
      color:white; display:grid; place-items:center; font-weight:800; font-size:20px;
      line-height:1; box-shadow:0 2px 7px rgba(23,33,43,.4); }}
    .status-good {{ background:#15803d; }} .status-busy {{ background:#d97706; }}
    .status-bad {{ background:#c62828; }}
    .leaflet-popup-content-wrapper {{ border-radius:6px; }}
    .popup {{ min-width:220px; }} .popup h2 {{ font-size:15px; margin:0 0 3px; }}
    .popup .street {{ color:var(--muted); margin-bottom:9px; }}
    .metrics {{ display:grid; grid-template-columns:1fr auto; gap:5px 14px; font-size:12px; }}
    .metrics strong {{ text-align:right; }} .popup a {{ display:inline-block; margin-top:10px;
      color:#075ea8; font-size:12px; }}
    @media (max-width:600px) {{ .toolbar {{ top:8px; left:8px; width:calc(100% - 16px); }}
      .leaflet-control-zoom {{ margin-top:218px !important; }} }}
  </style>
</head>
<body>
  <main id="map" aria-label="Khlong Sam Wa traffic replacement map"></main>
  <section class="toolbar" aria-label="Map controls">
    <h1>Khlong Sam Wa Traffic Map</h1>
    <div class="controls">
      <div><label for="demand">Demand</label><select id="demand" data-testid="demand-select">{demand_options}</select></div>
      <div><label for="design">Design</label><select id="design" data-testid="design-select">{design_options}</select></div>
    </div>
    <div class="counts" aria-live="polite">
      <div class="count" style="--color:#15803d"><strong id="good-count">0</strong>Good</div>
      <div class="count" style="--color:#d97706"><strong id="busy-count">0</strong>Busy</div>
      <div class="count" style="--color:#c62828"><strong id="bad-count">0</strong>Bad</div>
    </div>
    <p class="method">Traffic efficiency: green ≥ 0.80, amber 0.60–0.79, red &lt; 0.60.</p>
  </section>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const results = {data_json};
    const designs = {design_json};
    const map = L.map('map', {{ zoomControl:true }});
    L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom:19, attribution:'&copy; OpenStreetMap contributors'
    }}).addTo(map);
    const markerLayer = L.layerGroup().addTo(map);
    const bounds = L.latLngBounds(results.map(row => [row.latitude, row.longitude]));
    map.fitBounds(bounds.pad(0.22));

    function escapeHtml(value) {{
      return String(value ?? '').replace(/[&<>"']/g, char => ({{
        '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#039;'
      }})[char]);
    }}
    function statusFor(efficiency) {{
      if (efficiency >= 0.8) return {{ key:'good', label:'Good traffic' }};
      if (efficiency >= 0.6) return {{ key:'busy', label:'Busy traffic' }};
      return {{ key:'bad', label:'Bad traffic' }};
    }}
    function metric(value, digits=2) {{ return Number(value).toFixed(digits); }}
    function render() {{
      markerLayer.clearLayers();
      const demand = document.getElementById('demand').value;
      const design = document.getElementById('design').value;
      const counts = {{ good:0, busy:0, bad:0 }};
      results.filter(row => row.demand_level === demand && row.design === design).forEach(row => {{
        const status = statusFor(row.traffic_efficiency); counts[status.key] += 1;
        const symbol = designs[design].symbol;
        const icon = L.divIcon({{
          className:'', iconSize:[34,34], iconAnchor:[17,17], popupAnchor:[0,-17],
          html:`<div class="traffic-marker status-${{status.key}}">${{symbol}}</div>`
        }});
        const street = row.street_names || `OSM node ${{row.osm_node_id}}`;
        const popup = `<div class="popup"><h2>#${{row.importance_rank}} · ${{escapeHtml(designs[design].label)}}</h2>
          <div class="street">${{escapeHtml(street)}} · ${{status.label}}</div>
          <div class="metrics"><span>Traffic efficiency</span><strong>${{metric(row.traffic_efficiency,3)}}</strong>
          <span>Average waiting</span><strong>${{metric(row.avg_waiting_time_s)}} s</strong>
          <span>Average queue</span><strong>${{metric(row.avg_queue_length)}} vehicles</strong>
          <span>Throughput</span><strong>${{metric(row.throughput_veh_s,3)}} veh/s</strong>
          <span>Completion rate</span><strong>${{metric(row.completion_rate*100,1)}}%</strong></div>
          <a href="https://www.openstreetmap.org/node/${{encodeURIComponent(row.osm_node_id)}}" target="_blank" rel="noopener">Open OSM location</a></div>`;
        L.marker([row.latitude,row.longitude], {{icon}}).bindTooltip(`#${{row.importance_rank}} · ${{status.label}}`).bindPopup(popup).addTo(markerLayer);
      }});
      for (const key of ['good','busy','bad']) document.getElementById(`${{key}}-count`).textContent = counts[key];
    }}
    document.getElementById('demand').addEventListener('change', render);
    document.getElementById('design').addEventListener('change', render);
    render();
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def plot_replacement_traffic_map(
    graph, results: pd.DataFrame, output_path: str | Path
) -> Path:
    """Plot an offline road map for every demand/design combination."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    levels = results[["demand_level", "total_demand_rate"]].drop_duplicates()
    if levels.empty:
        raise ValueError("cannot plot a replacement map without results")

    minor_segments, arterial_segments = [], []
    for start, end, data in graph.edges(data=True):
        geometry = data.get("geometry")
        if geometry is not None and hasattr(geometry, "coords"):
            segment = list(geometry.coords)
        else:
            start_data, end_data = graph.nodes[start], graph.nodes[end]
            segment = [
                (float(start_data["x"]), float(start_data["y"])),
                (float(end_data["x"]), float(end_data["y"])),
            ]
        highway = data.get("highway", "")
        highway_values = highway if isinstance(highway, list) else [highway]
        target = (
            arterial_segments
            if any(str(value) in ARTERIAL_CLASSES for value in highway_values)
            else minor_segments
        )
        target.append(segment)

    design_symbols = {
        "four_way": "+",
        "roundabout": "↻",
        "flyover": "↑",
        "underpass": "↓",
    }
    status_colors = {"good": "#15803d", "busy": "#d97706", "bad": "#c62828"}
    site_count = results["importance_rank"].nunique()
    marker_size = 105 if site_count > 20 else 165
    symbol_size = 10 if site_count > 20 else 13
    show_rank_labels = site_count <= 15
    figure, axes = plt.subplots(
        len(levels),
        len(DESIGN_NAMES),
        figsize=(18, 4.6 * len(levels)),
        squeeze=False,
    )
    min_lon, max_lon = results["longitude"].min(), results["longitude"].max()
    min_lat, max_lat = results["latitude"].min(), results["latitude"].max()
    mean_latitude = (min_lat + max_lat) / 2
    center_lon, center_lat = (min_lon + max_lon) / 2, mean_latitude
    longitude_scale = math.cos(math.radians(mean_latitude))
    longitude_span = max_lon - min_lon
    latitude_span = max_lat - min_lat
    geographic_width = longitude_span * longitude_scale
    target_span = max(geographic_width, latitude_span, 0.008)
    longitude_span = target_span / longitude_scale
    latitude_span = target_span
    min_lon, max_lon = center_lon - longitude_span / 2, center_lon + longitude_span / 2
    min_lat, max_lat = center_lat - latitude_span / 2, center_lat + latitude_span / 2
    lon_padding = longitude_span * 0.07
    lat_padding = latitude_span * 0.07

    for row_index, level in levels.reset_index(drop=True).iterrows():
        for column_index, (design_key, design_name) in enumerate(DESIGN_NAMES.items()):
            axis = axes[row_index, column_index]
            axis.add_collection(
                LineCollection(minor_segments, colors="#d5d9dc", linewidths=0.35)
            )
            axis.add_collection(
                LineCollection(arterial_segments, colors="#8b949c", linewidths=0.8)
            )
            rows = results[
                (results["demand_level"] == level["demand_level"])
                & (results["design"] == design_key)
            ].sort_values("importance_rank")
            for site in rows.itertuples(index=False):
                rank = int(site.importance_rank)
                status = (
                    "good"
                    if site.traffic_efficiency >= 0.8
                    else "busy" if site.traffic_efficiency >= 0.6 else "bad"
                )
                axis.scatter(
                    site.longitude,
                    site.latitude,
                    s=marker_size,
                    color=status_colors[status],
                    edgecolor="white",
                    linewidth=1.6,
                    zorder=4,
                )
                axis.text(
                    site.longitude,
                    site.latitude,
                    design_symbols[design_key],
                    color="white",
                    fontsize=symbol_size,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=5,
                )
                if show_rank_labels:
                    label_x, label_y = {
                        4: (-8, 8),
                        7: (-8, -11),
                    }.get(rank, (7, 6))
                    axis.annotate(
                        f"#{rank}",
                        (site.longitude, site.latitude),
                        xytext=(label_x, label_y),
                        textcoords="offset points",
                        fontsize=7,
                        color="#17212b",
                        ha="right" if label_x < 0 else "left",
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.78,
                            "pad": 1,
                        },
                        zorder=6,
                    )
            axis.set_xlim(min_lon - lon_padding, max_lon + lon_padding)
            axis.set_ylim(min_lat - lat_padding, max_lat + lat_padding)
            axis.set_aspect(1 / math.cos(math.radians(mean_latitude)))
            axis.set_title(
                f"{design_name}\n{str(level['demand_level']).title()} demand "
                f"({level['total_demand_rate']:g} veh/s)",
                fontsize=11,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#c8cdd2")

    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=10,
            label=label,
        )
        for color, label in (
            (status_colors["good"], "Good: efficiency ≥ 0.80"),
            (status_colors["busy"], "Busy: efficiency 0.60–0.79"),
            (status_colors["bad"], "Bad: efficiency < 0.60"),
        )
    ]
    figure.legend(handles=legend, ncol=3, frameon=False, loc="lower center")
    figure.suptitle(
        "Khlong Sam Wa: Simulated Traffic by Intersection Design",
        fontsize=17,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.97))
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def _write_selected_sites(graph, sites, output_dir: Path) -> dict:
    sites_path = output_dir / "selected_intersections.csv"
    sites_json_path = output_dir / "selected_intersections.json"
    sites_geojson_path = output_dir / "selected_intersections.geojson"
    sites.to_csv(sites_path, index=False, encoding="utf-8-sig")
    sites_json = build_selected_intersections_json(graph, sites)
    sites_json_path.write_text(
        json.dumps(sites_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    gpd.GeoDataFrame(
        sites.copy(),
        geometry=[Point(xy) for xy in zip(sites["longitude"], sites["latitude"])],
        crs="EPSG:4326",
    ).to_file(sites_geojson_path, driver="GeoJSON")
    return {
        "sites_csv": sites_path,
        "sites_json": sites_json_path,
        "sites_geojson": sites_geojson_path,
    }


def export_replacement_experiment(
    graph,
    output_dir: str | Path,
    *,
    site_count: int = 10,
    sim_time: int = 600,
    n_runs: int = 5,
    total_demand_rate: float = 1.0,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sites = select_important_four_way_intersections(graph, count=site_count)
    results = run_replacement_experiment(
        graph,
        sites,
        sim_time=sim_time,
        n_runs=n_runs,
        total_demand_rate=total_demand_rate,
    )
    results_path = output_dir / "replacement_results.csv"
    plot_path = output_dir / "replacement_comparison.png"
    site_paths = _write_selected_sites(graph, sites, output_dir)
    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    plot_replacement_comparison(results, plot_path)
    return {
        "sites": sites,
        "results": results,
        **site_paths,
        "results_csv": results_path,
        "comparison_plot": plot_path,
    }


def export_replacement_demand_sweep(
    graph,
    output_dir: str | Path,
    *,
    demand_levels: dict[str, float],
    site_count: int = 10,
    sim_time: int = 600,
    n_runs: int = 5,
) -> dict:
    """Run the four designs at the same selected sites for each demand level."""
    if not demand_levels or any(rate <= 0 for rate in demand_levels.values()):
        raise ValueError("demand levels must contain positive rates")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sites = select_important_four_way_intersections(graph, count=site_count)
    frames = []
    for level, rate in demand_levels.items():
        results = run_replacement_experiment(
            graph,
            sites,
            sim_time=sim_time,
            n_runs=n_runs,
            total_demand_rate=rate,
        )
        results.insert(6, "demand_level", level)
        results.insert(7, "total_demand_rate", rate)
        frames.append(results)
    combined = pd.concat(frames, ignore_index=True)

    results_path = output_dir / "replacement_demand_results.csv"
    summary_path = output_dir / "replacement_demand_summary.json"
    plot_path = output_dir / "replacement_demand_comparison.png"
    combined.to_csv(results_path, index=False, encoding="utf-8-sig")
    summary = (
        combined.groupby(
            ["demand_level", "total_demand_rate", "design", "design_name"],
            as_index=False,
            sort=False,
        )
        .agg(
            mean_efficiency=("traffic_efficiency", "mean"),
            efficiency_change_pct=("traffic_efficiency_change_pct", "mean"),
            mean_waiting_s=("avg_waiting_time_s", "mean"),
            waiting_change_pct=("avg_waiting_time_s_change_pct", "mean"),
            mean_throughput=("throughput_veh_s", "mean"),
        )
    )
    summary_path.write_text(
        json.dumps(summary.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_demand_sweep_comparison(combined, plot_path)
    site_paths = _write_selected_sites(graph, sites, output_dir)
    return {
        "sites": sites,
        "results": combined,
        "summary": summary,
        **site_paths,
        "results_csv": results_path,
        "summary_json": summary_path,
        "comparison_plot": plot_path,
    }
