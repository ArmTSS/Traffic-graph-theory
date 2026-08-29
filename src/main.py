"""
main.py
=======
The final program (Section 24). Running this file:

  1. Validates every design (Section 23)
  2. Runs Experiment A -- same demand, different designs (the MAIN experiment)
  3. Runs Experiment B -- increasing demand
  4. Runs Experiment C -- signal timing sweep
  5. Runs Experiment D -- bottleneck analysis
  6. Prints a results table (Section 20) and graph-theory summary (Section 10)
  7. Prints research interpretation for every experiment (Section 21)
  8. Saves every required chart to OUTPUT_DIR (Section 15/16)

Usage:
    python main.py                 # full run using config.py defaults
    python main.py --quick         # fast smoke-test run (smaller sim/time/runs)
"""

import argparse
import json
import os
import sys
import pandas as pd

import config as cfg
from designs import DESIGN_REGISTRY, DESIGN_DISPLAY_NAMES
from graph_model import IntersectionNetwork
from experiments import (
    run_experiment_a,
    run_experiment_b,
    run_experiment_c,
    run_experiment_d,
    interpret_experiment_a,
    interpret_experiment_b,
    interpret_experiment_c,
    interpret_experiment_d,
    compare_intersections,
)
from validation import run_all_validations
from visualization import generate_all_plots
from osm_network import (
    intersection_candidates,
    extract_intersection_geometry,
    load_osm_network,
    load_osm_network_file,
    save_osm_network,
)
from osm_simulation import run_osm_simulation_study
from intersection_catalog import export_four_way_intersections
from replacement_experiment import export_replacement_experiment


def build_results_table(exp_a_output: dict) -> pd.DataFrame:
    """Section 20."""
    rows = []
    for key, agg in exp_a_output["results"].items():
        s = agg.stats
        rows.append(
            {
                "Intersection": DESIGN_DISPLAY_NAMES[key],
                "Avg Waiting (s)": round(s["avg_waiting_time"]["mean"], 1),
                "Max Queue": round(s["max_queue_length"]["mean"], 1),
                "Throughput (veh/s)": round(s["throughput_per_sec"]["mean"], 3),
                "Completion %": round(s["completion_rate"]["mean"] * 100, 1),
                "Avg Route Length (edges)": round(s["avg_route_length"]["mean"], 2),
                "Efficiency": round(s["efficiency"]["mean"], 3),
            }
        )
    return pd.DataFrame(rows)


def build_graph_theory_table(exp_a_output: dict) -> pd.DataFrame:
    """Section 10 summary, one row per design."""
    rows = []
    for key, net in exp_a_output["networks"].items():
        s = net.structural_summary()
        rows.append(
            {
                "Intersection": DESIGN_DISPLAY_NAMES[key],
                "|V| Nodes": s["num_nodes"],
                "|E| Edges": s["num_edges"],
                "Approaches": s["num_entry_approaches"],
                "Max Core Degree": s["max_core_degree"],
                "Avg Path Length (s)": s["avg_path_length_sec"],
                "Weighted Graph Efficiency": s["weighted_efficiency"],
                "Fully Connected": s["fully_connected"],
                "Highest-Betweenness Node": s["top_betweenness_node"],
            }
        )
    return pd.DataFrame(rows)


def print_independent_dependent_controlled_variables():
    """Section 22."""
    print("""
INDEPENDENT VARIABLES (what we deliberately change between experiments):
  - Intersection design / graph structure   (Experiment A)
  - Traffic volume / demand level           (Experiment B)
  - Traffic-light green time                (Experiment C)

DEPENDENT VARIABLES (what we measure as a result):
  - Average / maximum waiting time
  - Average / maximum queue length
  - Throughput and completion rate
  - Total travel time
  - Efficiency score

CONTROLLED VARIABLES (held constant so comparisons are fair):
  - Simulation duration (SIMULATION_TIME)
  - Vehicle generation method (Poisson arrivals at the configured rate)
  - Road capacity and travel-time defaults, except where explicitly varied
  - Destination probability rules (uniform, excluding U-turns)
  - Random-seed methodology (same base seed + run index across designs)

Controlling these variables matters because if two things changed at once
(e.g. different demand AND different capacities between two designs), a
difference in the results could no longer be attributed to any single
cause. Isolating one independent variable per experiment is what lets us
say "structure X caused effect Y" rather than "something changed".
""")


def main():
    parser = argparse.ArgumentParser(
        description="Intersection Design & Graph Theory traffic study"
    )
    parser.add_argument("--quick", action="store_true", help="fast smoke-test run")
    parser.add_argument(
        "--sim-time", type=int, help="override simulation duration in seconds"
    )
    parser.add_argument("--runs", type=int, help="override repeated-run count")
    parser.add_argument("--outdir", default=cfg.OUTPUT_DIR)
    parser.add_argument(
        "--engine",
        choices=("step", "simpy"),
        default=cfg.SIMULATION_ENGINE,
        help="traffic simulation backend",
    )
    parser.add_argument("--osm-place", help="optional OSM place to inspect")
    parser.add_argument("--osm-latitude", type=float)
    parser.add_argument("--osm-longitude", type=float)
    parser.add_argument("--osm-radius", type=float, default=cfg.OSM_ANALYSIS_RADIUS_M)
    parser.add_argument("--osm-node", help="optional OSM node ID for geometry")
    parser.add_argument(
        "--osm-load-graph",
        help="load a previously saved OSM GraphML file instead of downloading",
    )
    parser.add_argument(
        "--osm-save-graph",
        help="save the downloaded OSM graph to this GraphML file",
    )
    parser.add_argument(
        "--osm-export-four-way",
        help="export four-way candidates to CSV and GeoJSON using this file prefix",
    )
    parser.add_argument(
        "--run-replacements",
        action="store_true",
        help="compare four local designs at important four-way intersections",
    )
    parser.add_argument(
        "--replacement-sites",
        type=int,
        default=10,
        help="number of important intersections in the replacement experiment",
    )
    parser.add_argument(
        "--osm-simulate",
        action="store_true",
        help="convert the selected OSM network and run the real-map simulation",
    )
    parser.add_argument(
        "--osm-portals",
        type=int,
        default=cfg.OSM_PORTAL_COUNT,
        help="number of boundary origin/destination portals",
    )
    parser.add_argument(
        "--osm-demand-rate",
        type=float,
        default=cfg.OSM_TOTAL_DEMAND_PER_SECOND,
        help="total synthetic arrivals per second across all OSM portals",
    )
    parser.add_argument(
        "--od-demand",
        help='optional JSON file with OD keys such as {"N->S": 2}',
    )
    args = parser.parse_args()
    cfg.validate_config()

    if (args.osm_latitude is None) != (args.osm_longitude is None):
        parser.error("--osm-latitude and --osm-longitude must be supplied together")
    if args.osm_load_graph and (
        args.osm_place or args.osm_latitude is not None or args.osm_longitude is not None
    ):
        parser.error("--osm-load-graph cannot be combined with a place or coordinate")
    if args.osm_save_graph and not (
        args.osm_place or args.osm_latitude is not None or args.osm_load_graph
    ):
        parser.error("--osm-save-graph requires a place, coordinate, or loaded graph")
    if args.osm_export_four_way and not (
        args.osm_place or args.osm_latitude is not None or args.osm_load_graph
    ):
        parser.error("--osm-export-four-way requires a place, coordinate, or loaded graph")
    if args.run_replacements and not (
        args.osm_place or args.osm_latitude is not None or args.osm_load_graph
    ):
        parser.error("--run-replacements requires a place, coordinate, or loaded graph")
    if args.replacement_sites <= 0:
        parser.error("--replacement-sites must be positive")

    od_demand = None
    if args.od_demand:
        try:
            with open(args.od_demand, encoding="utf-8") as demand_file:
                raw_demand = json.load(demand_file)
            od_demand = {}
            for key, value in raw_demand.items():
                origin, destination = key.split("->", maxsplit=1)
                od_demand[(origin.strip(), destination.strip())] = float(value)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            parser.error(f"invalid OD demand file: {exc}")

    sim_time = args.sim_time if args.sim_time is not None else (
        60 if args.quick else cfg.SIMULATION_TIME
    )
    n_runs = args.runs if args.runs is not None else (
        3 if args.quick else cfg.NUMBER_OF_RUNS
    )
    if sim_time <= 0 or n_runs <= 0:
        parser.error("--sim-time and --runs must be positive")

    if (
        args.osm_place
        or args.osm_latitude is not None
        or args.osm_load_graph
        or args.osm_export_four_way
        or args.run_replacements
        or args.osm_simulate
    ):
        print("\nOSM NETWORK INSPECTION")
        try:
            if args.osm_load_graph:
                osm = load_osm_network_file(args.osm_load_graph)
                print(f"Loaded graph: {args.osm_load_graph}")
            else:
                osm = load_osm_network(
                    place=args.osm_place or cfg.OSM_PLACE,
                    latitude=args.osm_latitude,
                    longitude=args.osm_longitude,
                    radius_m=args.osm_radius,
                )
            if args.osm_save_graph:
                saved_graph = save_osm_network(osm, args.osm_save_graph)
                print(f"Saved graph: {saved_graph}")
            candidates = intersection_candidates(osm.graph)
            print(f"Source: {osm.source}")
            print(
                f"Network: {osm.graph.number_of_nodes()} nodes, "
                f"{osm.graph.number_of_edges()} edges"
            )
            print(f"Top intersection candidates: {candidates[:5]}")
            if args.osm_export_four_way:
                catalog = export_four_way_intersections(
                    osm.graph, args.osm_export_four_way
                )
                table = catalog["table"]
                likely_count = int(table["likely_geometric_cross"].sum())
                print(
                    f"Four-way catalog: {len(table)} topology candidates, "
                    f"{likely_count} likely geometric crosses"
                )
                print(f"Saved CSV: {catalog['csv']}")
                print(f"Saved GeoJSON: {catalog['geojson']}")
                if not args.osm_simulate and not args.run_replacements:
                    return
            if args.run_replacements:
                print("\nLOCAL INTERSECTION REPLACEMENT EXPERIMENT")
                replacement = export_replacement_experiment(
                    osm.graph,
                    args.outdir,
                    site_count=args.replacement_sites,
                    sim_time=sim_time,
                    n_runs=n_runs,
                    total_demand_rate=args.osm_demand_rate,
                )
                summary = (
                    replacement["results"]
                    .groupby("design_name", as_index=False)
                    .agg(
                        mean_efficiency=("traffic_efficiency", "mean"),
                        mean_efficiency_change_pct=(
                            "traffic_efficiency_change_pct",
                            "mean",
                        ),
                        mean_wait_s=("avg_waiting_time_s", "mean"),
                        mean_throughput=("throughput_veh_s", "mean"),
                    )
                    .sort_values("mean_efficiency", ascending=False)
                )
                print("\n--- Selected Sites ---")
                print(
                    replacement["sites"][
                        [
                            "importance_rank",
                            "osm_node_id",
                            "street_names",
                            "highway_types",
                            "betweenness_centrality",
                        ]
                    ].to_string(index=False)
                )
                print("\n--- Mean Results Across Sites ---")
                print(summary.to_string(index=False))
                print(f"\nSaved sites: {replacement['sites_csv']}")
                print(f"Saved readable site JSON: {replacement['sites_json']}")
                print(f"Saved site map: {replacement['sites_geojson']}")
                print(f"Saved results: {replacement['results_csv']}")
                print(f"Saved comparison graph: {replacement['comparison_plot']}")
                print(
                    "Flyover and underpass intentionally use the same traffic "
                    "graph. Their traffic results should match unless separate "
                    "speed, capacity, cost, or risk assumptions are introduced."
                )
                if not args.osm_simulate:
                    return
            if args.osm_node is not None:
                node_id = args.osm_node
                if node_id not in osm.nodes.index:
                    try:
                        node_id = int(node_id)
                    except ValueError:
                        pass
                geometry = extract_intersection_geometry(osm.nodes, osm.edges, node_id)
                print(
                    f"Selected node {node_id}: "
                    f"{len(geometry.nearby_edges)} nearby road geometries, "
                    f"{geometry.projected_crs}"
                )
            if args.osm_simulate:
                print("\nOSM REAL-MAP SIMULATION")
                study = run_osm_simulation_study(
                    osm.graph,
                    name=osm.source,
                    sim_time=sim_time,
                    n_runs=n_runs,
                    total_demand_rate=args.osm_demand_rate,
                    portal_count=args.osm_portals,
                    engine=args.engine,
                )
                aggregate = study["traffic_metrics"]
                stats = aggregate.stats
                traffic_table = pd.DataFrame(
                    [
                        {
                            "Network": osm.source,
                            "Runs": n_runs,
                            "Generated (mean)": round(
                                sum(r.vehicles_generated for r in aggregate.raw_runs)
                                / n_runs,
                                1,
                            ),
                            "Completion %": round(
                                stats["completion_rate"]["mean"] * 100, 2
                            ),
                            "Avg Waiting (s)": round(
                                stats["avg_waiting_time"]["mean"], 2
                            ),
                            "Avg Queue": round(
                                stats["avg_queue_length"]["mean"], 2
                            ),
                            "Throughput (veh/s)": round(
                                stats["throughput_per_sec"]["mean"], 3
                            ),
                            "Efficiency": round(stats["efficiency"]["mean"], 3),
                        }
                    ]
                )
                graph = study["graph_metrics"]
                graph_table = pd.DataFrame(
                    [
                        {
                            "Network": osm.source,
                            "Nodes": graph["nodes"],
                            "Edges": graph["edges"],
                            "OD Portals": graph["portals"],
                            "Avg Portal Path (s)": round(
                                graph["avg_portal_path_time_s"], 2
                            ),
                            "Demand-Weighted Graph Efficiency": round(
                                graph["demand_weighted_graph_efficiency"], 6
                            ),
                            "Highest-Betweenness Node": graph[
                                "top_betweenness_nodes"
                            ][0][0],
                        }
                    ]
                )
                bottleneck_table = pd.DataFrame(
                    [
                        {
                            "Road": f"{start} -> {end}",
                            "Mean Saturation": round(saturation, 4),
                            "Length (m)": study["network"].nx_graph.edges[
                                start, end
                            ].get("length_m"),
                            "Highway": study["network"].nx_graph.edges[
                                start, end
                            ].get("highway"),
                            "Lanes": study["network"].nx_graph.edges[
                                start, end
                            ].get("lanes"),
                        }
                        for (start, end), saturation in study["bottlenecks"]
                    ]
                )
                print("\n--- Traffic Metrics ---")
                print(traffic_table.to_string(index=False))
                print("\n--- Graph-Theory Metrics ---")
                print(graph_table.to_string(index=False))
                print("\n--- Most Saturated Roads ---")
                print(bottleneck_table.to_string(index=False))
                print(
                    "\nAssumption: OSM supplies geometry and road tags, while "
                    "vehicle demand and control are synthetic. Unknown signal "
                    "timing is modeled as free-flow capacity control."
                )
                os.makedirs(args.outdir, exist_ok=True)
                traffic_path = os.path.join(args.outdir, "osm_traffic_metrics.csv")
                graph_path = os.path.join(args.outdir, "osm_graph_metrics.csv")
                bottleneck_path = os.path.join(args.outdir, "osm_bottlenecks.csv")
                traffic_table.to_csv(traffic_path, index=False)
                graph_table.to_csv(graph_path, index=False)
                bottleneck_table.to_csv(bottleneck_path, index=False)
                print(f"\nSaved: {traffic_path}, {graph_path}, {bottleneck_path}")
                return
        except (KeyError, RuntimeError, ValueError) as exc:
            parser.error(f"OSM study failed: {exc}")

    os.makedirs(args.outdir, exist_ok=True)

    # -----------------------------------------------------------------
    print("#" * 78)
    print("# ANALYSIS OF THE IMPACT OF INTERSECTION DESIGN ON TRAFFIC FLOW")
    print("# USING GRAPH THEORY")
    print("#" * 78)

    print(
        f"\nConfiguration: SIMULATION_TIME={sim_time}s, NUMBER_OF_RUNS={n_runs}, "
        f"RANDOM_SEED={cfg.RANDOM_SEED}, ENGINE={args.engine}"
    )
    print(f"Traffic demand (vehicles/sec): {cfg.DEMAND}")
    if od_demand is not None:
        print(f"OD demand (vehicles/sec): {od_demand}")
    print(f"Signal green time: {cfg.GREEN_LIGHT_TIME}s")

    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("STEP 0: VALIDATION (Section 23)")
    print("=" * 78)
    report = run_all_validations()
    report.print_report()
    if not report.all_passed:
        print(
            "\nAborting: validation failed. Fix the reported issue before trusting results."
        )
        sys.exit(1)

    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SECTION 22: EXPERIMENTAL VARIABLES")
    print("=" * 78)
    print_independent_dependent_controlled_variables()

    # -----------------------------------------------------------------
    print("=" * 78)
    print("EXPERIMENT A: Same traffic demand, different intersection designs")
    print("=" * 78)
    exp_a = compare_intersections(
        designs=list(DESIGN_REGISTRY.keys()),
        simulation_time=sim_time,
        n_runs=n_runs,
        engine=args.engine,
        od_demand=od_demand,
    )

    results_table = build_results_table(exp_a)
    graph_table = build_graph_theory_table(exp_a)

    print("\n--- Results Table (Section 20) ---")
    print(results_table.to_string(index=False))
    print("\n--- Graph-Theory Structural Summary (Section 10) ---")
    print(graph_table.to_string(index=False))

    print("\n--- Interpretation (Section 21) ---")
    print(interpret_experiment_a(exp_a))

    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("EXPERIMENT B: Increasing traffic demand")
    print("=" * 78)
    exp_b = run_experiment_b(sim_time=sim_time, n_runs=n_runs, engine=args.engine)
    print("\n--- Interpretation ---")
    print(interpret_experiment_b(exp_b))

    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("EXPERIMENT C: Traffic-light timing sweep (signalised designs only)")
    print("=" * 78)
    exp_c = run_experiment_c(sim_time=sim_time, n_runs=n_runs, engine=args.engine)
    print("\n--- Interpretation ---")
    print(interpret_experiment_c(exp_c))

    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("EXPERIMENT D: Bottleneck analysis")
    print("=" * 78)
    exp_d = run_experiment_d(sim_time=sim_time, n_runs=n_runs, engine=args.engine)
    print("\n--- Interpretation ---")
    print(interpret_experiment_d(exp_d))

    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("GENERATING VISUALIZATIONS (Section 15/16)")
    print("=" * 78)
    paths = generate_all_plots(exp_a, exp_b, args.outdir)
    for p in paths:
        print(f"  saved: {p}")

    results_table.to_csv(os.path.join(args.outdir, "results_table.csv"), index=False)
    graph_table.to_csv(os.path.join(args.outdir, "graph_theory_table.csv"), index=False)
    print(f"  saved: {os.path.join(args.outdir, 'results_table.csv')}")
    print(f"  saved: {os.path.join(args.outdir, 'graph_theory_table.csv')}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("OVERALL CONCLUSION")
    print("=" * 78)
    best_key = max(
        exp_a["results"].items(), key=lambda kv: kv[1].stats["efficiency"]["mean"]
    )[0]
    print(f"""
Across every experiment, the graph structure of an intersection changed
traffic outcomes even though vehicle behaviour, routing rules, and total
demand were held constant. Under the specific demand pattern used here,
'{DESIGN_DISPLAY_NAMES[best_key]}' produced the best overall efficiency score,
but Experiment B shows this ranking is NOT fixed -- some designs that
struggle at high demand (e.g. the roundabout's yield-limited merging)
perform very competitively at low demand, and Experiment C shows every
signalised design has a different "best" cycle length. This supports the
central claim of the project: intersection structure, expressed as a
graph, is a first-order cause of differences in connectivity, routing,
bottleneck location, queueing, and ultimately traffic flow -- not merely
a cosmetic difference in how an intersection looks.
""")

    print("Done. All charts and tables were written to:", args.outdir)


if __name__ == "__main__":
    main()
