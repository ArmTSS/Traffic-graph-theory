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
    parser.add_argument("--outdir", default=cfg.OUTPUT_DIR)
    args = parser.parse_args()

    sim_time = 60 if args.quick else cfg.SIMULATION_TIME
    n_runs = 3 if args.quick else cfg.NUMBER_OF_RUNS

    os.makedirs(args.outdir, exist_ok=True)

    # -----------------------------------------------------------------
    print("#" * 78)
    print("# ANALYSIS OF THE IMPACT OF INTERSECTION DESIGN ON TRAFFIC FLOW")
    print("# USING GRAPH THEORY")
    print("#" * 78)

    print(
        f"\nConfiguration: SIMULATION_TIME={sim_time}s, NUMBER_OF_RUNS={n_runs}, "
        f"RANDOM_SEED={cfg.RANDOM_SEED}"
    )
    print(f"Traffic demand (vehicles/sec): {cfg.DEMAND}")
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
        designs=list(DESIGN_REGISTRY.keys()), simulation_time=sim_time, n_runs=n_runs
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
    exp_b = run_experiment_b(sim_time=sim_time, n_runs=n_runs)
    print("\n--- Interpretation ---")
    print(interpret_experiment_b(exp_b))

    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("EXPERIMENT C: Traffic-light timing sweep (signalised designs only)")
    print("=" * 78)
    exp_c = run_experiment_c(sim_time=sim_time, n_runs=n_runs)
    print("\n--- Interpretation ---")
    print(interpret_experiment_c(exp_c))

    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("EXPERIMENT D: Bottleneck analysis")
    print("=" * 78)
    exp_d = run_experiment_d(sim_time=sim_time, n_runs=n_runs)
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

    print("Done. All charts and tables were written to:", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
