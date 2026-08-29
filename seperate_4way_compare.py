"""Run the ten-site Khlong Sam Wa four-design comparison."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
sys.path.insert(0, str(ROOT / "src"))

from osm_network import load_osm_network_file
from replacement_experiment import export_replacement_demand_sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="graph_districts/khlong_sam_wa.graphml")
    parser.add_argument("--sites", type=int, default=10)
    parser.add_argument("--low-demand", type=float, default=0.5)
    parser.add_argument("--medium-demand", type=float, default=1.0)
    parser.add_argument("--high-demand", type=float, default=2.0)
    parser.add_argument("--time", type=int, default=600)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", default="output/khlong_sam_wa_replacements")
    args = parser.parse_args()
    rates = {
        "low": args.low_demand,
        "medium": args.medium_demand,
        "high": args.high_demand,
    }
    if (
        args.sites <= 0
        or any(rate <= 0 for rate in rates.values())
        or args.time <= 0
        or args.runs <= 0
    ):
        parser.error("sites, demand levels, time, and runs must be positive")

    osm = load_osm_network_file(args.graph)
    experiment = export_replacement_demand_sweep(
        osm.graph,
        args.output,
        demand_levels=rates,
        site_count=args.sites,
        sim_time=args.time,
        n_runs=args.runs,
    )
    print(
        f"\nCompared 4 designs at {len(experiment['sites'])} intersections "
        "under low, medium, and high demand."
    )
    print(experiment["summary"].to_string(index=False))
    print(f"\nReadable intersections: {experiment['sites_json']}")
    print(f"Full results: {experiment['results_csv']}")
    print(f"Readable summary: {experiment['summary_json']}")
    print(f"Comparison graph: {experiment['comparison_plot']}")


if __name__ == "__main__":
    main()
