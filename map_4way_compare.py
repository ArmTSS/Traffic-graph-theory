"""Create map visualizations from the latest Khlong Sam Wa results."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
sys.path.insert(0, str(ROOT / "src"))

from osm_network import load_osm_network_file
from district_replacement_experiment import export_coupled_district_sweep
from replacement_experiment import (
    export_replacement_traffic_map,
    plot_replacement_traffic_map,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="graph_districts/khlong_sam_wa.graphml")
    parser.add_argument(
        "--results",
        default="output/khlong_sam_wa_replacements/replacement_demand_results.csv",
    )
    parser.add_argument(
        "--picture",
        default="output/khlong_sam_wa_replacements/replacement_traffic_map.png",
    )
    parser.add_argument(
        "--interactive",
        default="output/khlong_sam_wa_replacements/replacement_traffic_map.html",
    )
    parser.add_argument(
        "--sites",
        type=int,
        help="show only the top N ranked intersections (default: all in results)",
    )
    parser.add_argument(
        "--coupled",
        action="store_true",
        help="simulate selected intersections together in the full district graph",
    )
    parser.add_argument("--time", type=int, default=600)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--portals", type=int, default=8)
    parser.add_argument("--low-demand", type=float, default=0.5)
    parser.add_argument("--medium-demand", type=float, default=1.0)
    parser.add_argument("--high-demand", type=float, default=2.0)
    parser.add_argument(
        "--coupled-output",
        default="output/khlong_sam_wa_replacements/coupled",
    )
    args = parser.parse_args()

    if args.sites is not None and args.sites <= 0:
        parser.error("--sites must be positive")
    if (
        args.time <= 0
        or args.runs <= 0
        or args.portals < 2
        or min(args.low_demand, args.medium_demand, args.high_demand) <= 0
    ):
        parser.error("time, runs, demand levels, and at least two portals are required")

    osm = load_osm_network_file(args.graph)
    if args.coupled:
        site_count = args.sites or 10
        experiment = export_coupled_district_sweep(
            osm.graph,
            args.coupled_output,
            site_count=site_count,
            demand_levels={
                "low": args.low_demand,
                "medium": args.medium_demand,
                "high": args.high_demand,
            },
            sim_time=args.time,
            n_runs=args.runs,
            portal_count=args.portals,
        )
        print(f"\nCoupled intersections: {len(experiment['sites'])}")
        print(f"Coupled results: {experiment['results_csv']}")
        print(f"Map picture: {experiment['picture']}")
        print(f"Interactive map: {experiment['interactive']}")
        return

    results_path = Path(args.results)
    if not results_path.is_file():
        parser.error(
            f"results file not found: {results_path}. "
            "Run seperate_4way_compare.py first."
        )
    results = pd.read_csv(results_path)
    available_sites = sorted(results["importance_rank"].unique())
    if args.sites is not None:
        if args.sites > len(available_sites):
            parser.error(
                f"requested {args.sites} sites, but the results contain only "
                f"{len(available_sites)}. Run seperate_4way_compare.py "
                f"--sites {args.sites} first."
            )
        selected_ranks = set(available_sites[: args.sites])
        results = results[results["importance_rank"].isin(selected_ranks)].copy()
    picture = plot_replacement_traffic_map(osm.graph, results, args.picture)
    interactive = export_replacement_traffic_map(results, args.interactive)
    print(f"\nMapped intersections: {results['importance_rank'].nunique()}")
    print(f"Map picture: {picture}")
    print(f"Interactive map: {interactive}")


if __name__ == "__main__":
    main()
