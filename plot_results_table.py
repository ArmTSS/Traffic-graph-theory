"""
plot_results_table.py
=====================
Generate a multi-panel comparison chart from output/results_table.csv.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_results(csv_path: Path, output_path: Path):
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    
    # Set clean modern style
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
    plt.rcParams["axes.edgecolor"] = "#d1d5db"
    plt.rcParams["axes.linewidth"] = 0.8

    # Define color scheme for the intersections
    color_map = {
        "Four-Way": "#3b82f6",       # Blue
        "T-Intersection": "#10b981", # Green
        "Staggered": "#f59e0b",      # Amber
        "Roundabout": "#ef4444",     # Red
        "Five-Way": "#8b5cf6",       # Purple
    }
    
    intersections = df["Intersection"].tolist()
    bar_colors = [color_map.get(name, "#6b7280") for name in intersections]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), dpi=180)
    fig.patch.set_facecolor("#f8fafc")

    metrics = [
        {
            "col": "Efficiency",
            "title": "Composite Traffic Efficiency Score",
            "ylabel": "Score (0 - 1, higher is better)",
            "ylim": (0, 0.8),
            "fmt": "{:.3f}",
            "highlight_max": True,
        },
        {
            "col": "Completion %",
            "title": "Vehicle Completion Rate",
            "ylabel": "Percentage (%)",
            "ylim": (0, 100),
            "fmt": "{:.1f}%",
            "highlight_max": True,
        },
        {
            "col": "Throughput (veh/s)",
            "title": "Throughput",
            "ylabel": "Vehicles / Second",
            "ylim": (0, 7.5),
            "fmt": "{:.2f}",
            "highlight_max": True,
        },
        {
            "col": "Avg Waiting (s)",
            "title": "Average Waiting Time",
            "ylabel": "Seconds (lower is better)",
            "ylim": (0, 25),
            "fmt": "{:.1f}s",
            "highlight_max": False, # lower is better
        },
        {
            "col": "Max Queue",
            "title": "Maximum Queue Length",
            "ylabel": "Vehicles (lower is better)",
            "ylim": (0, 650),
            "fmt": "{:.0f}",
            "highlight_max": False,
        },
        {
            "col": "Avg Route Length (edges)",
            "title": "Average Route Length",
            "ylabel": "Edges / Hops",
            "ylim": (0, 6),
            "fmt": "{:.2f}",
            "highlight_max": False,
        },
    ]

    for idx, (ax, m) in enumerate(zip(axes.flatten(), metrics)):
        ax.set_facecolor("#ffffff")
        col = m["col"]
        values = df[col].to_numpy()
        
        bars = ax.bar(
            intersections,
            values,
            color=bar_colors,
            width=0.55,
            edgecolor="#ffffff",
            linewidth=1.2,
            zorder=3,
        )

        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                m["fmt"].format(val),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="600",
                color="#1e293b",
            )

        ax.set_title(m["title"], fontsize=12.5, fontweight="bold", pad=12, color="#0f172a")
        ax.set_ylabel(m["ylabel"], fontsize=10, color="#475569")
        ax.set_ylim(m["ylim"])
        ax.tick_params(axis="x", rotation=15, labelsize=9.5)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.5, color="#cbd5e1", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Intersection Performance Comparison (results_table.csv)",
        fontsize=16,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )
    
    # Add footnote explaining key takeaways
    fig.text(
        0.5,
        0.015,
        "Note: T-Intersection achieves highest efficiency (0.632) due to lower movement conflicts (3 legs, 2 phases); Roundabout experiences high delays at this volume due to circulating ring capacity.",
        ha="center",
        fontsize=9.5,
        color="#64748b",
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.035, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Chart successfully saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot intersection results from results_table.csv")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("output/results_table.csv"),
        help="Path to results_table.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/results_table_chart.png"),
        help="Output image path (.png)",
    )
    args = parser.parse_args()
    plot_results(args.csv, args.output)


if __name__ == "__main__":
    main()
