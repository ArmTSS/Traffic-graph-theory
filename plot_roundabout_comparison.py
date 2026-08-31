"""Plot old vs new roundabout results side-by-side."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    old = pd.read_csv("output/results_table_OLD.csv")
    new = pd.read_csv("output/results_table.csv")

    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#d1d5db"
    plt.rcParams["axes.linewidth"] = 0.8

    color_map = {
        "Four-Way": "#3b82f6",
        "T-Intersection": "#10b981",
        "Staggered": "#f59e0b",
        "Roundabout": "#ef4444",
        "Five-Way": "#8b5cf6",
    }

    intersections = old["Intersection"].tolist()
    x = np.arange(len(intersections))
    bar_w = 0.32

    metrics = [
        ("Efficiency", "Composite Traffic Efficiency", (0, 0.8), "{:.3f}", True),
        ("Completion %", "Vehicle Completion Rate (%)", (0, 100), "{:.1f}", True),
        ("Throughput (veh/s)", "Throughput (veh/s)", (0, 7.5), "{:.2f}", True),
        ("Avg Waiting (s)", "Avg Waiting Time (s) — lower is better", (0, 25), "{:.1f}", False),
        ("Max Queue", "Max Queue Length — lower is better", (0, 600), "{:.0f}", False),
        ("Avg Route Length (edges)", "Avg Route Length (edges)", (0, 6), "{:.2f}", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), dpi=180)
    fig.patch.set_facecolor("#f8fafc")

    for ax, (col, title, ylim, fmt, higher_better) in zip(axes.flatten(), metrics):
        ax.set_facecolor("#ffffff")
        old_vals = old[col].to_numpy()
        new_vals = new[col].to_numpy()

        bars_old = ax.bar(
            x - bar_w / 2, old_vals, bar_w,
            color=["#cbd5e1"] * len(intersections),
            edgecolor="#94a3b8", linewidth=0.8,
            label="Old Model", zorder=3,
        )
        bar_colors = [color_map.get(n, "#6b7280") for n in intersections]
        bars_new = ax.bar(
            x + bar_w / 2, new_vals, bar_w,
            color=bar_colors,
            edgecolor="#ffffff", linewidth=0.8,
            label="Improved Model", zorder=3,
        )

        for bar, val in zip(bars_old, old_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + ylim[1] * 0.012,
                fmt.format(val), ha="center", va="bottom",
                fontsize=8, color="#64748b",
            )
        for i, (bar, val) in enumerate(zip(bars_new, new_vals)):
            change = val - old_vals[i]
            is_roundabout = intersections[i] == "Roundabout"
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + ylim[1] * 0.012,
                fmt.format(val), ha="center", va="bottom",
                fontsize=8, fontweight="bold" if is_roundabout else "normal",
                color="#dc2626" if is_roundabout else "#1e293b",
            )
            if abs(change) > 0.001 and is_roundabout:
                sign = "+" if change > 0 else ""
                improvement = (
                    (higher_better and change > 0) or
                    (not higher_better and change < 0)
                )
                arrow_color = "#15803d" if improvement else "#dc2626"
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + ylim[1] * 0.06,
                    f"{sign}{fmt.format(change)}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                    color=arrow_color,
                    bbox=dict(facecolor="white", edgecolor=arrow_color, alpha=0.9, pad=1.5, linewidth=0.8),
                )

        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=12, color="#0f172a")
        ax.set_xticks(x)
        ax.set_xticklabels(intersections, fontsize=9.5, rotation=15)
        ax.set_ylim(ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.45, color="#cbd5e1", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0, 0].legend(fontsize=9.5, loc="upper right", framealpha=0.9)

    fig.suptitle(
        "Roundabout Model Improvement: Old vs. New Results",
        fontsize=17, fontweight="bold", color="#0f172a", y=0.99,
    )
    fig.text(
        0.5, 0.012,
        "Grey = Old binary model (1 merge/s, capacity 6, any-vehicle blocking)  •  "
        "Color = Improved model (2 merges/s, capacity 10, 50% occupancy gap threshold)  •  "
        "Green annotations = Roundabout improvement",
        ha="center", fontsize=9.5, color="#64748b", style="italic",
    )
    plt.tight_layout(rect=[0, 0.035, 1, 0.96])

    out = Path("output/roundabout_improvement_comparison.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Comparison chart saved to: {out}")


if __name__ == "__main__":
    main()
