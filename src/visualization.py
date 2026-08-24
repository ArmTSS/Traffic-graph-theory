"""Plotting helpers for the traffic intersection experiments."""

import os

from designs import DESIGN_DISPLAY_NAMES


def _save_bar_chart(labels, values, title, ylabel, path):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(labels, values, color="#2f6f8f")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def generate_all_plots(
    exp_a_output: dict, exp_b_output: dict, outdir: str
) -> list[str]:
    """Generate summary charts and return their file paths."""
    os.makedirs(outdir, exist_ok=True)
    paths = []
    design_keys = list(exp_a_output["results"])
    labels = [DESIGN_DISPLAY_NAMES[key] for key in design_keys]

    efficiency = [
        exp_a_output["results"][key].stats["efficiency"]["mean"] for key in design_keys
    ]
    efficiency_path = os.path.join(outdir, "efficiency_by_design.png")
    _save_bar_chart(
        labels,
        efficiency,
        "Mean efficiency by intersection design",
        "Efficiency score",
        efficiency_path,
    )
    paths.append(efficiency_path)

    average_wait = [
        exp_a_output["results"][key].stats["avg_waiting_time"]["mean"]
        for key in design_keys
    ]
    wait_path = os.path.join(outdir, "waiting_time_by_design.png")
    _save_bar_chart(
        labels,
        average_wait,
        "Mean waiting time by intersection design",
        "Waiting time (seconds)",
        wait_path,
    )
    paths.append(wait_path)

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5))
    for key in design_keys:
        level_results = exp_b_output["results"][key]
        levels = list(level_results)
        values = [level_results[level].stats["efficiency"]["mean"] for level in levels]
        axis.plot(levels, values, marker="o", label=DESIGN_DISPLAY_NAMES[key])
    axis.set_title("Efficiency as traffic demand increases")
    axis.set_xlabel("Demand level")
    axis.set_ylabel("Efficiency score")
    axis.legend()
    figure.tight_layout()
    demand_path = os.path.join(outdir, "efficiency_by_demand.png")
    figure.savefig(demand_path, dpi=160)
    plt.close(figure)
    paths.append(demand_path)

    return paths
