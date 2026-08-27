"""
experiments.py
==============
Section 13: the four controlled experiments, plus compare_intersections()
(Section 24) which is the main entry point most users will call.

Every experiment:
  - keeps EVERY variable fixed except the one being tested (Section 22)
  - runs NUMBER_OF_RUNS repeated simulations per configuration with
    different seeds (Section 14) and aggregates mean/min/max/std
  - returns plain data structures (dicts / lists of AggregatedMetrics)
    so visualization.py and main.py can consume them without re-deriving
    anything
"""

import copy

import config as cfg
from designs import DESIGN_REGISTRY, DESIGN_DISPLAY_NAMES
from simulation import TrafficSimulation
from metrics import (
    compute_metrics,
    aggregate_runs,
    AggregatedMetrics,
    find_bottleneck_edges,
)
from graph_model import IntersectionNetwork
from demand import od_demand_to_inputs


def _run_design_repeated(
    design_key: str,
    demand: dict[str, float],
    sim_time: int,
    n_runs: int,
    base_seed: int,
    green_light_time: int = None,
    engine: str = "step",
    destination_probs_override: dict[str, dict[str, float]] | None = None,
):
    """Run one design n_runs times with different seeds; return
    (AggregatedMetrics, list_of_raw_SimulationResult, IntersectionNetwork)."""
    derived_runs = []
    raw_results = []
    net = None
    for i in range(n_runs):
        # rebuild fresh each run so vehicle-on-road state never leaks between runs
        net, controller, dest_probs = DESIGN_REGISTRY[design_key]()
        if destination_probs_override is not None:
            dest_probs = destination_probs_override
        if green_light_time is not None:
            _override_green_time(controller, green_light_time)
        simulator_class = TrafficSimulation
        if engine == "simpy":
            from simpy_simulation import SimPyTrafficSimulation

            simulator_class = SimPyTrafficSimulation
        elif engine != "step":
            raise ValueError("engine must be 'step' or 'simpy'")
        sim = simulator_class(
            net,
            controller,
            dest_probs,
            demand,
            design_key,
            DESIGN_DISPLAY_NAMES[design_key],
        )
        seed = base_seed + i
        result = sim.run(sim_time=sim_time, seed=seed)
        raw_results.append(result)
        derived_runs.append(compute_metrics(result, demand))
    aggregated = aggregate_runs(derived_runs)
    return aggregated, raw_results, net


def _override_green_time(controller, green_light_time: int):
    """Used by Experiment C to rescale a signal controller's green times
    while keeping the relative split between phases the same."""
    from controller import FixedTimeSignalController

    if not isinstance(controller, FixedTimeSignalController):
        return  # yield controllers (roundabout) have no signal timing
    n_phases = len(controller.green_times)
    controller.green_times = [green_light_time] * n_phases
    controller.cycle_length = sum(controller.green_times)


# ===========================================================================
# EXPERIMENT A -- same traffic demand, different intersection designs
# ===========================================================================
def run_experiment_a(
    designs: list[str] = None,
    demand: dict[str, float] = None,
    sim_time: int = None,
    n_runs: int = None,
    seed: int = None,
    engine: str = "step",
    od_demand: dict[tuple[str, str], float] | None = None,
):
    designs = designs or list(DESIGN_REGISTRY.keys())
    if od_demand is not None:
        demand, destination_probs = od_demand_to_inputs(od_demand)
    else:
        demand = demand or cfg.DEMAND
        destination_probs = None
    sim_time = sim_time or cfg.SIMULATION_TIME
    n_runs = n_runs or cfg.NUMBER_OF_RUNS
    seed = seed if seed is not None else cfg.RANDOM_SEED

    results = {}
    networks = {}
    sample_raw = {}
    all_raw = {}
    for key in designs:
        agg, raw, net = _run_design_repeated(
            key,
            demand,
            sim_time,
            n_runs,
            seed,
            engine=engine,
            destination_probs_override=destination_probs,
        )
        results[key] = agg
        networks[key] = net
        sample_raw[key] = raw[0]  # first seed's raw result, used for structural plots
        all_raw[key] = raw  # every repeated run, used to average time-series plots
    return {
        "results": results,
        "networks": networks,
        "sample_raw": sample_raw,
        "all_raw": all_raw,
        "demand": demand,
        "sim_time": sim_time,
        "n_runs": n_runs,
        "engine": engine,
        "od_demand": od_demand,
    }


# Section 24's requested public API name:
def compare_intersections(designs: list[str], simulation_time: int = None, **kwargs):
    return run_experiment_a(designs=designs, sim_time=simulation_time, **kwargs)


# ===========================================================================
# EXPERIMENT B -- increasing traffic demand
# ===========================================================================
DEMAND_LEVELS = {
    "Low": {"N": 1, "S": 1, "W": 1, "E": 1, "NE": 1},
    "Medium": {"N": 2, "S": 1, "W": 3, "E": 2, "NE": 1},
    "High": {"N": 2, "S": 1, "W": 5, "E": 2, "NE": 1},  # = default DEMAND
    "Very High": {"N": 4, "S": 2, "W": 8, "E": 4, "NE": 2},
}


def run_experiment_b(
    designs: list[str] = None,
    demand_levels: dict[str, dict] = None,
    sim_time: int = None,
    n_runs: int = None,
    seed: int = None,
    engine: str = "step",
):
    designs = designs or list(DESIGN_REGISTRY.keys())
    demand_levels = demand_levels or DEMAND_LEVELS
    sim_time = sim_time or cfg.SIMULATION_TIME
    n_runs = n_runs or cfg.NUMBER_OF_RUNS
    seed = seed if seed is not None else cfg.RANDOM_SEED

    # {design_key: {level_name: AggregatedMetrics}}
    results = {key: {} for key in designs}
    for level_name, demand in demand_levels.items():
        for key in designs:
            agg, _, _ = _run_design_repeated(
                key, demand, sim_time, n_runs, seed, engine=engine
            )
            results[key][level_name] = agg
    return {
        "results": results,
        "demand_levels": demand_levels,
        "sim_time": sim_time,
        "n_runs": n_runs,
        "engine": engine,
    }


# ===========================================================================
# EXPERIMENT C -- traffic-light timing (signalised designs only)
# ===========================================================================
GREEN_TIME_OPTIONS = [20, 30, 40, 50, 60]


def run_experiment_c(
    designs: list[str] = None,
    green_times: list[int] = None,
    demand: dict[str, float] = None,
    sim_time: int = None,
    n_runs: int = None,
    seed: int = None,
    engine: str = "step",
):
    from controller import FixedTimeSignalController

    all_signalised = [k for k in DESIGN_REGISTRY if not _is_roundabout(k)]
    designs = designs or all_signalised
    green_times = green_times or GREEN_TIME_OPTIONS
    demand = demand or cfg.DEMAND
    sim_time = sim_time or cfg.SIMULATION_TIME
    n_runs = n_runs or cfg.NUMBER_OF_RUNS
    seed = seed if seed is not None else cfg.RANDOM_SEED

    results = {key: {} for key in designs}
    for key in designs:
        for g in green_times:
            agg, _, _ = _run_design_repeated(
                key,
                demand,
                sim_time,
                n_runs,
                seed,
                green_light_time=g,
                engine=engine,
            )
            results[key][g] = agg
    return {
        "results": results,
        "green_times": green_times,
        "sim_time": sim_time,
        "n_runs": n_runs,
        "engine": engine,
    }


def _is_roundabout(design_key: str) -> bool:
    from controller import YieldController

    _, controller, _ = DESIGN_REGISTRY[design_key]()
    return isinstance(controller, YieldController)


# ===========================================================================
# EXPERIMENT D -- bottleneck analysis
# ===========================================================================
def run_experiment_d(
    designs: list[str] = None,
    demand: dict[str, float] = None,
    sim_time: int = None,
    n_runs: int = None,
    seed: int = None,
    engine: str = "step",
):
    designs = designs or list(DESIGN_REGISTRY.keys())
    demand = demand or cfg.DEMAND
    sim_time = sim_time or cfg.SIMULATION_TIME
    n_runs = n_runs or cfg.NUMBER_OF_RUNS
    seed = seed if seed is not None else cfg.RANDOM_SEED

    bottlenecks = {}
    for key in designs:
        # average saturation per edge across all repeated runs
        _, raw_results, net = _run_design_repeated(
            key, demand, sim_time, n_runs, seed, engine=engine
        )
        edge_sat_sums: dict = {}
        for r in raw_results:
            for edge, sat in r.edge_saturation.items():
                edge_sat_sums.setdefault(edge, []).append(sat)
        avg_sat = {edge: sum(v) / len(v) for edge, v in edge_sat_sums.items()}
        ranked = sorted(avg_sat.items(), key=lambda kv: kv[1], reverse=True)
        bottlenecks[key] = {"ranked_edges": ranked, "network": net}
    return bottlenecks


# ===========================================================================
# INTERPRETATION (Section 21) -- rule-based, references the actual numbers
# ===========================================================================
def interpret_experiment_a(exp_a_output: dict) -> str:
    results = exp_a_output["results"]
    ranked = sorted(
        results.items(), key=lambda kv: kv[1].stats["efficiency"]["mean"], reverse=True
    )
    best_key, best = ranked[0]
    worst_key, worst = ranked[-1]

    lines = []
    lines.append(
        f"Under identical traffic demand, '{best.design_display_name}' achieved the "
        f"highest mean efficiency score ({best.stats['efficiency']['mean']:.3f}), "
        f"completing {best.stats['completion_rate']['mean']:.1%} of generated vehicles "
        f"with an average wait of {best.stats['avg_waiting_time']['mean']:.1f}s."
    )
    lines.append(
        f"'{worst.design_display_name}' scored lowest ({worst.stats['efficiency']['mean']:.3f}), "
        f"completing only {worst.stats['completion_rate']['mean']:.1%} of vehicles with an "
        f"average wait of {worst.stats['avg_waiting_time']['mean']:.1f}s."
    )

    lines.append(
        "These differences come from graph structure, not driver behaviour: every design "
        "faced the exact same arrival rates and used the same routing algorithm. Designs with "
        "fewer conflicting movements at a single node (e.g. a T-intersection's 3 approaches, "
        "2 signal phases) can afford each approach a larger share of green time than designs "
        "with more conflicting movements (e.g. the five-way's 3 phases, or a standard four-way's "
        "2 phases splitting time across 4 approaches). Designs with extra structural nodes "
        "(the staggered intersection's link edge, the roundabout's ring segments) add travel "
        "distance and an additional capacity constraint that every crossing movement must clear, "
        "which shows up directly as extra waiting and lower throughput once demand is high."
    )
    lines.append(
        "These results describe ONLY the traffic assumptions and parameters used in this "
        "simulation (this specific demand pattern, these road capacities, this signal timing) "
        "-- they are not a universal ranking of intersection designs."
    )
    return "\n\n".join(lines)


def interpret_experiment_b(exp_b_output: dict) -> str:
    results = exp_b_output["results"]
    levels = list(exp_b_output["demand_levels"].keys())
    lines = []
    for key, per_level in results.items():
        name = per_level[levels[0]].design_display_name
        breaking_level = None
        for level in levels:
            if per_level[level].stats["completion_rate"]["mean"] < 0.8:
                breaking_level = level
                break
        if breaking_level:
            lines.append(
                f"- {name}: completion rate first drops below 80% at '{breaking_level}' demand."
            )
        else:
            lines.append(
                f"- {name}: stayed at or above 80% completion through every demand level tested."
            )
    lines.append(
        "\nAs demand rises, the SAME graph structure that looked fine at low volume can "
        "become a genuine bottleneck: any design whose controlled entry edges (or, for the "
        "roundabout, whose merge rate) cannot keep pace with arrivals will start to queue "
        "vehicles faster than it can release them, and completion rate falls even though no "
        "vehicle 'behaves' any differently -- only the arrival rate changed."
    )
    return "\n".join(lines)


def interpret_experiment_c(exp_c_output: dict) -> str:
    results = exp_c_output["results"]
    green_times_tested = exp_c_output["green_times"]
    shortest = min(green_times_tested)
    lines = []
    boundary_hits = 0
    for key, per_green in results.items():
        name = list(per_green.values())[0].design_display_name
        best_g = max(
            per_green.items(), key=lambda kv: kv[1].stats["efficiency"]["mean"]
        )
        lines.append(
            f"- {name}: best mean efficiency at green time = {best_g[0]}s "
            f"(efficiency={best_g[1].stats['efficiency']['mean']:.3f})."
        )
        if best_g[0] == shortest:
            boundary_hits += 1

    lines.append(
        "\nEvery phase in this project uses a FIXED SPLIT: each approach always gets the same "
        "SHARE of the cycle regardless of how long the cycle is, so making the cycle shorter "
        "does not reduce anyone's share of green time -- it just repeats that share more often. "
        "Classic signal-timing theory (Webster's delay formula) predicts that under this kind of "
        "fixed-split control, average delay grows roughly with cycle length as long as demand is "
        "below capacity, because a vehicle that just missed its green has to wait for a "
        "proportionally longer red either way."
    )
    if boundary_hits == len(results):
        lines.append(
            f"\nEvery design tested here was still improving at the SHORTEST green time tested "
            f"({shortest}s) -- this is a boundary effect, not proof that {shortest}s is optimal. "
            f"Real intersections cannot shrink the cycle indefinitely (driver reaction time and "
            f"the mandatory yellow/all-red clearance interval between phases set a practical "
            f"floor that this simplified model does not include); a full study would test shorter "
            f"green times specifically to find where efficiency actually peaks."
        )
    else:
        lines.append(
            "\nNot every design peaked at the shortest cycle tested, which means for at least one "
            "design demand was high enough relative to capacity that a longer green time was "
            "needed to avoid the queue growing faster than it could clear."
        )
    return "\n".join(lines)


def interpret_experiment_d(exp_d_output: dict) -> str:
    lines = []
    for key, data in exp_d_output.items():
        ranked = data["ranked_edges"]
        if not ranked:
            continue
        top_edge, top_sat = ranked[0]
        name = data["network"].name
        lines.append(
            f"- {name}: heaviest-used edge is {top_edge[0]}->{top_edge[1]} "
            f"(avg saturation {top_sat:.1%} of its capacity-time budget)."
        )
    lines.append(
        "\nBottlenecks are not always the edge you'd expect from demand alone. A structural "
        "edge that every 'crossing' route is forced through (the staggered link, a roundabout "
        "ring segment) can become the busiest part of the network even though no single "
        "approach feeding it has the highest raw demand -- this is a direct, measurable "
        "consequence of graph structure (see edge betweenness centrality) rather than of "
        "traffic volume alone."
    )
    return "\n".join(lines)
