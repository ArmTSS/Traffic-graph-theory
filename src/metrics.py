"""
metrics.py
==========
Turns a raw SimulationResult (or a list of them, for repeated runs) into
the traffic-flow statistics required by Section 11, the efficiency score
required by Section 12, and the repeated-run mean/min/max/std required
by Section 14.
"""

import statistics as st
from dataclasses import dataclass
from typing import List, Dict, Tuple

import config as cfg
from simulation import SimulationResult


def _safe_mean(xs):
    return st.mean(xs) if xs else 0.0


def _safe_max(xs):
    return max(xs) if xs else 0.0


@dataclass
class DerivedMetrics:
    design_key: str
    design_display_name: str
    seed: int

    vehicles_generated: int
    vehicles_completed: int
    vehicles_remaining: int
    completion_rate: float

    avg_waiting_time: float
    max_waiting_time: float
    avg_queue_length: float
    max_queue_length: float
    congestion_index: float  # avg queue length PER APPROACH (fair across designs)

    total_travel_time: float
    total_waiting_time: float
    avg_route_length: float

    throughput_per_sec: float  # completed vehicles / sim_time
    normalized_throughput: float  # throughput / total demand rate, capped at 1.0

    efficiency: float  # Section 12 composite score, in [0, 1]


def compute_metrics(
    result: SimulationResult, demand_per_second: dict[str, float]
) -> DerivedMetrics:
    completion_rate = (
        result.vehicles_completed / result.vehicles_generated
        if result.vehicles_generated
        else 0.0
    )

    avg_wait = _safe_mean(result.completed_waiting_times)
    max_wait = _safe_max(result.completed_waiting_times)
    avg_queue = _safe_mean(result.queue_length_over_time)
    max_queue = _safe_max(result.queue_length_over_time)
    congestion_index = avg_queue / max(result.num_entry_approaches, 1)

    total_travel = sum(result.completed_travel_times)
    total_wait = sum(result.completed_waiting_times)
    avg_route_len = _safe_mean(result.completed_route_lengths)

    throughput = result.vehicles_completed / result.sim_time if result.sim_time else 0.0
    total_demand_rate = sum(demand_per_second.values())
    normalized_throughput = (
        min(throughput / total_demand_rate, 1.0) if total_demand_rate else 0.0
    )

    efficiency = compute_efficiency(
        completion_rate, normalized_throughput, avg_wait, congestion_index
    )

    return DerivedMetrics(
        design_key=result.design_key,
        design_display_name=result.design_display_name,
        seed=result.seed,
        vehicles_generated=result.vehicles_generated,
        vehicles_completed=result.vehicles_completed,
        vehicles_remaining=result.vehicles_remaining,
        completion_rate=completion_rate,
        avg_waiting_time=avg_wait,
        max_waiting_time=max_wait,
        avg_queue_length=avg_queue,
        max_queue_length=max_queue,
        congestion_index=congestion_index,
        total_travel_time=total_travel,
        total_waiting_time=total_wait,
        avg_route_length=avg_route_len,
        throughput_per_sec=throughput,
        normalized_throughput=normalized_throughput,
        efficiency=efficiency,
    )


def compute_efficiency(
    completion_rate: float,
    normalized_throughput: float,
    avg_waiting_time: float,
    congestion_index: float,
) -> float:
    """
    Section 12: Intersection Efficiency.

    efficiency = w1*completion_rate + w2*normalized_throughput
               + w3*wait_score      + w4*queue_score

    Every term is already bounded to [0, 1], so the weighted sum is also
    bounded to [0, 1] as long as the weights sum to 1. HIGHER is always
    BETTER and the meaning of each term is unambiguous:

      - completion_rate     : did generated vehicles actually get through?
                               (1.0 = every vehicle reached its destination)
      - normalized_throughput: how close to the maximum possible service
                               rate (= total demand) the design achieved
      - wait_score           : 1 / (1 + avg_wait / REFERENCE_WAIT)
                               -> 1.0 for ~zero waiting, ~0.5 at the
                               reference wait, approaching 0 as waiting
                               grows without bound
      - queue_score          : same shape, but for queue length per approach

    completion_rate and throughput are weighted highest (0.35 + 0.25 = 0.60)
    because a design that cannot actually move vehicles through the network
    has failed at its primary job, regardless of how short any individual
    wait looked. wait_score and queue_score (0.25 + 0.15 = 0.40) capture
    the quality of service experienced by drivers who DID get through.
    """
    wait_score = 1.0 / (1.0 + avg_waiting_time / cfg.EFFICIENCY_REFERENCE_WAIT)
    queue_score = 1.0 / (1.0 + congestion_index / cfg.EFFICIENCY_REFERENCE_QUEUE)

    w = cfg.EFFICIENCY_WEIGHTS
    score = (
        w["completion_rate"] * completion_rate
        + w["throughput"] * normalized_throughput
        + w["wait_score"] * wait_score
        + w["queue_score"] * queue_score
    )
    return score


# ---------------------------------------------------------------------------
@dataclass
class AggregatedMetrics:
    """Section 14: mean / min / max / std across NUMBER_OF_RUNS seeds."""

    design_key: str
    design_display_name: str
    n_runs: int
    stats: dict[str, dict[str, float]]  # metric_name -> {mean, min, max, std}
    raw_runs: list[DerivedMetrics]


_AGGREGATE_FIELDS = [
    "completion_rate",
    "avg_waiting_time",
    "max_waiting_time",
    "avg_queue_length",
    "max_queue_length",
    "congestion_index",
    "throughput_per_sec",
    "normalized_throughput",
    "efficiency",
    "vehicles_completed",
    "avg_route_length",
]


def aggregate_runs(runs: list[DerivedMetrics]) -> AggregatedMetrics:
    assert runs, "aggregate_runs() called with an empty list"
    stats = {}
    for field_name in _AGGREGATE_FIELDS:
        values = [getattr(r, field_name) for r in runs]
        stats[field_name] = {
            "mean": st.mean(values),
            "min": min(values),
            "max": max(values),
            "std": st.pstdev(values) if len(values) > 1 else 0.0,
        }
    return AggregatedMetrics(
        design_key=runs[0].design_key,
        design_display_name=runs[0].design_display_name,
        n_runs=len(runs),
        stats=stats,
        raw_runs=runs,
    )


def find_bottleneck_edges(
    result: SimulationResult, top_n: int = 3
) -> list[tuple[tuple[str, str], float]]:
    """Section 10 / Experiment D: edges ranked by saturation (occupancy
    relative to their own capacity*time budget) -- the closer to 1.0,
    the more that road segment was a limiting factor."""
    ranked = sorted(result.edge_saturation.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_n]
