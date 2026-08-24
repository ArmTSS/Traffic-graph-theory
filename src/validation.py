"""
validation.py
=============
Section 23: basic correctness checks that must hold for ANY design,
ANY demand level, and ANY random seed. These are sanity checks for a
school research project, not a full formal test-suite.
"""

from designs import DESIGN_REGISTRY
from simulation import TrafficSimulation
import config as cfg


class ValidationReport:
    def __init__(self):
        self.checks: list[tuple[str, bool, str]] = []  # (name, passed, detail)

    def add(self, name: str, passed: bool, detail: str = ""):
        self.checks.append((name, passed, detail))

    @property
    def all_passed(self) -> bool:
        return all(p for _, p, _ in self.checks)

    def print_report(self):
        print("\n=== VALIDATION REPORT (Section 23) ===")
        for name, passed, detail in self.checks:
            status = "PASS" if passed else "FAIL"
            line = f"[{status}] {name}"
            if detail:
                line += f" -- {detail}"
            print(line)
        print(
            f"\nOverall: {'ALL CHECKS PASSED' if self.all_passed else 'SOME CHECKS FAILED'}"
        )


def validate_vehicle_conservation(
    design_key: str, demand=None, sim_time=100, seed=1
) -> tuple[bool, str]:
    """generated == completed + currently_in_network, at every point we check."""
    demand = demand or cfg.DEMAND
    net, controller, dest_probs = DESIGN_REGISTRY[design_key]()
    sim = TrafficSimulation(net, controller, dest_probs, demand, design_key, design_key)
    result = sim.run(sim_time=sim_time, seed=seed)

    still_in_network = sum(len(q) for q in sim.node_queues.values()) + sum(
        len(r.vehicles_on_road) for r in net.roads.values()
    )
    ok = result.vehicles_generated == result.vehicles_completed + still_in_network
    detail = (
        f"generated={result.vehicles_generated}, completed={result.vehicles_completed}, "
        f"in_network={still_in_network}"
    )
    return ok, detail


def validate_no_negative_queues(
    design_key: str, demand=None, sim_time=100, seed=1
) -> tuple[bool, str]:
    demand = demand or cfg.DEMAND
    net, controller, dest_probs = DESIGN_REGISTRY[design_key]()
    sim = TrafficSimulation(net, controller, dest_probs, demand, design_key, design_key)
    result = sim.run(sim_time=sim_time, seed=seed)
    negative = [q for q in result.queue_length_over_time if q < 0]
    return len(negative) == 0, f"{len(negative)} negative-queue timesteps found"


def validate_reachability(design_key: str) -> tuple[bool, str]:
    net, controller, dest_probs = DESIGN_REGISTRY[design_key]()
    report = net.connectivity_report()
    unreachable = [k for k, v in report.items() if not v]
    return (
        len(unreachable) == 0,
        f"unreachable pairs: {unreachable}" if unreachable else "all pairs reachable",
    )


def validate_reproducibility(
    design_key: str, demand=None, sim_time=100, seed=1
) -> tuple[bool, str]:
    """Same seed -> identical results, run twice from a fresh network."""
    demand = demand or cfg.DEMAND
    outcomes = []
    for _ in range(2):
        net, controller, dest_probs = DESIGN_REGISTRY[design_key]()
        sim = TrafficSimulation(
            net, controller, dest_probs, demand, design_key, design_key
        )
        result = sim.run(sim_time=sim_time, seed=seed)
        outcomes.append(
            (
                result.vehicles_generated,
                result.vehicles_completed,
                tuple(result.queue_length_over_time),
            )
        )
    ok = outcomes[0] == outcomes[1]
    return (
        ok,
        "identical generated/completed/queue-curve across two runs with same seed"
        if ok
        else "MISMATCH",
    )


def validate_valid_graph_connections(design_key: str) -> tuple[bool, str]:
    """Every edge must connect two nodes that actually exist, and every
    entry node must have at least one outgoing edge (no dead ends)."""
    net, controller, dest_probs = DESIGN_REGISTRY[design_key]()
    problems = []
    for u, v in net.roads:
        if u not in net.nx_graph.nodes or v not in net.nx_graph.nodes:
            problems.append(f"edge {u}->{v} references a missing node")
    for entry in net.entry_nodes:
        if net.nx_graph.out_degree(entry) == 0:
            problems.append(f"entry node {entry} has no outgoing edge")
    return len(problems) == 0, "; ".join(
        problems
    ) if problems else "graph structurally valid"


def validate_destination_probabilities(design_key: str) -> tuple[bool, str]:
    """Every origin's destination probabilities must sum to ~1.0."""
    _, _, dest_probs = DESIGN_REGISTRY[design_key]()
    bad = []
    for origin, probs in dest_probs.items():
        total = sum(probs.values())
        if abs(total - 1.0) > 1e-6:
            bad.append(f"{origin} sums to {total}")
    return len(bad) == 0, "; ".join(bad) if bad else "all distributions sum to 1.0"


def run_all_validations(designs=None) -> ValidationReport:
    designs = designs or list(DESIGN_REGISTRY.keys())
    report = ValidationReport()
    for key in designs:
        ok, detail = validate_valid_graph_connections(key)
        report.add(f"[{key}] valid graph connections", ok, detail)

        ok, detail = validate_reachability(key)
        report.add(f"[{key}] every destination reachable", ok, detail)

        ok, detail = validate_destination_probabilities(key)
        report.add(f"[{key}] destination probabilities sum to 1", ok, detail)

        ok, detail = validate_vehicle_conservation(key)
        report.add(f"[{key}] vehicle conservation", ok, detail)

        ok, detail = validate_no_negative_queues(key)
        report.add(f"[{key}] no negative queue lengths", ok, detail)

        ok, detail = validate_reproducibility(key)
        report.add(f"[{key}] reproducible with fixed seed", ok, detail)
    return report
