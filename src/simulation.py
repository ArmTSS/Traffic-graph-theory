"""
simulation.py
==============
Implements the per-second traffic flow loop described in Section 6:

  1. Generate incoming vehicles
  2. Determine which vehicles are allowed to move
  3. Determine their next node/edge (pre-computed route, Section 8)
  4. Check whether the next edge has capacity
  5. Move vehicles when possible
  6. Keep blocked vehicles waiting
  7. Increase waiting time for vehicles that cannot move
  8. Record traffic statistics
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from vehicle import Vehicle
from graph_model import IntersectionNetwork
from controller import TrafficController, YieldController


@dataclass
class SimulationResult:
    design_key: str
    design_display_name: str
    seed: int
    sim_time: int
    num_entry_approaches: int

    vehicles_generated: int = 0
    vehicles_completed: int = 0
    vehicles_remaining: int = 0

    completed_waiting_times: List[int] = field(default_factory=list)
    completed_travel_times: List[int] = field(default_factory=list)
    completed_route_lengths: List[int] = field(default_factory=list)

    queue_length_over_time: List[int] = field(default_factory=list)  # len == sim_time
    per_approach_queue_over_time: Dict[str, List[int]] = field(default_factory=dict)

    edge_total_entries: Dict[Tuple[str, str], int] = field(default_factory=dict)
    edge_saturation: Dict[Tuple[str, str], float] = field(default_factory=dict)
    edge_capacity: Dict[Tuple[str, str], int] = field(default_factory=dict)

    generated_count: int = 0  # kept for validation cross-check


class TrafficSimulation:
    def __init__(
        self,
        network: IntersectionNetwork,
        controller: TrafficController,
        destination_probs: Dict[str, Dict[str, float]],
        demand_per_second: Dict[str, float],
        design_key: str,
        design_display_name: str,
    ):
        self.network = network
        self.controller = controller
        self.destination_probs = destination_probs
        self.demand_per_second = demand_per_second
        self.design_key = design_key
        self.design_display_name = design_display_name

        # node -> list[Vehicle] currently waiting at that node for their next edge
        self.node_queues: Dict[str, List[Vehicle]] = {
            n: [] for n in network.nx_graph.nodes
        }

        self._next_vehicle_id = 0
        self._all_vehicles: List[Vehicle] = []

    def _demand_key(self, entry_node: str) -> str:
        """Support both legacy N_in labels and direct real-map node labels."""
        if entry_node in self.demand_per_second:
            return entry_node
        return entry_node.split("_")[0]

    def _destination_node(self, entry_node: str, demand_key: str, rng) -> str:
        probabilities = self.destination_probs.get(
            entry_node, self.destination_probs.get(demand_key)
        )
        if not probabilities:
            raise ValueError(f"No destination probabilities for {entry_node!r}")
        destination = _weighted_choice(probabilities, rng)
        if destination in self.network.exit_nodes:
            return destination
        return f"{destination}_out"

    # ------------------------------------------------------------------
    def _generate_vehicles(self, t: int, rng: random.Random):
        for entry_node in self.network.entry_nodes:
            demand_key = self._demand_key(entry_node)
            rate = self.demand_per_second.get(demand_key, 0)
            if rate <= 0:
                continue
            # Poisson arrivals -- captures the natural randomness of real
            # traffic instead of a smooth/deterministic `rate * t` formula.
            n_arrivals = _poisson(rate, rng)
            for _ in range(n_arrivals):
                exit_node = self._destination_node(entry_node, demand_key, rng)
                route = self.network.shortest_path_dijkstra(entry_node, exit_node)
                v = Vehicle(
                    vehicle_id=self._next_vehicle_id,
                    origin=entry_node,
                    destination=exit_node,
                    route=route,
                    created_at=t,
                )
                self._next_vehicle_id += 1
                self._all_vehicles.append(v)
                self.node_queues[entry_node].append(v)

    # ------------------------------------------------------------------
    def _process_node_queues(self, t: int):
        """Try to move every waiting vehicle onto its next edge, in FIFO
        order per node. A vehicle that can't move yet just keeps waiting."""
        for node, queue in self.node_queues.items():
            if not queue:
                continue
            still_waiting = []
            for v in queue:
                edge = v.next_edge
                if edge is None:
                    # already at destination node with nowhere further to go
                    continue
                road = self.network.roads[edge]
                allowed = self.controller.is_allowed(edge, t)
                if allowed and road.has_capacity():
                    road.enter(v)
                    if isinstance(self.controller, YieldController):
                        self.controller.notify_merge(edge, t)
                else:
                    v.waiting_time += 1
                    still_waiting.append(v)
            self.node_queues[node] = still_waiting

    # ------------------------------------------------------------------
    def _advance_roads(self, t: int):
        for road in self.network.roads.values():
            arrived = road.step()
            for v in arrived:
                if v.current_node == v.destination:
                    v.completed = True
                    v.completion_time = t + 1
                else:
                    self.node_queues[v.current_node].append(v)

    # ------------------------------------------------------------------
    def run(self, sim_time: int, seed: int) -> SimulationResult:
        rng = random.Random(seed)

        result = SimulationResult(
            design_key=self.design_key,
            design_display_name=self.design_display_name,
            seed=seed,
            sim_time=sim_time,
            num_entry_approaches=len(self.network.entry_nodes),
        )
        result.per_approach_queue_over_time = {n: [] for n in self.network.entry_nodes}

        for t in range(sim_time):
            self._generate_vehicles(t, rng)
            self._process_node_queues(t)
            self._advance_roads(t)

            total_waiting = sum(len(q) for q in self.node_queues.values())
            result.queue_length_over_time.append(total_waiting)
            for n in self.network.entry_nodes:
                result.per_approach_queue_over_time[n].append(len(self.node_queues[n]))

        # ---- final bookkeeping -------------------------------------------------
        for v in self._all_vehicles:
            if v.completed:
                result.completed_waiting_times.append(v.waiting_time)
                result.completed_travel_times.append(v.travel_time)
                result.completed_route_lengths.append(v.route_length)

        result.vehicles_generated = len(self._all_vehicles)
        result.vehicles_completed = sum(1 for v in self._all_vehicles if v.completed)
        result.vehicles_remaining = (
            result.vehicles_generated - result.vehicles_completed
        )
        result.generated_count = result.vehicles_generated

        for key, road in self.network.roads.items():
            result.edge_total_entries[key] = road.total_entries
            result.edge_saturation[key] = road.saturation(sim_time)
            result.edge_capacity[key] = road.capacity

        return result


# ---------------------------------------------------------------------------
def _poisson(rate: float, rng: random.Random) -> int:
    """Knuth's algorithm -- avoids a numpy dependency inside the hot loop."""
    if rate <= 0:
        return 0
    import math

    L = math.exp(-rate)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def _weighted_choice(prob_dict: Dict[str, float], rng: random.Random) -> str:
    keys = list(prob_dict.keys())
    weights = list(prob_dict.values())
    return rng.choices(keys, weights=weights, k=1)[0]
