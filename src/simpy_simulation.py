"""SimPy discrete-event vehicle simulation compatible with the prototype API."""

import random
import simpy

import config as cfg
from simulation import SimulationResult, _poisson, _weighted_choice
from vehicle import Vehicle


class SimPyTrafficSimulation:
    """Run route-based vehicles as SimPy processes on capacity resources.

    The existing ``TrafficSimulation`` remains available as a transparent
    per-second reference implementation; this class is the event-driven
    implementation for experiments that need explicit SimPy processes.
    """

    def __init__(
        self,
        network,
        controller,
        destination_probs,
        demand_per_second,
        design_key,
        design_display_name,
    ):
        self.network = network
        self.controller = controller
        self.destination_probs = destination_probs
        self.demand_per_second = demand_per_second
        self.design_key = design_key
        self.design_display_name = design_display_name
        self.node_queues = {node: [] for node in network.nx_graph.nodes}
        self._all_vehicles = []
        self._next_vehicle_id = 0
        self._resources = {}

    def _monitor(
        self, env, sim_time, queue_history, approach_history, occupancy_history
    ):
        while env.now < sim_time:
            queue_history.append(sum(len(queue) for queue in self.node_queues.values()))
            for node in self.network.entry_nodes:
                approach_history[node].append(len(self.node_queues[node]))
            for edge, road in self.network.roads.items():
                occupancy_history[edge].append(len(road.vehicles_on_road))
            yield env.timeout(cfg.QUEUE_SAMPLE_INTERVAL)

    def _vehicle_process(self, env, vehicle):
        for start, end in zip(vehicle.route, vehicle.route[1:]):
            edge = (start, end)
            self.node_queues[start].append(vehicle)
            while not self.controller.is_allowed(edge, int(env.now)):
                vehicle.waiting_time += 1
                yield env.timeout(1)
            request_started = env.now
            with self._resources[edge].request() as request:
                yield request
                vehicle.waiting_time += int(env.now - request_started)
                self.node_queues[start].remove(vehicle)
                if vehicle.current_node != start:
                    vehicle.current_node = start
                road = self.network.roads[edge]
                travel_time = road.congested_travel_time()
                vehicle.advance(travel_time)
                road.total_entries += 1
                road.vehicles_on_road.append(vehicle)
                if hasattr(self.controller, "notify_merge"):
                    self.controller.notify_merge(edge, int(env.now))
                yield env.timeout(travel_time)
                vehicle.arrive_at_next_node()
                road.vehicles_on_road.remove(vehicle)
        vehicle.completed = True
        vehicle.completion_time = int(env.now)

    def _arrival_process(self, env, sim_time, rng):
        while env.now < sim_time:
            for entry_node in self.network.entry_nodes:
                direction = entry_node.split("_")[0]
                for _ in range(_poisson(self.demand_per_second.get(direction, 0), rng)):
                    destination = _weighted_choice(
                        self.destination_probs[direction], rng
                    )
                    route = self.network.shortest_path_dijkstra(
                        entry_node, f"{destination}_out"
                    )
                    vehicle = Vehicle(
                        self._next_vehicle_id,
                        entry_node,
                        f"{destination}_out",
                        route,
                        int(env.now),
                    )
                    self._next_vehicle_id += 1
                    self._all_vehicles.append(vehicle)
                    env.process(self._vehicle_process(env, vehicle))
            yield env.timeout(1)

    def run(self, sim_time: int, seed: int) -> SimulationResult:
        if sim_time <= 0:
            raise ValueError("sim_time must be positive")
        env = simpy.Environment()
        self.node_queues = {node: [] for node in self.network.nx_graph.nodes}
        self._all_vehicles = []
        self._next_vehicle_id = 0
        for road in self.network.roads.values():
            road.vehicles_on_road.clear()
            road.total_entries = 0
        self._resources = {
            edge: simpy.Resource(env, capacity=road.capacity)
            for edge, road in self.network.roads.items()
        }
        queue_history = []
        approach_history = {node: [] for node in self.network.entry_nodes}
        occupancy_history = {edge: [] for edge in self.network.roads}
        env.process(
            self._monitor(
                env, sim_time, queue_history, approach_history, occupancy_history
            )
        )
        env.process(self._arrival_process(env, sim_time, random.Random(seed)))
        env.run(until=sim_time)
        completed = [vehicle for vehicle in self._all_vehicles if vehicle.completed]
        result = SimulationResult(
            self.design_key,
            self.design_display_name,
            seed,
            sim_time,
            len(self.network.entry_nodes),
        )
        result.vehicles_generated = len(self._all_vehicles)
        result.vehicles_completed = len(completed)
        result.vehicles_remaining = (
            result.vehicles_generated - result.vehicles_completed
        )
        result.completed_waiting_times = [v.waiting_time for v in completed]
        result.completed_travel_times = [v.travel_time for v in completed]
        result.completed_route_lengths = [v.route_length for v in completed]
        result.queue_length_over_time = queue_history
        result.per_approach_queue_over_time = approach_history
        result.generated_count = result.vehicles_generated
        for edge, road in self.network.roads.items():
            result.edge_total_entries[edge] = road.total_entries
            result.edge_capacity[edge] = road.capacity
            occupancy = occupancy_history[edge]
            result.edge_saturation[edge] = (
                sum(occupancy) / (road.capacity * sim_time)
                if road.capacity and occupancy
                else 0.0
            )
        return result
