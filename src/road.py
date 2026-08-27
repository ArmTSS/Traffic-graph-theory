"""
road.py
=======
A Road is a directed edge of the intersection graph (Section 7).
It has a capacity (how many vehicles can physically be on it at once)
and a travel_time (how many seconds it takes to cross when moving freely).

A congested edge becomes a bottleneck simply because vehicles cannot
ENTER it once it is at capacity -- they queue up at the node behind it.
"""

from dataclasses import dataclass, field

import config as cfg
from congestion import bpr_travel_time


@dataclass
class Road:
    start: str
    end: str
    capacity: int
    travel_time: int
    length: float = None  # purely descriptive / for weighted routing
    controlled: bool = False  # True if a TrafficController gates entry

    # --- live simulation state ---------------------------------------------
    vehicles_on_road: list[object] = field(default_factory=list)  # list[Vehicle]

    # --- lifetime statistics (Section 10 / 15: bottlenecks, edge utilisation)
    total_entries: int = 0  # how many vehicles ever used this edge
    occupied_step_sum: int = 0  # sum over time of len(vehicles_on_road)
    # -> average occupancy = occupied_step_sum / sim_time
    max_occupancy_seen: int = 0

    def __post_init__(self):
        if self.length is None:
            self.length = float(self.travel_time)

    @property
    def key(self) -> tuple[str, str]:
        return (self.start, self.end)

    def has_capacity(self) -> bool:
        return len(self.vehicles_on_road) < self.capacity

    @property
    def free_flow_travel_time(self) -> float:
        return float(self.travel_time)

    def congested_travel_time(self) -> int:
        return max(
            1,
            round(
                bpr_travel_time(
                    self.free_flow_travel_time,
                    len(self.vehicles_on_road),
                    self.capacity,
                    cfg.CONGESTION_ALPHA,
                    cfg.CONGESTION_BETA,
                )
            ),
        )

    def enter(self, vehicle):
        """Admit a vehicle onto this road. Caller must check has_capacity() first."""
        vehicle.advance(self.congested_travel_time())
        self.vehicles_on_road.append(vehicle)
        self.total_entries += 1

    def step(self):
        """
        Advance every vehicle currently on this road by one second.
        Returns the list of vehicles that finished crossing this step.
        """
        arrived = []
        still_on_road = []
        for v in self.vehicles_on_road:
            start, end, remaining = v.on_edge
            remaining -= 1
            if remaining <= 0:
                v.arrive_at_next_node()
                arrived.append(v)
            else:
                v.on_edge = (start, end, remaining)
                still_on_road.append(v)
        self.vehicles_on_road = still_on_road

        # record occupancy stats (measured AFTER arrivals leave, i.e. the
        # "resting" occupancy an observer would see at the end of the second)
        occ = len(self.vehicles_on_road)
        self.occupied_step_sum += occ
        self.max_occupancy_seen = max(self.max_occupancy_seen, occ)
        return arrived

    def saturation(self, sim_time: int) -> float:
        """Fraction of theoretical max occupancy-seconds actually used. 0..1ish."""
        if sim_time <= 0 or self.capacity <= 0:
            return 0.0
        return self.occupied_step_sum / (self.capacity * sim_time)

    def __repr__(self):
        return (
            f"Road({self.start}->{self.end}, cap={self.capacity}, t={self.travel_time})"
        )
