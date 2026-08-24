"""
vehicle.py
==========
Vehicles are simulated as real entities (Section 5), not just a formula
like `queue = incoming * time`. Every vehicle carries its own route,
computed once at creation time using a shortest-path algorithm on the
intersection graph (see graph_model.py / Section 8).
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Vehicle:
    vehicle_id: int
    origin: str  # e.g. "N_in"
    destination: str  # e.g. "S_out"
    route: list[str]  # full node path, e.g. ["N_in", "I", "S_out"]
    created_at: int  # simulation second the vehicle was generated

    # --- mutable simulation state -----------------------------------------
    current_node: str = field(init=False)
    route_index: int = field(init=False, default=0)  # index of current_node in route
    on_edge: Optional[tuple] = None  # (start, end, remaining_time) while travelling
    waiting_time: int = 0  # total seconds spent stopped/blocked
    completed: bool = False
    completion_time: Optional[int] = None

    def __post_init__(self):
        self.current_node = self.route[0]

    # ------------------------------------------------------------------
    @property
    def next_node(self) -> Optional[str]:
        """The node this vehicle is trying to reach next, or None if done."""
        if self.route_index + 1 >= len(self.route):
            return None
        return self.route[self.route_index + 1]

    @property
    def next_edge(self) -> Optional[tuple]:
        nxt = self.next_node
        if nxt is None:
            return None
        return (self.current_node, nxt)

    @property
    def route_length(self) -> int:
        """Number of edges travelled (Section 11: average route length)."""
        return len(self.route) - 1

    @property
    def travel_time(self) -> Optional[int]:
        """Total time from creation to completion (Section 11: total travel time)."""
        if self.completion_time is None:
            return None
        return self.completion_time - self.created_at

    def advance(self, remaining_time: int):
        """Vehicle has been let onto the next edge."""
        start, end = self.next_edge
        self.on_edge = (start, end, remaining_time)

    def arrive_at_next_node(self):
        """Vehicle finished crossing its current edge."""
        _, end, _ = self.on_edge
        self.on_edge = None
        self.current_node = end
        self.route_index += 1

    def __repr__(self):
        state = (
            "DONE"
            if self.completed
            else (
                f"on {self.on_edge}" if self.on_edge else f"waiting@{self.current_node}"
            )
        )
        return f"Vehicle#{self.vehicle_id}({self.origin}->{self.destination}, {state})"
