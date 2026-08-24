"""
controller.py
=============
Section 9: traffic-control rules must depend on the intersection design.
Not every design uses traffic lights -- a roundabout is controlled by a
yield / gap-acceptance rule instead.

Both controllers implement the same tiny interface:

    is_allowed(edge, t) -> bool

A controller ONLY ever gates the "entry" edges of a network -- the edges
whose start node is a true traffic-generating approach (N_in, S_in, ...).
Every other edge (internal links, roundabout ring segments, departure
edges) is free-flowing and limited only by Road capacity. This mirrors
reality: a traffic signal (or a yield sign) controls entry into an
intersection, not what happens once you're already through it.
"""

from abc import ABC, abstractmethod


class TrafficController(ABC):
    @abstractmethod
    def is_allowed(self, edge: tuple[str, str], t: int) -> bool: ...

    @abstractmethod
    def describe(self) -> str: ...

    def current_phase_label(self, t: int) -> str:
        return "n/a"


class FixedTimeSignalController(TrafficController):
    """
    Classic traffic-light controller (Section 9).

    `phase_groups` is a list of groups of edges that get to move together
    (e.g. [ [N_in->I, S_in->I], [E_in->I, W_in->I] ] for a standard 2-phase
    four-way light). `green_times` gives how many seconds each phase stays
    green, in the same order. The controller cycles through the phases
    forever, i.e. it is a fixed-time signal (no adaptive/actuated logic --
    appropriate for a first research-level model and easy to reason about).
    """

    def __init__(
        self,
        phase_groups: list[list[tuple[str, str]]],
        green_times: list[int],
        phase_names: list[str] = None,
    ):
        assert len(phase_groups) == len(green_times)
        self.phase_groups = phase_groups
        self.green_times = green_times
        self.phase_names = phase_names or [
            f"PHASE_{i}" for i in range(len(phase_groups))
        ]
        self.cycle_length = sum(green_times)
        # Only edges that appear in SOME phase group are actually gated.
        # Any edge not mentioned here (e.g. a free-flowing departure edge)
        # must always be allowed -- the signal has no opinion about it.
        self.gated_edges = set()
        for group in phase_groups:
            self.gated_edges.update(group)

    def _phase_index_at(self, t: int) -> int:
        pos = t % self.cycle_length
        acc = 0
        for i, g in enumerate(self.green_times):
            acc += g
            if pos < acc:
                return i
        return len(self.green_times) - 1  # fallback, shouldn't happen

    def is_allowed(self, edge: tuple[str, str], t: int) -> bool:
        if edge not in self.gated_edges:
            return True  # not a signal-controlled edge -> always free-flowing
        phase = self._phase_index_at(t)
        return edge in self.phase_groups[phase]

    def current_phase_label(self, t: int) -> str:
        return self.phase_names[self._phase_index_at(t)]

    def describe(self) -> str:
        parts = [
            f"{name} green for {g}s"
            for name, g in zip(self.phase_names, self.green_times)
        ]
        return f"Fixed-time signal, cycle={self.cycle_length}s ({', '.join(parts)})"


class YieldController(TrafficController):
    """
    Roundabout-style controller (Section 9): there is NO red/green phase.
    Instead, each entry approach may merge at most `max_entry_per_step`
    vehicles per second, modelling the driver behaviour of yielding to
    circulating traffic and only merging when a gap appears. This is a
    structurally different control mechanism from a fixed-time signal,
    not just "a signal with a short green time".
    """

    def __init__(self, entry_edges: list[tuple[str, str]], max_entry_per_step: int = 1):
        self.entry_edges = set(entry_edges)
        self.max_entry_per_step = max_entry_per_step
        self._merges_this_step: dict[tuple[str, str], int] = {}
        self._current_t = -1

    def _reset_if_new_step(self, t: int):
        if t != self._current_t:
            self._current_t = t
            self._merges_this_step = {e: 0 for e in self.entry_edges}

    def is_allowed(self, edge: tuple[str, str], t: int) -> bool:
        if edge not in self.entry_edges:
            return True  # not a controlled edge -> free flow
        self._reset_if_new_step(t)
        return self._merges_this_step[edge] < self.max_entry_per_step

    def notify_merge(self, edge: tuple[str, str], t: int):
        """Call this once a vehicle actually merges, to consume its slot."""
        self._reset_if_new_step(t)
        if edge in self._merges_this_step:
            self._merges_this_step[edge] += 1

    def current_phase_label(self, t: int) -> str:
        return "YIELD"

    def describe(self) -> str:
        return (
            f"Yield / gap-acceptance control, max {self.max_entry_per_step} "
            f"merge(s) per approach per second"
        )
