"""
designs.py
==========
Section 3: five distinct intersection designs, each returning a
(IntersectionNetwork, TrafficController, destination_probabilities) tuple.

Naming convention used everywhere in this project:
    "<DIR>_in"  = the queueing point where vehicles from that direction wait
    "<DIR>_out" = the point where vehicles leave the system after crossing
    "I", "I1", "I2", "R_<DIR>" = internal graph-theory nodes with no
                                  real-world traffic of their own -- they
                                  only exist to represent structure.

Every design is built with add_road(..., controlled=True) on exactly the
edges leaving an "_in" node into the network -- those are the only edges a
TrafficController is ever allowed to gate (see controller.py docstring).
"""

from graph_model import IntersectionNetwork
from controller import FixedTimeSignalController, YieldController
import config as cfg


def _default_dest_probs(origin_dir: str, exit_dirs) -> dict[str, float]:
    """Uniform probability over every direction that isn't a U-turn back
    the way you came. Designs may override this with something specific."""
    others = [d for d in exit_dirs if d != origin_dir]
    p = 1.0 / len(others)
    return {d: p for d in others}


# ===========================================================================
# DESIGN 1 -- Standard four-way signalised intersection
# ===========================================================================
def create_four_way_intersection():
    net = IntersectionNetwork("Four-Way Intersection")
    dirs = ["N", "S", "E", "W"]
    net.set_entry_nodes([f"{d}_in" for d in dirs])
    net.set_exit_nodes([f"{d}_out" for d in dirs])

    for d in dirs:
        net.add_road(
            f"{d}_in",
            "I",
            cfg.DEFAULT_APPROACH_CAPACITY,
            cfg.DEFAULT_APPROACH_TRAVEL_TIME,
            controlled=True,
        )
        net.add_road(
            "I",
            f"{d}_out",
            cfg.DEFAULT_DEPARTURE_CAPACITY,
            cfg.DEFAULT_DEPARTURE_TRAVEL_TIME,
            controlled=False,
        )

    # 2-phase signal: opposing through-movements (N/S) share a phase,
    # opposing (E/W) share the other -- the standard real-world layout.
    phase_groups = [
        [("N_in", "I"), ("S_in", "I")],
        [("E_in", "I"), ("W_in", "I")],
    ]
    controller = FixedTimeSignalController(
        phase_groups,
        [cfg.GREEN_LIGHT_TIME, cfg.GREEN_LIGHT_TIME],
        phase_names=["NS_GREEN", "WE_GREEN"],
    )

    dest_probs = {d: _default_dest_probs(d, dirs) for d in dirs}
    return net, controller, dest_probs


# ===========================================================================
# DESIGN 2 -- Staggered / offset intersection
# ===========================================================================
def create_staggered_intersection():
    """
    The N and S legs do NOT meet the through road (W<->E) at the same
    point. Instead there are two control nodes, I1 (where W and N meet)
    and I2 (where E and S meet), joined by a short link edge in each
    direction. Any movement that needs to "cross over" (e.g. W->S, N->E)
    must traverse the extra link edge and therefore takes extra travel
    time and is exposed to an extra capacity constraint -- this is the
    real structural signature of a staggered / offset crossroads.
    """
    net = IntersectionNetwork("Staggered Intersection")
    net.set_entry_nodes(["N_in", "S_in", "E_in", "W_in"])
    net.set_exit_nodes(["N_out", "S_out", "E_out", "W_out"])

    # approaches into the two control nodes
    net.add_road(
        "W_in",
        "I1",
        cfg.DEFAULT_APPROACH_CAPACITY,
        cfg.DEFAULT_APPROACH_TRAVEL_TIME,
        controlled=True,
    )
    net.add_road(
        "N_in",
        "I1",
        cfg.DEFAULT_APPROACH_CAPACITY,
        cfg.DEFAULT_APPROACH_TRAVEL_TIME,
        controlled=True,
    )
    net.add_road(
        "E_in",
        "I2",
        cfg.DEFAULT_APPROACH_CAPACITY,
        cfg.DEFAULT_APPROACH_TRAVEL_TIME,
        controlled=True,
    )
    net.add_road(
        "S_in",
        "I2",
        cfg.DEFAULT_APPROACH_CAPACITY,
        cfg.DEFAULT_APPROACH_TRAVEL_TIME,
        controlled=True,
    )

    # the offset link, both directions, free-flowing (capacity-gated only)
    net.add_road(
        "I1",
        "I2",
        cfg.DEFAULT_LINK_CAPACITY,
        cfg.DEFAULT_LINK_TRAVEL_TIME,
        controlled=False,
    )
    net.add_road(
        "I2",
        "I1",
        cfg.DEFAULT_LINK_CAPACITY,
        cfg.DEFAULT_LINK_TRAVEL_TIME,
        controlled=False,
    )

    # direct departures (no link needed -- "same side" movements)
    net.add_road(
        "I1",
        "N_out",
        cfg.DEFAULT_DEPARTURE_CAPACITY,
        cfg.DEFAULT_DEPARTURE_TRAVEL_TIME,
        controlled=False,
    )
    net.add_road(
        "I1",
        "W_out",
        cfg.DEFAULT_DEPARTURE_CAPACITY,
        cfg.DEFAULT_DEPARTURE_TRAVEL_TIME,
        controlled=False,
    )
    net.add_road(
        "I2",
        "S_out",
        cfg.DEFAULT_DEPARTURE_CAPACITY,
        cfg.DEFAULT_DEPARTURE_TRAVEL_TIME,
        controlled=False,
    )
    net.add_road(
        "I2",
        "E_out",
        cfg.DEFAULT_DEPARTURE_CAPACITY,
        cfg.DEFAULT_DEPARTURE_TRAVEL_TIME,
        controlled=False,
    )

    # I1 has 2 conflicting approaches (W, N); I2 has 2 conflicting approaches (E, S)
    phase_groups = [
        [("W_in", "I1")],
        [("N_in", "I1")],
        [("E_in", "I2")],
        [("S_in", "I2")],
    ]
    half = cfg.GREEN_LIGHT_TIME // 2
    controller = FixedTimeSignalController(
        phase_groups,
        [half, half, half, half],
        phase_names=["I1_W_GREEN", "I1_N_GREEN", "I2_E_GREEN", "I2_S_GREEN"],
    )

    dirs = ["N", "S", "E", "W"]
    dest_probs = {d: _default_dest_probs(d, dirs) for d in dirs}
    return net, controller, dest_probs


# ===========================================================================
# DESIGN 3 -- T-intersection (no south leg)
# ===========================================================================
def create_t_intersection():
    net = IntersectionNetwork("T-Intersection")
    dirs = ["N", "E", "W"]
    net.set_entry_nodes([f"{d}_in" for d in dirs])
    net.set_exit_nodes([f"{d}_out" for d in dirs])

    for d in dirs:
        net.add_road(
            f"{d}_in",
            "I",
            cfg.DEFAULT_APPROACH_CAPACITY,
            cfg.DEFAULT_APPROACH_TRAVEL_TIME,
            controlled=True,
        )
        net.add_road(
            "I",
            f"{d}_out",
            cfg.DEFAULT_DEPARTURE_CAPACITY,
            cfg.DEFAULT_DEPARTURE_TRAVEL_TIME,
            controlled=False,
        )

    # W<->E is the continuous through road (non-conflicting with itself),
    # N is the terminating leg and gets its own phase because it conflicts
    # with the through movement.
    phase_groups = [
        [("W_in", "I"), ("E_in", "I")],
        [("N_in", "I")],
    ]
    controller = FixedTimeSignalController(
        phase_groups,
        [cfg.GREEN_LIGHT_TIME, cfg.GREEN_LIGHT_TIME],
        phase_names=["WE_THROUGH_GREEN", "N_GREEN"],
    )

    dest_probs = {d: _default_dest_probs(d, dirs) for d in dirs}
    return net, controller, dest_probs


# ===========================================================================
# DESIGN 4 -- Roundabout
# ===========================================================================
def create_roundabout():
    """
    The ring is a directed 4-cycle R_N -> R_E -> R_S -> R_W -> R_N
    (clockwise). Vehicles merge in at the ring node matching their
    approach direction and exit at the ring node matching their
    destination direction, travelling the ring in the merge direction
    only (a real roundabout is one-way) -- so a "left turn" equivalent
    means travelling most of the way around, exactly as in real life.
    """
    net = IntersectionNetwork("Roundabout")
    dirs = ["N", "E", "S", "W"]  # clockwise order matters here
    net.set_entry_nodes([f"{d}_in" for d in dirs])
    net.set_exit_nodes([f"{d}_out" for d in dirs])

    ring_nodes = {d: f"R_{d}" for d in dirs}
    queue_nodes = {d: f"Q_{d}" for d in dirs}

    entry_edges = []
    for d in dirs:
        net.add_road(
            f"{d}_in",
            queue_nodes[d],
            cfg.DEFAULT_APPROACH_CAPACITY,
            cfg.DEFAULT_APPROACH_TRAVEL_TIME,
            controlled=False,
        )
        net.add_road(
            queue_nodes[d],
            ring_nodes[d],
            cfg.DEFAULT_LINK_CAPACITY,
            1,
            controlled=True,
        )
        entry_edges.append((queue_nodes[d], ring_nodes[d]))
        net.add_road(
            ring_nodes[d],
            f"{d}_out",
            cfg.DEFAULT_DEPARTURE_CAPACITY,
            cfg.DEFAULT_DEPARTURE_TRAVEL_TIME,
            controlled=False,
        )

    for i in range(len(dirs)):
        a, b = dirs[i], dirs[(i + 1) % len(dirs)]
        net.add_road(
            ring_nodes[a],
            ring_nodes[b],
            cfg.ROUNDABOUT_RING_CAPACITY,
            cfg.ROUNDABOUT_RING_TRAVEL_TIME,
            controlled=False,
        )

    conflict_edges = {}
    downstream_edges = {}
    for i, d in enumerate(dirs):
        previous = dirs[(i - 1) % len(dirs)]
        following = dirs[(i + 1) % len(dirs)]
        merge_edge = (queue_nodes[d], ring_nodes[d])
        conflict_edges[merge_edge] = (ring_nodes[previous], ring_nodes[d])
        downstream_edges[merge_edge] = (ring_nodes[d], ring_nodes[following])

    controller = YieldController(
        entry_edges,
        max_entry_per_step=cfg.ROUNDABOUT_MAX_MERGE_PER_STEP,
        roads=net.roads,
        conflict_edges=conflict_edges,
        downstream_edges=downstream_edges,
        critical_occupancy=cfg.ROUNDABOUT_CRITICAL_GAP_OCCUPANCY,
    )

    dest_probs = {d: _default_dest_probs(d, dirs) for d in dirs}
    return net, controller, dest_probs


# ===========================================================================
# DESIGN 5 -- Five-way intersection (optional additional design)
# ===========================================================================
def create_five_way_intersection():
    """
    A fifth "NE" leg is added to the standard four-way. This increases the
    degree of the central node from 8 (4 in + 4 out) to 10, and -- more
    importantly for traffic flow -- it cannot be paired opposite anything,
    so it needs its OWN signal phase. Going from 2 phases to 3 phases means
    every approach gets a smaller share of green time per cycle even
    though the physical roads are otherwise identical to the four-way
    design. This isolates "how many conflicting streams meet at a node"
    as a variable, independent of demand.
    """
    net = IntersectionNetwork("Five-Way Intersection")
    dirs = ["N", "S", "E", "W", "NE"]
    net.set_entry_nodes([f"{d}_in" for d in dirs])
    net.set_exit_nodes([f"{d}_out" for d in dirs])

    for d in dirs:
        net.add_road(
            f"{d}_in",
            "I",
            cfg.DEFAULT_APPROACH_CAPACITY,
            cfg.DEFAULT_APPROACH_TRAVEL_TIME,
            controlled=True,
        )
        net.add_road(
            "I",
            f"{d}_out",
            cfg.DEFAULT_DEPARTURE_CAPACITY,
            cfg.DEFAULT_DEPARTURE_TRAVEL_TIME,
            controlled=False,
        )

    third = cfg.GREEN_LIGHT_TIME  # each phase still gets the configured green duration
    phase_groups = [
        [("N_in", "I"), ("S_in", "I")],
        [("E_in", "I"), ("W_in", "I")],
        [("NE_in", "I")],
    ]
    controller = FixedTimeSignalController(
        phase_groups,
        [third, third, third],
        phase_names=["NS_GREEN", "WE_GREEN", "NE_GREEN"],
    )

    dest_probs = {d: _default_dest_probs(d, dirs) for d in dirs}
    return net, controller, dest_probs


# ===========================================================================
DESIGN_REGISTRY = {
    "four_way": create_four_way_intersection,
    "t_intersection": create_t_intersection,
    "staggered": create_staggered_intersection,
    "roundabout": create_roundabout,
    "five_way": create_five_way_intersection,
}

DESIGN_DISPLAY_NAMES = {
    "four_way": "Four-Way",
    "t_intersection": "T-Intersection",
    "staggered": "Staggered",
    "roundabout": "Roundabout",
    "five_way": "Five-Way",
}
