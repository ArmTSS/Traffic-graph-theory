"""Configurable congestion functions for road travel time."""


def bpr_travel_time(
    free_flow_time: float,
    volume: float,
    capacity: float,
    alpha: float = 0.15,
    beta: float = 4.0,
) -> float:
    """Return BPR travel time: T0 * (1 + alpha * (V/C)**beta)."""
    if free_flow_time < 0 or volume < 0 or capacity <= 0:
        raise ValueError(
            "free-flow time and volume must be non-negative; capacity positive"
        )
    if alpha < 0 or beta <= 0:
        raise ValueError("alpha must be non-negative and beta positive")
    return free_flow_time * (1 + alpha * (volume / capacity) ** beta)
