"""Traffic-demand representations and validation helpers."""


def od_demand_to_inputs(
    od_demand: dict[tuple[str, str], float],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Convert vehicles-per-second OD demand to legacy simulator inputs."""
    if not od_demand:
        raise ValueError("od_demand cannot be empty")
    totals: dict[str, float] = {}
    probabilities: dict[str, dict[str, float]] = {}
    for (origin, destination), value in od_demand.items():
        if value < 0:
            raise ValueError("OD demand values must be non-negative")
        if origin == destination:
            raise ValueError("U-turn OD demand is not supported")
        totals[origin] = totals.get(origin, 0.0) + value
        probabilities.setdefault(origin, {})[destination] = value
    for origin, destinations in probabilities.items():
        total = totals[origin]
        if total:
            probabilities[origin] = {
                destination: value / total
                for destination, value in destinations.items()
            }
    return totals, probabilities
