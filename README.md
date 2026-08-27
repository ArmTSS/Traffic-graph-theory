# Traffic Graph Theory

Research prototype for measuring how intersection graph structure changes
traffic flow under controlled demand. The existing synthetic designs remain
the fast, offline reference experiment; the project now also provides the
building blocks for local Bangkok studies.

## Run the reference experiment

```powershell
uv sync
uv run python src/main.py --quick
```

The step-based simulator in `simulation.py` is retained as a transparent
reference implementation. `simpy_simulation.py` provides an event-driven
alternative using capacity-constrained road resources and the same
`SimulationResult` contract. Use `SimPyTrafficSimulation` in an experiment
when explicit SimPy processes are required.

## Load a local OSM network

```python
from osm_network import load_osm_network, intersection_candidates

network = load_osm_network(
    latitude=13.7563,
    longitude=100.5018,
    radius_m=1000,
)
candidates = intersection_candidates(network.graph)
```

OSM contributes road geometry, connectivity, and tagged attributes. It does
not provide the synthetic vehicle demand used by the simulator. Missing OSM
attributes must be handled with documented fallbacks by the model that maps
OSM edges into simulation roads. Meter-based intersection buffers are created
in a projected UTM CRS by `intersection_geometry.py`.

## Metrics

`efficiency.py` implements weighted graph efficiency as

$$E = \frac{\sum_{i,j} d_{ij}/t_{ij}}{\sum_{i,j} d_{ij}}$$

where $d_{ij}$ is optional OD demand and $t_{ij}$ is weighted shortest-path
travel time. Without demand, all ordered reachable pairs receive equal
weight. This is separate from the traffic efficiency score in `metrics.py`.

All experiment comparisons should hold demand, seeds, duration, surrounding
network, and metric definitions constant. Geographic road structure from OSM
is real; vehicle arrivals, queues, capacity assumptions, and congestion are
simulated unless external traffic counts are explicitly supplied.
