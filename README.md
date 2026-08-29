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

Run the ten-intersection, four-design comparison with the compact entry point:

```powershell
uv run python seperate_4way_compare.py
```

The compact command automatically runs low (`0.5` veh/s), medium (`1.0`
veh/s), and high (`2.0` veh/s) total demand. Override them with
`--low-demand`, `--medium-demand`, and `--high-demand`. Other controls are
`--sites`, `--time`, `--runs`, `--graph`, and `--output`.

The step-based simulator in `simulation.py` is retained as a transparent
reference implementation. `simpy_simulation.py` provides an event-driven
alternative using capacity-constrained road resources and the same
`SimulationResult` contract. Use `SimPyTrafficSimulation` in an experiment
when explicit SimPy processes are required.

## Run a real Bangkok road network

Use a coordinate and radius for a repeatable study area:

```powershell
uv run python src/main.py --osm-simulate --quick `
  --osm-latitude 13.7563 --osm-longitude 100.5018 --osm-radius 1000
```

Or select a named Bangkok district:

```powershell
uv run python src/main.py --osm-simulate --quick `
  --osm-place "Pathum Wan, Bangkok, Thailand"
```

Download Khlong Sam Wa once and keep its reusable road graph outside `src`:

```powershell
uv run python src/main.py --osm-simulate --quick `
  --osm-place "Khlong Sam Wa, Bangkok, Thailand" `
  --osm-save-graph graph_districts/khlong_sam_wa.graphml
```

Run future simulations from the saved graph without downloading OSM again:

```powershell
uv run python src/main.py --osm-simulate `
  --osm-load-graph graph_districts/khlong_sam_wa.graphml `
  --osm-portals 6 --osm-demand-rate 0.5 `
  --sim-time 3600 --runs 3
```

GraphML preserves the directed road topology, geometry, and OSM road tags.
The project converts it into `IntersectionNetwork` and `Road` objects at run
time, so the original map remains reusable when model assumptions change.
District-scale routes are much longer than intersection routes, so use
`--sim-time` to give vehicles enough time to finish. `--quick` is intended
only to verify that downloading, conversion, and simulation work.

Export every four-branch intersection candidate from a saved district graph:

```powershell
uv run python src/main.py `
  --osm-load-graph graph_districts/khlong_sam_wa.graphml `
  --osm-export-four-way graph_districts/khlong_sam_wa_4way_intersections
```

This writes CSV and GeoJSON catalogs containing coordinates, connecting street
names and classes, in/out degree, signal tags, branch bearings, angle gaps, and
a bearing-based `likely_geometric_cross` flag. The flag is a reproducible graph
classification and should be checked against imagery for high-stakes use.

Run the controlled four-design experiment at ten important real intersections:

```powershell
uv run python src/main.py `
  --osm-load-graph graph_districts/khlong_sam_wa.graphml `
  --run-replacements --replacement-sites 10 `
  --osm-demand-rate 1.0 --sim-time 600 --runs 5 `
  --outdir output/khlong_sam_wa_replacements
```

Sites must be likely geometric crosses, fully bidirectional, and touch an
arterial-class road. They are ranked by approximate length-weighted betweenness
and selected with spatial separation. Each local variant preserves the site's
OSM approach length, free-flow time, lane-derived storage, and directionality.
Only the center changes between signalized four-way, roundabout, flyover, and
underpass. Flyover and underpass intentionally have identical traffic graphs;
distinguishing them requires non-topological assumptions such as cost, grade,
flood risk, speed, or capacity.

`selected_intersections.json` is the human-readable site catalog. Each
intersection contains approximate `north`, `east`, `south`, and `west` road
connections with the connected OSM node, street name, highway class, bearing,
and whether travel is allowed toward and away from the intersection. Compass
labels use a one-to-one minimum-angle assignment and are approximate for skewed
junctions. CSV remains available for statistical analysis and GeoJSON for GIS.
Every replacement run also writes `replacement_comparison.png`, a three-panel
line chart comparing efficiency, waiting time, and throughput at each selected
intersection. Shaded bands show one standard deviation across repeated runs.

The clean single-node candidates do not carry OSM signal tags, so fixed-time
signal control is an explicit baseline experiment assumption. Signal-tagged
district junctions are asymmetric multi-node complexes and require a later
intersection-clustering model. Repeated-run standard deviations and percentage
changes from each site's baseline are included in the result CSV.

`--quick` runs 60 simulated seconds with three seeds. Remove it for the full
configured duration and repeat count. `--osm-portals` controls how many
well-separated boundary nodes generate and receive trips, while
`--osm-demand-rate` sets total synthetic arrivals per second across the area.

The command writes `osm_traffic_metrics.csv`, `osm_graph_metrics.csv`, and
`osm_bottlenecks.csv` to the output directory. OSM mode exits after the map
study; without `--osm-simulate`, the OSM options only inspect the downloaded
network before the reference synthetic experiments run.

## OSM model assumptions

The conversion keeps the largest strongly connected drivable component and
maps every selected directed OSM edge to a `Road`. Parallel edges between the
same nodes are reduced to the fastest option because `IntersectionNetwork`
uses a directed graph rather than a directed multigraph.

- Free-flow travel time is `ceil(length / maxspeed)` in seconds.
- Missing speed limits use documented defaults by OSM highway class.
- Capacity means simultaneous vehicle storage: `floor(length * lanes / 7.5m)`.
- Two-way `lanes` tags are divided into directional lane counts.
- OSM directionality supplies one-way restrictions.
- Demand is uniform synthetic portal-to-portal demand; OSM has no traffic counts.
- Unknown traffic-signal timing is treated as free flow with road-capacity limits.

These assumptions make the model runnable and explainable, but they should be
calibrated with Bangkok counts, observed speeds, and signal plans before its
absolute predictions are treated as real-world forecasts.

## Load a local OSM network in Python

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
in a projected UTM CRS by `osm_network.py`.

## Metrics

`graph_model.py` implements weighted graph efficiency as

$$E = \frac{\sum_{i,j} d_{ij}/t_{ij}}{\sum_{i,j} d_{ij}}$$

where $d_{ij}$ is optional OD demand and $t_{ij}$ is weighted shortest-path
travel time. Without demand, all ordered reachable pairs receive equal
weight. This is separate from the traffic efficiency score in `metrics.py`.

All experiment comparisons should hold demand, seeds, duration, surrounding
network, and metric definitions constant. Geographic road structure from OSM
is real; vehicle arrivals, queues, capacity assumptions, and congestion are
simulated unless external traffic counts are explicitly supplied.
