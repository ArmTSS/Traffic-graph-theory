"""
graph_model.py
==============
This is where "graph theory" actually lives (Section 2 / Section 10).

Each intersection design is represented as a directed graph:
    - vertices  = intersection points / approach queue points / exit points
    - edges     = physical road segments (wrapped as Road objects, road.py)
    - weights   = travel_time (seconds to cross), used for Dijkstra routing

We use networkx as the graph engine for the pure graph-theory analysis
(shortest paths, centrality, connectivity) and keep our own Road objects
for the simulation state (capacity, vehicles currently on the segment).
"""

from typing import Set
import networkx as nx

from road import Road


class IntersectionNetwork:
    """A directed graph representing one intersection design."""

    def __init__(self, name: str):
        self.name = name
        self.nx_graph = nx.DiGraph()
        self.roads: dict[tuple[str, str], Road] = {}
        self.entry_nodes: list[str] = []  # where vehicles are generated
        self.exit_nodes: list[str] = []  # where vehicles leave the system
        self.controlled_edges: Set[tuple[str, str]] = (
            set()
        )  # gated by a TrafficController

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def add_road(
        self,
        start: str,
        end: str,
        capacity: int,
        travel_time: int,
        controlled: bool = False,
        length: float = None,
    ) -> Road:
        road = Road(
            start=start,
            end=end,
            capacity=capacity,
            travel_time=travel_time,
            controlled=controlled,
            length=length,
        )
        self.roads[(start, end)] = road
        self.nx_graph.add_edge(
            start, end, weight=travel_time, capacity=capacity, controlled=controlled
        )
        if controlled:
            self.controlled_edges.add((start, end))
        return road

    def set_entry_nodes(self, nodes: list[str]):
        self.entry_nodes = list(nodes)
        for n in nodes:
            self.nx_graph.add_node(n)

    def set_exit_nodes(self, nodes: list[str]):
        self.exit_nodes = list(nodes)
        for n in nodes:
            self.nx_graph.add_node(n)

    # ------------------------------------------------------------------
    # routing (Section 8)
    # ------------------------------------------------------------------
    def shortest_path_dijkstra(self, source: str, target: str) -> list[str]:
        """
        Weighted shortest path using travel_time as edge weight.
        This is the DEFAULT routing algorithm for this project because our
        edges do NOT all take the same amount of time to cross (e.g. the
        roundabout ring segments and the staggered link are short/long
        relative to approach roads) -- Dijkstra correctly finds the path
        that minimises total crossing time, not just the fewest hops.
        """
        return nx.dijkstra_path(self.nx_graph, source, target, weight="weight")

    def shortest_path_bfs(self, source: str, target: str) -> list[str]:
        """
        Unweighted shortest path (fewest hops), included for completeness /
        to show the special case where all edge weights are treated equal.
        Appropriate ONLY when every edge is assumed to take the same time,
        which is not true for our designs -- this is why the simulation
        uses Dijkstra by default and BFS only as an educational comparison.
        """
        return nx.shortest_path(
            self.nx_graph, source, target
        )  # BFS under the hood (unweighted)

    def is_reachable(self, source: str, target: str) -> bool:
        return nx.has_path(self.nx_graph, source, target)

    # ------------------------------------------------------------------
    # graph-theory analysis (Section 10)
    # ------------------------------------------------------------------
    def num_nodes(self) -> int:
        return self.nx_graph.number_of_nodes()

    def num_edges(self) -> int:
        return self.nx_graph.number_of_edges()

    def degree(self, node: str) -> int:
        """Total degree (in + out) of a node -- a simple proxy for how many
        conflicting traffic streams that node has to manage."""
        return self.nx_graph.in_degree(node) + self.nx_graph.out_degree(node)

    def core_node_degrees(self) -> dict[str, int]:
        """Degree of every node that is NOT a pure entry/exit stub, i.e. the
        actual intersection control points (I, I1/I2, ring nodes, ...)."""
        stubs = set(self.entry_nodes) | set(self.exit_nodes)
        return {n: self.degree(n) for n in self.nx_graph.nodes if n not in stubs}

    def average_path_length(self) -> float:
        """Average shortest-path travel time across every reachable
        (entry, exit) pair with entry-direction != exit-direction (i.e.
        every realistic origin-destination movement)."""
        lengths = []
        for o in self.entry_nodes:
            for d in self.exit_nodes:
                if self._same_direction(o, d):
                    continue
                if nx.has_path(self.nx_graph, o, d):
                    lengths.append(
                        nx.dijkstra_path_length(self.nx_graph, o, d, weight="weight")
                    )
        return sum(lengths) / len(lengths) if lengths else float("nan")

    @staticmethod
    def _same_direction(entry_node: str, exit_node: str) -> bool:
        return entry_node.split("_")[0] == exit_node.split("_")[0]

    def connectivity_report(self) -> dict[tuple[str, str], bool]:
        """Section 23: every selected destination must be reachable."""
        report = {}
        for o in self.entry_nodes:
            for d in self.exit_nodes:
                if self._same_direction(o, d):
                    continue
                report[(o, d)] = nx.has_path(self.nx_graph, o, d)
        return report

    def degree_centrality(self) -> dict[str, float]:
        return nx.degree_centrality(self.nx_graph)

    def betweenness_centrality(self) -> dict[str, float]:
        """
        Betweenness centrality = fraction of all shortest paths that pass
        through a node. In a road network this is a natural measure of how
        much traffic a node is FORCED to funnel through regardless of
        origin/destination -- high-betweenness nodes are structural
        bottleneck candidates before a single vehicle is even simulated.
        """
        return nx.betweenness_centrality(self.nx_graph, weight="weight")

    def edge_betweenness_centrality(self) -> dict[tuple[str, str], float]:
        return nx.edge_betweenness_centrality(self.nx_graph, weight="weight")

    def structural_summary(self) -> dict:
        core_degrees = self.core_node_degrees()
        return {
            "design": self.name,
            "num_nodes": self.num_nodes(),
            "num_edges": self.num_edges(),
            "core_node_degrees": core_degrees,
            "max_core_degree": max(core_degrees.values()) if core_degrees else 0,
            "avg_path_length_sec": round(self.average_path_length(), 3),
            "fully_connected": all(self.connectivity_report().values()),
            "num_entry_approaches": len(self.entry_nodes),
            "top_betweenness_node": max(
                self.betweenness_centrality().items(), key=lambda kv: kv[1]
            )[0],
        }
