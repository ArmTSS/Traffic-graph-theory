import math
import sys
import tempfile
import unittest
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from controller import FreeFlowController, YieldController
from designs import create_roundabout
from graph_model import IntersectionNetwork
from intersection_catalog import (
    export_four_way_intersections,
    extract_four_way_intersections,
)
from osm_network import OSMNetwork, load_osm_network_file, save_osm_network
from osm_simulation import (
    estimate_road,
    generate_synthetic_od_demand,
    osm_graph_to_intersection_network,
    parse_maxspeed_kph,
    run_osm_simulation_study,
)
from replacement_experiment import (
    build_local_replacement_variants,
    build_selected_intersections_json,
)
from simulation import TrafficSimulation


def sample_osm_graph():
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "epsg:4326"
    positions = {
        1: (100.0, 13.0),
        2: (100.01, 13.0),
        3: (100.01, 13.01),
        4: (100.0, 13.01),
    }
    for node, (x, y) in positions.items():
        graph.add_node(node, x=x, y=y)
    for start, end in [(1, 2), (2, 3), (3, 4), (4, 1)]:
        graph.add_edge(
            start,
            end,
            length=100,
            highway="primary",
            lanes="2",
            maxspeed="36",
            oneway=True,
        )
        graph.add_edge(
            end,
            start,
            length=100,
            highway="primary",
            lanes="2",
            maxspeed="36",
            oneway=True,
        )
    graph.add_edge(1, 2, length=200, highway="service", maxspeed="20")
    return graph


class RoadEstimationTests(unittest.TestCase):
    def test_estimates_time_directional_lanes_and_storage_capacity(self):
        estimate = estimate_road(
            {
                "length": 100,
                "highway": "primary",
                "lanes": "2",
                "maxspeed": "36 km/h",
                "oneway": True,
            }
        )
        self.assertEqual(estimate.travel_time_s, 10)
        self.assertEqual(estimate.lanes, 2)
        self.assertEqual(estimate.capacity, 26)

    def test_parses_mph_and_uses_highway_fallback(self):
        self.assertAlmostEqual(parse_maxspeed_kph("30 mph", "residential"), 48.28, places=2)
        self.assertEqual(parse_maxspeed_kph(None, "primary"), 60.0)


class OSMConversionTests(unittest.TestCase):
    def test_roundabout_queues_at_merge_and_yields_to_circulating_traffic(self):
        network, controller, _ = create_roundabout()
        self.assertIsInstance(controller, YieldController)
        self.assertIn(("N_in", "Q_N"), network.roads)
        self.assertIn(("Q_N", "R_N"), network.controlled_edges)
        self.assertNotIn(("N_in", "R_N"), network.roads)

        north_merge = ("Q_N", "R_N")
        east_merge = ("Q_E", "R_E")
        conflict = network.roads[("R_W", "R_N")]
        conflict.vehicles_on_road.append(object())
        self.assertFalse(controller.is_allowed(north_merge, 0))
        self.assertTrue(controller.is_allowed(east_merge, 0))
        conflict.vehicles_on_road.clear()

        downstream = network.roads[("R_N", "R_E")]
        downstream.vehicles_on_road.extend(object() for _ in range(downstream.capacity))
        self.assertFalse(controller.is_allowed(north_merge, 1))
        downstream.vehicles_on_road.clear()

        self.assertTrue(controller.is_allowed(north_merge, 2))
        controller.notify_merge(north_merge, 2)
        self.assertFalse(controller.is_allowed(north_merge, 2))
        self.assertTrue(controller.is_allowed(east_merge, 2))

    def test_real_site_replacement_variants_are_connected_and_comparable(self):
        graph = nx.MultiDiGraph(crs="epsg:4326")
        graph.add_node(0, x=100.0, y=13.0)
        for node, x, y in [
            (1, 100.0, 13.01),
            (2, 100.01, 13.0),
            (3, 100.0, 12.99),
            (4, 99.99, 13.0),
        ]:
            graph.add_node(node, x=x, y=y)
            graph.add_edge(
                0,
                node,
                length=100,
                highway="secondary",
                lanes="2",
                maxspeed="40",
                oneway=False,
            )
            graph.add_edge(
                node,
                0,
                length=100,
                highway="secondary",
                lanes="2",
                maxspeed="40",
                oneway=False,
            )
        variants = build_local_replacement_variants(
            graph, 0, total_demand_rate=0.2
        )
        self.assertEqual(
            set(variants), {"four_way", "roundabout", "flyover", "underpass"}
        )
        for network, _, probabilities, demand in variants.values():
            self.assertTrue(all(network.connectivity_report().values()))
            self.assertAlmostEqual(sum(demand.values()), 0.2)
            self.assertTrue(
                all(math.isclose(sum(row.values()), 1.0) for row in probabilities.values())
            )

        flyover = variants["flyover"][0]
        underpass = variants["underpass"][0]
        self.assertEqual(
            set(flyover.nx_graph.edges(data="weight")),
            set(underpass.nx_graph.edges(data="weight")),
        )

    def test_four_way_catalog_classifies_and_exports_cardinal_cross(self):
        graph = nx.MultiDiGraph(crs="epsg:4326")
        graph.add_node(0, x=100.0, y=13.0, highway="traffic_signals")
        for node, x, y, name in [
            (1, 100.0, 13.01, "North Road"),
            (2, 100.01, 13.0, "East Road"),
            (3, 100.0, 12.99, "South Road"),
            (4, 99.99, 13.0, "West Road"),
        ]:
            graph.add_node(node, x=x, y=y)
            graph.add_edge(0, node, length=100, highway="primary", name=name)
            graph.add_edge(node, 0, length=100, highway="primary", name=name)
        table = extract_four_way_intersections(graph)
        self.assertEqual(len(table), 1)
        self.assertTrue(bool(table.iloc[0]["likely_geometric_cross"]))
        self.assertTrue(bool(table.iloc[0]["traffic_signals"]))
        self.assertAlmostEqual(table.iloc[0]["cross_score"], 1.0, places=3)

        selected = table.copy()
        selected.insert(0, "importance_rank", [1])
        selected["betweenness_centrality"] = 0.5
        selected["baseline_control"] = "assumed fixed-time signal"
        payload = build_selected_intersections_json(graph, selected)
        roads = payload["intersections"][0]["roads"]
        self.assertEqual(set(roads), {"north", "east", "south", "west"})
        self.assertEqual(roads["north"]["connected_osm_node"], "1")
        self.assertTrue(roads["north"]["travel_toward_intersection"])
        self.assertTrue(roads["north"]["travel_away_from_intersection"])

        with tempfile.TemporaryDirectory() as directory:
            result = export_four_way_intersections(
                graph, Path(directory) / "four_way"
            )
            self.assertTrue(result["csv"].is_file())
            self.assertTrue(result["geojson"].is_file())

    def test_saved_graph_round_trip_preserves_source_and_topology(self):
        graph = sample_osm_graph()
        network = OSMNetwork(graph=graph, nodes=None, edges=None, source="test source")
        with tempfile.TemporaryDirectory() as directory:
            filepath = Path(directory) / "district.graphml"
            save_osm_network(network, filepath)
            loaded = load_osm_network_file(filepath)
        self.assertEqual(loaded.source, "test source")
        self.assertEqual(loaded.graph.number_of_nodes(), graph.number_of_nodes())
        self.assertEqual(loaded.graph.number_of_edges(), graph.number_of_edges())

    def test_converts_component_and_selects_boundary_portals(self):
        network = osm_graph_to_intersection_network(
            sample_osm_graph(), portal_count=4
        )
        self.assertEqual(network.num_nodes(), 4)
        self.assertEqual(network.num_edges(), 8)
        self.assertEqual(len(network.entry_nodes), 4)
        self.assertEqual(network.roads[("osm:1", "osm:2")].travel_time, 10)
        self.assertTrue(all(network.connectivity_report().values()))

        demand, probabilities = generate_synthetic_od_demand(network, total_rate=0.8)
        self.assertAlmostEqual(sum(demand.values()), 0.8)
        self.assertTrue(all(math.isclose(sum(row.values()), 1.0) for row in probabilities.values()))

    def test_real_map_study_runs_end_to_end(self):
        for engine in ("step", "simpy"):
            with self.subTest(engine=engine):
                study = run_osm_simulation_study(
                    sample_osm_graph(),
                    name="test map",
                    sim_time=30,
                    n_runs=2,
                    total_demand_rate=0.2,
                    portal_count=4,
                    engine=engine,
                )
                self.assertEqual(study["graph_metrics"]["nodes"], 4)
                self.assertEqual(study["traffic_metrics"].n_runs, 2)
                self.assertEqual(len(study["bottlenecks"]), 5)

    def test_vehicle_only_completes_at_its_chosen_exit(self):
        network = IntersectionNetwork("pass-through portal")
        network.add_road("portal:a", "portal:b", capacity=10, travel_time=1)
        network.add_road("portal:b", "portal:c", capacity=10, travel_time=1)
        network.set_entry_nodes(["portal:a"])
        network.set_exit_nodes(["portal:b", "portal:c"])
        simulation = TrafficSimulation(
            network,
            FreeFlowController(),
            {"portal:a": {"portal:c": 1.0}},
            {"portal:a": 1.0},
            "osm",
            "test",
        )
        result = simulation.run(sim_time=5, seed=42)
        self.assertTrue(result.completed_travel_times)
        self.assertTrue(all(value >= 2 for value in result.completed_travel_times))


if __name__ == "__main__":
    unittest.main()
